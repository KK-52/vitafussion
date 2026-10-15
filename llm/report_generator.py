"""
VitaFusion AI — LLM Report Generator
=======================================
Supports OpenAI GPT-4o, Google Gemini, Groq (LLaMA/Mixtral), and local Ollama.
Configured via environment variables.
"""

import os
import time
import logging
from typing import Dict, Generator, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Provider Implementations ──────────────────────────────────────────────────

class OpenAIProvider:
    """OpenAI GPT-4o report generator."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.available = True
        except ImportError:
            logger.warning("openai package not installed.")
            self.available = False

    def generate(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.4) -> str:
        if not self.available:
            raise RuntimeError("OpenAI package not installed.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a clinical AI assistant for VitaFusion AI health platform."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def stream(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.4) -> Generator[str, None, None]:
        if not self.available:
            raise RuntimeError("OpenAI package not installed.")
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a clinical AI assistant for VitaFusion AI health platform."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class GeminiProvider:
    """Google Gemini report generator."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        self.model_name = model
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            self.available = True
        except ImportError:
            logger.warning("google-generativeai package not installed.")
            self.available = False

    def generate(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.4) -> str:
        if not self.available:
            raise RuntimeError("google-generativeai not installed.")
        import google.generativeai as genai
        config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = self.model.generate_content(prompt, generation_config=config)
        return response.text

    def stream(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.4) -> Generator[str, None, None]:
        if not self.available:
            raise RuntimeError("google-generativeai not installed.")
        import google.generativeai as genai
        config = genai.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
        response = self.model.generate_content(prompt, generation_config=config, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text


class OllamaProvider:
    """Local Ollama LLM provider (no API key needed)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.available = self._check_available()

    def _check_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.4) -> str:
        import requests
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")

    def stream(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.4) -> Generator[str, None, None]:
        import requests, json
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        with requests.post(f"{self.base_url}/api/generate", json=payload,
                           stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break


class GroqProvider:
    """
    Groq LLM provider — ultra-fast inference via Groq Cloud API.
    Supports: llama-3.3-70b-versatile, llama-3.1-8b-instant,
              mixtral-8x7b-32768, gemma2-9b-it.
    Free tier available at console.groq.com.
    Uses the `groq` Python SDK (openai-compatible interface).
    """

    SYSTEM_PROMPT = "You are a clinical AI assistant for VitaFusion AI health platform."

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.model_name = model
        try:
            from groq import Groq
            self.client    = Groq(api_key=api_key)
            self.available = True
        except ImportError:
            logger.warning(
                "groq package not installed. Run: pip install groq"
            )
            self.available = False

    def generate(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.4) -> str:
        if not self.available:
            raise RuntimeError("groq package not installed. Run: pip install groq")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def stream(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.4) -> Generator[str, None, None]:
        if not self.available:
            raise RuntimeError("groq package not installed. Run: pip install groq")
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Fallback Report (no LLM key) ─────────────────────────────────────────────

def generate_fallback_report(
    patient_info: Dict,
    risk_scores: Dict[str, float],
    retinal_findings: Optional[Dict] = None,
    tongue_findings: Optional[Dict] = None,
    ecg_findings: Optional[Dict] = None,
    clinical_features: Optional[Dict] = None,
) -> str:
    """
    Generate a template-based clinical report without LLM.
    Used when no LLM API key is configured.
    """
    name   = patient_info.get("name", "Patient")
    age    = patient_info.get("age", "N/A")
    gender = patient_info.get("gender", "N/A")

    def risk_label(score):
        if score < 30:   return "🟢 LOW"
        if score < 60:   return "🟡 MODERATE"
        if score < 80:   return "🔴 HIGH"
        return "🟣 CRITICAL"

    high_risks = [(d, s) for d, s in risk_scores.items() if s >= 60]
    mod_risks  = [(d, s) for d, s in risk_scores.items() if 30 <= s < 60]

    # Recommendations based on top risk
    recs = {
        "cardiovascular": "• Adopt a heart-healthy diet (DASH or Mediterranean)\n• Exercise 150 min/week\n• Monitor blood pressure regularly\n• Consider cardiologist referral",
        "diabetes":       "• Reduce refined sugars and processed carbohydrates\n• Monitor fasting glucose monthly\n• Increase physical activity\n• Schedule HbA1c test",
        "hypertension":   "• Reduce sodium intake to <2,300 mg/day\n• Manage stress with mindfulness\n• Monitor blood pressure daily\n• Discuss medication with your doctor",
        "arrhythmia":     "• Avoid stimulants (caffeine, energy drinks)\n• Schedule a Holter monitor test\n• Electrolyte balance check\n• Cardiology referral recommended",
        "metabolic":      "• Mediterranean or plant-based diet\n• Weight management program\n• Regular metabolic panel (lipids, glucose)\n• Lifestyle counseling",
    }

    top_risk = max(risk_scores, key=risk_scores.get)
    top_recs = recs.get(top_risk, "• Maintain a healthy lifestyle\n• Regular check-ups")

    # Build report
    risk_table = "\n".join([
        f"| {d.replace('_', ' ').title()} | {s:.1f}% | {risk_label(s)} |"
        for d, s in risk_scores.items()
    ])

    alerts = ""
    if high_risks:
        alerts = "### ⚠️ Critical Alerts\n"
        for d, s in high_risks:
            alerts += f"- **{d.replace('_', ' ').title()} Risk at {s:.0f}%** — Immediate medical consultation recommended.\n"

    ecg_line = ""
    if ecg_findings:
        ecg_line = f"\n**ECG:** {ecg_findings.get('arrhythmia_class', 'N/A')} | HR: {ecg_findings.get('heart_rate', 'N/A'):.0f} bpm | HRV: {ecg_findings.get('hrv_ms', 'N/A'):.1f} ms"

    retinal_line = ""
    if retinal_findings:
        retinal_line = f"\n**Retinal Analysis:** DR Grade — {retinal_findings.get('dr_grade_label', 'N/A')}"

    tongue_line = ""
    if tongue_findings:
        tongue_line = f"\n**Tongue Analysis:** Classification — {tongue_findings.get('tongue_class', 'N/A')}"

    report = f"""# 🏥 VitaFusion AI Health Risk Report

**Patient:** {name} &nbsp;|&nbsp; **Age:** {age} &nbsp;|&nbsp; **Gender:** {gender}

---

## 1. Executive Summary

This multimodal health assessment analyzed {sum([retinal_findings is not None, tongue_findings is not None, ecg_findings is not None, True])} data sources (retinal imaging, tongue imaging, ECG signals, clinical data) to generate a comprehensive risk profile.
{"Your results indicate elevated risk in the following areas: " + ", ".join(d.replace("_"," ").title() for d, _ in high_risks) + "." if high_risks else "Your overall health risk profile is within manageable ranges."}

---

## 2. Risk Score Summary

| Disease | Risk Score | Level |
|---------|-----------|-------|
{risk_table}

---

## 3. Modality Findings
{retinal_line}{tongue_line}{ecg_line}

---

{alerts}

## 4. Personalized Recommendations

Based on your highest risk area ({top_risk.replace("_"," ").title()}):

{top_recs}

**General Wellness:**
- Aim for 7–9 hours of quality sleep per night
- Stay hydrated (8 glasses of water daily)
- Schedule annual comprehensive health screenings

---

## 5. Three-Month Preventive Plan

| Month | Focus | Actions |
|-------|-------|---------|
| Month 1 | Baseline Assessment | Full blood panel, ECG follow-up, lifestyle audit |
| Month 2 | Lifestyle Intervention | Diet changes, exercise routine, stress management |
| Month 3 | Re-evaluation | Repeat key biomarkers, review progress |

---

## 6. Follow-up Schedule

- **1 week:** Consult primary care physician with this report
- **1 month:** Blood pressure and glucose monitoring
- **3 months:** Repeat relevant laboratory tests
- **6 months:** Comprehensive follow-up assessment

---

> ⚕️ **Disclaimer:** This report is generated by VitaFusion AI for informational and screening purposes only. It is NOT a substitute for professional medical diagnosis, advice, or treatment. Please consult a qualified healthcare provider before making any health decisions.
"""
    return report


# ── Main Report Generator ─────────────────────────────────────────────────────

class ReportGenerator:
    """
    Main report generation interface.
    Selects LLM provider based on configuration.
    """

    def __init__(self):
        self.provider_name = os.getenv("LLM_PROVIDER", "gemini")
        self._provider = None
        self._init_provider()

    def _init_provider(self):
        provider = self.provider_name.lower()
        if provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            if key:
                self._provider = OpenAIProvider(key, model)

        elif provider == "gemini":
            key = os.getenv("GEMINI_API_KEY", "")
            model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            if key:
                self._provider = GeminiProvider(key, model)

        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model    = os.getenv("OLLAMA_MODEL", "llama3")
            self._provider = OllamaProvider(base_url, model)

        elif provider == "groq":
            key   = os.getenv("GROQ_API_KEY", "")
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            if key:
                self._provider = GroqProvider(key, model)

    @property
    def is_available(self) -> bool:
        return self._provider is not None and getattr(self._provider, "available", True)

    def generate_report(
        self,
        prompt: str,
        patient_info: Optional[Dict] = None,
        risk_scores: Optional[Dict] = None,
        retinal_findings: Optional[Dict] = None,
        tongue_findings: Optional[Dict] = None,
        ecg_findings: Optional[Dict] = None,
        clinical_features: Optional[Dict] = None,
        stream: bool = False,
    ):
        """
        Generate health report using configured LLM provider.

        If no LLM is available, returns a template-based report.
        """
        if not self.is_available:
            # Fallback template report
            return generate_fallback_report(
                patient_info=patient_info or {},
                risk_scores=risk_scores or {},
                retinal_findings=retinal_findings,
                tongue_findings=tongue_findings,
                ecg_findings=ecg_findings,
                clinical_features=clinical_features,
            )

        try:
            if stream:
                return self._provider.stream(prompt)
            else:
                return self._provider.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return generate_fallback_report(
                patient_info=patient_info or {},
                risk_scores=risk_scores or {},
                retinal_findings=retinal_findings,
                tongue_findings=tongue_findings,
                ecg_findings=ecg_findings,
                clinical_features=clinical_features,
            )

    def test_connection(self) -> Dict:
        """Test if the configured LLM provider is reachable."""
        if not self.is_available:
            return {
                "status": "unavailable",
                "provider": self.provider_name,
                "message": "No API key configured or provider unreachable.",
            }
        try:
            result = self._provider.generate("Respond with exactly: OK", max_tokens=5)
            return {
                "status": "connected",
                "provider": self.provider_name,
                "message": f"Provider: {self.provider_name} — Connected ✓",
                "test_response": result.strip(),
            }
        except Exception as e:
            return {
                "status": "error",
                "provider": self.provider_name,
                "message": str(e),
            }
