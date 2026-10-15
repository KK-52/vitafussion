"""
VitaFusion AI — LLM Prompt Templates
========================================
Carefully engineered clinical report prompts for GPT-4o / Gemini / Ollama.
"""

from typing import Dict, List, Optional


def build_clinical_report_prompt(
    patient_info: Dict,
    risk_scores: Dict[str, float],
    confidence_scores: Dict[str, float],
    retinal_findings: Optional[Dict] = None,
    tongue_findings: Optional[Dict] = None,
    ecg_findings: Optional[Dict] = None,
    clinical_features: Optional[Dict] = None,
    feature_importances: Optional[Dict] = None,
    report_style: str = "clinical",  # clinical | patient-friendly | summary
) -> str:
    """
    Build the main LLM prompt for generating a health risk report.

    Args:
        patient_info:        {name, age, gender, ...}
        risk_scores:         {disease: 0-100 pct}
        confidence_scores:   {disease: 0-1}
        retinal_findings:    Dict from retinal encoder
        tongue_findings:     Dict from tongue encoder
        ecg_findings:        Dict from ECG encoder
        clinical_features:   Raw clinical values
        feature_importances: SHAP-style feature contributions
        report_style:        Output style

    Returns:
        Formatted prompt string
    """
    name   = patient_info.get("name", "Patient")
    age    = patient_info.get("age", "Unknown")
    gender = patient_info.get("gender", "Unknown")

    # ── Risk summary section ──────────────────────────────────────────────────
    risk_lines = []
    for disease, score in risk_scores.items():
        level = _risk_level(score)
        conf  = confidence_scores.get(disease, 0.7)
        risk_lines.append(
            f"  - {disease.replace('_', ' ').title()}: {score:.1f}% ({level}) "
            f"[Confidence: {conf*100:.0f}%]"
        )
    risk_summary = "\n".join(risk_lines)

    # ── Retinal section ───────────────────────────────────────────────────────
    retinal_section = ""
    if retinal_findings:
        dr_grade = retinal_findings.get("dr_grade_label", "Unknown")
        vessel   = retinal_findings.get("vessel_score", "N/A")
        macula   = retinal_findings.get("macula_score", "N/A")
        retinal_section = f"""
RETINAL (FUNDUS) IMAGE ANALYSIS:
  - Diabetic Retinopathy Grade: {dr_grade}
  - Retinal Vessel Density Score: {float(vessel):.2f}
  - Macular Risk Score: {float(macula):.2f}
  - Optic Disc Health: {float(retinal_findings.get('optic_score', 0.5)):.2f}
"""

    # ── Tongue section ────────────────────────────────────────────────────────
    tongue_section = ""
    if tongue_findings:
        tongue_class = tongue_findings.get("tongue_class", "Unknown")
        coating  = tongue_findings.get("coating_score", "N/A")
        moisture = tongue_findings.get("moisture_score", "N/A")
        tongue_section = f"""
TONGUE IMAGE ANALYSIS:
  - Tongue Classification: {tongue_class}
  - Coating Thickness Score: {float(coating):.2f}
  - Moisture Level: {float(moisture):.2f}
"""

    # ── ECG section ───────────────────────────────────────────────────────────
    ecg_section = ""
    if ecg_findings:
        arr_class = ecg_findings.get("arrhythmia_class", "Unknown")
        hr    = ecg_findings.get("heart_rate", "N/A")
        hrv   = ecg_findings.get("hrv_ms", "N/A")
        pr    = ecg_findings.get("pr_interval", "N/A")
        qrs   = ecg_findings.get("qrs_duration", "N/A")
        qtc   = ecg_findings.get("qtc_interval", "N/A")
        ecg_section = f"""
ECG SIGNAL ANALYSIS:
  - Rhythm Classification: {arr_class}
  - Heart Rate: {hr:.0f} bpm
  - HRV (SDNN): {hrv:.1f} ms
  - PR Interval: {pr:.0f} ms
  - QRS Duration: {qrs:.0f} ms
  - QTc Interval: {qtc:.0f} ms
"""

    # ── Clinical section ──────────────────────────────────────────────────────
    clinical_section = ""
    if clinical_features:
        clinical_section = "\nKEY CLINICAL VALUES:\n"
        important_fields = [
            ("age", "Age", "years"),
            ("gender", "Gender", ""),
            ("bmi", "BMI", "kg/m²"),
            ("systolic_bp", "Systolic BP", "mmHg"),
            ("diastolic_bp", "Diastolic BP", "mmHg"),
            ("heart_rate", "Heart Rate", "bpm"),
            ("fasting_glucose", "Fasting Glucose", "mg/dL"),
            ("hba1c", "HbA1c", "%"),
            ("total_cholesterol", "Total Cholesterol", "mg/dL"),
            ("hdl_cholesterol", "HDL Cholesterol", "mg/dL"),
            ("triglycerides", "Triglycerides", "mg/dL"),
            ("smoking", "Smoking Status", ""),
            ("physical_activity", "Physical Activity", ""),
            ("diabetes_history", "Diabetes History", ""),
            ("family_heart_disease", "Family Hx Heart Disease", ""),
        ]
        for key, label, unit in important_fields:
            val = clinical_features.get(key)
            if val is not None:
                if isinstance(val, bool):
                    val_str = "Yes" if val else "No"
                elif isinstance(val, float):
                    val_str = f"{val:.1f}"
                else:
                    val_str = str(val)
                clinical_section += f"  - {label}: {val_str} {unit}\n"

    # ── Top risk factors ──────────────────────────────────────────────────────
    top_factors_section = ""
    if feature_importances:
        top_factors = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        factors_text = ", ".join([f"{k} ({'+' if v > 0 else ''}{v:.2f})" for k, v in top_factors])
        top_factors_section = f"\nTOP RISK FACTORS (SHAP): {factors_text}\n"

    # ── Style-dependent instructions ──────────────────────────────────────────
    if report_style == "patient-friendly":
        style_instructions = """
Write in clear, compassionate language that a patient with no medical background can understand.
Avoid medical jargon. Use simple explanations and relatable analogies.
Be empathetic, supportive, and encouraging.
"""
    elif report_style == "summary":
        style_instructions = """
Write a brief 3-4 sentence executive summary suitable for a medical record.
Be concise, precise, and use appropriate clinical terminology.
"""
    else:  # clinical
        style_instructions = """
Write in professional clinical language suitable for a healthcare provider.
Use standard medical terminology. Structure your findings clearly.
Include ICD-10 relevant considerations where appropriate.
"""

    # ── Final prompt ──────────────────────────────────────────────────────────
    prompt = f"""You are VitaFusion AI, an advanced multimodal health risk assessment system. 
You have analyzed data from retinal images, tongue images, ECG signals, and clinical measurements.
Generate a comprehensive, evidence-based health risk report for the following patient.

═══════════════════════════════════════════════════════════════
PATIENT INFORMATION:
  Name:   {name}
  Age:    {age}
  Gender: {gender}
═══════════════════════════════════════════════════════════════

MULTIMODAL RISK ASSESSMENT RESULTS:
{risk_summary}

{retinal_section}{tongue_section}{ecg_section}{clinical_section}{top_factors_section}

═══════════════════════════════════════════════════════════════
REPORT INSTRUCTIONS:
{style_instructions}

Your report MUST include ALL of the following sections:

## 1. Executive Summary
   Provide a 2-3 sentence overview of the patient's overall health risk profile.

## 2. Key Findings by Modality
   Summarize what each data source (retinal, tongue, ECG, clinical data) revealed.

## 3. Disease Risk Analysis
   For each disease with elevated risk (>30%), explain:
   - The specific risk level and what it means
   - Key contributing factors identified by the AI
   - Clinical significance

## 4. Critical Alerts
   List any urgent findings that require immediate medical attention (if any).
   Use ⚠️ emoji for warnings.

## 5. Personalized Recommendations
   Provide 5-7 specific, actionable recommendations including:
   - Lifestyle modifications
   - Suggested medical tests/referrals
   - Monitoring frequency
   - Dietary advice

## 6. Preventive Healthcare Plan
   Outline a 3-month preventive plan tailored to this patient's risk profile.

## 7. Follow-up Schedule
   Recommend follow-up timeline and what to monitor.

IMPORTANT DISCLAIMER: This AI assessment is for informational and screening purposes only.
It is NOT a substitute for professional medical diagnosis or treatment.
Always conclude with a reminder to consult a qualified healthcare professional.
═══════════════════════════════════════════════════════════════
"""
    return prompt


def _risk_level(score: float) -> str:
    """Convert numeric risk score to categorical label."""
    if score < 30:   return "LOW"
    if score < 60:   return "MODERATE"
    if score < 80:   return "HIGH"
    return "CRITICAL"


def build_followup_prompt(
    previous_report: str,
    new_risk_scores: Dict[str, float],
    days_elapsed: int = 90,
) -> str:
    """Prompt for generating a follow-up comparison report."""
    return f"""You are VitaFusion AI generating a follow-up health assessment report.
{days_elapsed} days have passed since the previous assessment.

PREVIOUS FINDINGS SUMMARY:
{previous_report[:500]}...

NEW RISK SCORES:
{chr(10).join(f"  - {k}: {v:.1f}%" for k, v in new_risk_scores.items())}

Generate a focused follow-up report that:
1. Compares new results to previous findings
2. Highlights improvements or deteriorations
3. Adjusts recommendations based on progress
4. Provides encouragement for positive changes
Keep the report concise (300-400 words).
"""


def build_question_answer_prompt(
    question: str,
    patient_context: Dict,
    risk_scores: Dict[str, float],
) -> str:
    """Prompt for answering specific patient questions about their results."""
    risk_context = "\n".join(f"  - {k}: {v:.1f}%" for k, v in risk_scores.items())
    return f"""You are VitaFusion AI, a helpful health information assistant.
A patient with the following risk profile is asking a question:

Patient Context:
  Age: {patient_context.get('age', 'Unknown')}, Gender: {patient_context.get('gender', 'Unknown')}

Risk Scores:
{risk_context}

Patient Question: {question}

Provide a clear, accurate, and empathetic answer in 2-3 paragraphs.
Always remind the patient to consult their doctor for medical decisions.
Do not provide specific diagnoses or treatment recommendations.
"""
