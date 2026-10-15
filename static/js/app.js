/* ═══════════════════════════════════════════════════════════════
   VitaFusion AI — Main JavaScript
   ═══════════════════════════════════════════════════════════════ */

// ── Global State ──────────────────────────────────────────────
const VF = {
  results: null,
  patientInfo: {},
  clinicalData: {},
  selectedDemo: null,
};

// ── Sidebar Toggle ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('sidebar');
  if (toggle && sidebar) {
    toggle.addEventListener('click', () => sidebar.classList.toggle('open'));
    document.addEventListener('click', (e) => {
      if (window.innerWidth <= 900 && sidebar.classList.contains('open') &&
          !sidebar.contains(e.target) && !toggle.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }

  // Restore cached results
  const cached = sessionStorage.getItem('vf_results');
  if (cached) {
    try {
      VF.results = JSON.parse(cached);
      updateAnalysisBadge(true);
    } catch(e) {}
  }
  const cachedPatient = sessionStorage.getItem('vf_patient');
  if (cachedPatient) try { VF.patientInfo = JSON.parse(cachedPatient); } catch(e) {}
  const cachedClinical = sessionStorage.getItem('vf_clinical');
  if (cachedClinical) try { VF.clinicalData = JSON.parse(cachedClinical); } catch(e) {}
});

// ── Toast Notifications ───────────────────────────────────────
function showToast(msg, type = 'info', duration = 4000) {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
    <span class="toast-msg">${msg}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;
  container.appendChild(toast);
  setTimeout(() => { if (toast.parentElement) toast.remove(); }, duration);
}

// ── Loading Overlay ───────────────────────────────────────────
function showLoading() {
  const ov = document.getElementById('loadingOverlay');
  if (ov) ov.classList.add('visible');
  animateLoadingSteps();
}
function hideLoading() {
  const ov = document.getElementById('loadingOverlay');
  if (ov) ov.classList.remove('visible');
}
function animateLoadingSteps() {
  const steps = ['ls-clinical', 'ls-retinal', 'ls-tongue', 'ls-ecg', 'ls-fusion'];
  let i = 0;
  const interval = setInterval(() => {
    if (i > 0) {
      const prev = document.getElementById(steps[i - 1]);
      if (prev) { prev.classList.remove('active'); prev.classList.add('done'); }
    }
    if (i < steps.length) {
      const cur = document.getElementById(steps[i]);
      if (cur) cur.classList.add('active');
    } else {
      clearInterval(interval);
    }
    i++;
  }, 600);
}

// ── Analysis Badge ────────────────────────────────────────────
function updateAnalysisBadge(ready) {
  const badge = document.getElementById('analysisBadge');
  if (badge) badge.style.display = ready ? 'flex' : 'none';
  const status = document.getElementById('sidebarStatus');
  if (status) {
    const dot = status.querySelector('.status-dot');
    const txt = status.querySelector('.status-text');
    if (ready) {
      dot.className = 'status-dot done';
      txt.textContent = 'Analysis Complete';
    } else {
      dot.className = 'status-dot idle';
      txt.textContent = 'Awaiting Input';
    }
  }
}

// ── API Calls ─────────────────────────────────────────────────
async function fetchDemoPatients() {
  const res = await fetch('/api/demo-patients');
  return res.json();
}

async function fetchDemoPatient(id) {
  const res = await fetch(`/api/demo/${id}`);
  return res.json();
}

async function runAnalysis(formData) {
  showLoading();
  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: formData });
    const json = await res.json();
    if (json.status === 'ok') {
      VF.results = json.data;
      sessionStorage.setItem('vf_results', JSON.stringify(json.data));
      sessionStorage.setItem('vf_patient', JSON.stringify(VF.patientInfo));
      sessionStorage.setItem('vf_clinical', JSON.stringify(VF.clinicalData));
      updateAnalysisBadge(true);
      hideLoading();
      showToast('Analysis complete! Navigate to Risk Assessment.', 'success');
      return json.data;
    } else {
      throw new Error(json.message || 'Analysis failed');
    }
  } catch (e) {
    hideLoading();
    showToast(`Error: ${e.message}`, 'error');
    throw e;
  }
}

async function fetchGradCAM(modality, riskScore, imageB64) {
  const res = await fetch('/api/gradcam', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modality, risk_score: riskScore, image_b64: imageB64 || null }),
  });
  return res.json();
}

async function fetchSHAP(importances, baseVal, outputVal, featureVals) {
  const res = await fetch('/api/shap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      feature_importances: importances,
      base_value: baseVal,
      output_value: outputVal,
      feature_values: featureVals,
    }),
  });
  return res.json();
}

async function fetchECGPlot(ecgPreds) {
  const res = await fetch('/api/ecg-plot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ecg_preds: ecgPreds }),
  });
  return res.json();
}

async function generateReport(data) {
  const res = await fetch('/api/generate-report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return res.json();
}

// ── Gauge SVG Builder ─────────────────────────────────────────
function buildGaugeSVG(score, color) {
  // Semi-circle gauge using SVG arc
  const radius = 55;
  const circumference = Math.PI * radius; // half-circle
  const pct = Math.max(0, Math.min(100, score));
  const offset = circumference - (pct / 100) * circumference;

  return `
    <svg class="risk-gauge-svg" viewBox="0 0 140 85">
      <path class="gauge-bg" d="M 15 75 A 55 55 0 0 1 125 75" />
      <path class="gauge-fill" d="M 15 75 A 55 55 0 0 1 125 75"
            stroke="${color}"
            stroke-dasharray="${circumference}"
            stroke-dashoffset="${offset}" />
      <text class="gauge-text-val" x="70" y="65">${Math.round(score)}%</text>
    </svg>
  `;
}

// ── Risk Level ────────────────────────────────────────────────
function riskLevel(score) {
  if (score >= 80) return { label: 'CRITICAL', cls: 'level-critical' };
  if (score >= 60) return { label: 'HIGH',     cls: 'level-high' };
  if (score >= 30) return { label: 'MODERATE', cls: 'level-moderate' };
  return { label: 'LOW', cls: 'level-low' };
}

function riskColor(score) {
  if (score >= 80) return '#7C3AED';
  if (score >= 60) return '#EF4444';
  if (score >= 30) return '#F59E0B';
  return '#22C55E';
}

// ── Tab System ────────────────────────────────────────────────
function initTabs(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const btns = container.querySelectorAll('.tab-btn');
  const panels = container.querySelectorAll('.tab-panel');
  btns.forEach((btn) => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });
}

// ── Utility ───────────────────────────────────────────────────
function formatDiseaseName(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function getResults() {
  if (VF.results) return VF.results;
  const cached = sessionStorage.getItem('vf_results');
  if (cached) {
    try {
      VF.results = JSON.parse(cached);
      return VF.results;
    } catch(e) {}
  }
  return null;
}

// ── File to Base64 ────────────────────────────────────────────
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ── Image preview helper ──────────────────────────────────────
function setupImageUpload(inputId, previewId) {
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  if (!input || !preview) return;
  input.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      preview.innerHTML = `<img src="${url}" alt="Preview"/><div class="upload-success">✅ ${file.name}</div>`;
    }
  });
}
