/* ============================================================
   VitaFusion AI — Analysis Wizard JS
   ============================================================ */

let files    = { 1: null, 2: null };  // retinal, tongue
let ecgMode  = 'csv';                 // 'csv' | 'image'
let ecgCsvFile   = null;
let ecgImageFile = null;
let currentStep  = 1;

// ── ECG mode tabs ─────────────────────────────────────────────

function setEcgMode(mode) {
  ecgMode = mode;
  const csvDiv   = document.getElementById('ecgCsvMode');
  const imgDiv   = document.getElementById('ecgImageMode');
  const tabCsv   = document.getElementById('tabCsv');
  const tabImage = document.getElementById('tabImage');

  if (mode === 'csv') {
    csvDiv.style.display   = 'block';
    imgDiv.style.display   = 'none';
    tabCsv.style.background   = 'var(--primary)';
    tabCsv.style.color        = '#fff';
    tabImage.style.background = 'var(--surface)';
    tabImage.style.color      = 'var(--text-muted)';
  } else {
    csvDiv.style.display   = 'none';
    imgDiv.style.display   = 'block';
    tabImage.style.background = 'var(--primary)';
    tabImage.style.color      = '#fff';
    tabCsv.style.background   = 'var(--surface)';
    tabCsv.style.color        = 'var(--text-muted)';
  }
}

// ── Step navigation ───────────────────────────────────────────

function goStep(n) {
  document.getElementById('panel' + currentStep).classList.remove('active');
  currentStep = n;
  document.getElementById('panel' + n).classList.add('active');
  updateStepIndicator();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function skipStep(n) {
  files[n] = null;
  clearPreview(n);
  if (n < 3) goStep(n + 1);
}

function updateStepIndicator() {
  for (let i = 1; i <= 3; i++) {
    const el = document.getElementById('s' + i);
    el.classList.remove('active', 'done');
    if (i < currentStep)      el.classList.add('done');
    else if (i === currentStep) el.classList.add('active');
  }
}

// ── File handling ─────────────────────────────────────────────

function handleFile(step, input, subtype) {
  const file = input.files[0];
  if (!file) return;

  if (step === 1) {
    files[1] = file;
    showPreview(1, file, 'img');
  } else if (step === 2) {
    files[2] = file;
    showPreview(2, file, 'img');
  } else if (step === 3) {
    if (subtype === 'csv') {
      ecgCsvFile = file;
      showPreview3(file, 'csv');
    } else {
      ecgImageFile = file;
      showPreview3(file, 'image');
    }
  }
}

function showPreview(step, file, type) {
  const box   = document.getElementById('preview' + step);
  if (!box) return;
  box.style.display = 'flex';
  document.getElementById('previewName' + step).textContent = file.name;
  document.getElementById('previewSize' + step).textContent = formatSize(file.size);

  if (type === 'img' && file.type.startsWith('image/')) {
    const imgEl  = document.getElementById('previewImg' + step);
    const reader = new FileReader();
    reader.onload = e => { imgEl.src = e.target.result; imgEl.style.display = 'block'; };
    reader.readAsDataURL(file);
  }
}

function showPreview3(file, subtype) {
  const isImg = subtype === 'image';
  const boxId = isImg ? 'preview3img' : 'preview3csv';
  const box   = document.getElementById(boxId);
  box.style.display = 'flex';
  document.getElementById('previewName3' + subtype).textContent = file.name;
  document.getElementById('previewSize3' + subtype).textContent = formatSize(file.size);

  if (isImg && file.type.startsWith('image/')) {
    const imgEl  = document.getElementById('previewImg3img');
    const reader = new FileReader();
    reader.onload = e => { imgEl.src = e.target.result; imgEl.style.display = 'block'; };
    reader.readAsDataURL(file);
  }
}

function clearPreview(step) {
  const box = document.getElementById('preview' + step);
  if (box) box.style.display = 'none';
  const inp = document.getElementById(step === 1 ? 'retinalFile' : 'tongueFile');
  if (inp) inp.value = '';
  const imgEl = document.getElementById('previewImg' + step);
  if (imgEl) imgEl.src = '';
}

function removeFile(step, subtype) {
  if (step === 1) { files[1] = null; clearPreview(1); }
  else if (step === 2) { files[2] = null; clearPreview(2); }
  else if (step === 3) {
    if (subtype === 'csv') {
      ecgCsvFile = null;
      document.getElementById('preview3csv').style.display = 'none';
      const i = document.getElementById('ecgFile'); if (i) i.value = '';
    } else {
      ecgImageFile = null;
      document.getElementById('preview3img').style.display = 'none';
      const i = document.getElementById('ecgImageFile'); if (i) i.value = '';
      const img = document.getElementById('previewImg3img'); if (img) img.src = '';
    }
  }
}

function formatSize(bytes) {
  if (bytes < 1024)         return bytes + ' B';
  if (bytes < 1024 * 1024)  return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ── Drag & drop ───────────────────────────────────────────────

function setupDrop(dzId, step, subtype) {
  const dz = document.getElementById(dzId);
  if (!dz) return;
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragging'); });
  dz.addEventListener('dragleave', ()  => dz.classList.remove('dragging'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('dragging');
    if (e.dataTransfer.files.length) {
      const f = e.dataTransfer.files[0];
      if (step === 3) {
        if (subtype === 'csv') { ecgCsvFile = f; showPreview3(f, 'csv'); }
        else { ecgImageFile = f; showPreview3(f, 'image'); }
      } else {
        files[step] = f; showPreview(step, f, 'img');
      }
    }
  });
}

setupDrop('dropZone1',    1, null);
setupDrop('dropZone2',    2, null);
setupDrop('dropZone3csv', 3, 'csv');
setupDrop('dropZone3img', 3, 'image');

// ── Run analysis ──────────────────────────────────────────────

const PROC_STEPS = [
  'Initialising AI models…',
  'Processing retinal image…',
  'Analysing tongue colour patterns…',
  ecgMode === 'image'
    ? 'Classifying ECG image (EfficientNet-B0)…'
    : 'Decoding ECG signal…',
  'Running clinical risk models…',
  'Fusing multimodal predictions…',
  'Computing health category…',
  'Finalising your report…',
];

function showProcessing() {
  const overlay = document.getElementById('processingOverlay');
  overlay.classList.add('show');
  let i = 0;
  const stepEl   = document.getElementById('procStep');
  const interval = setInterval(() => {
    if (i < PROC_STEPS.length) stepEl.textContent = PROC_STEPS[i++];
  }, 1100);
  return interval;
}

async function runAnalysis() {
  const btn    = document.getElementById('runBtn');
  btn.disabled = true;

  // Update proc step for ECG image mode
  PROC_STEPS[3] = ecgMode === 'image'
    ? 'Classifying ECG image (EfficientNet-B0)…'
    : 'Decoding ECG signal…';

  const ticker = showProcessing();
  const fd     = new FormData();

  if (files[1])     fd.append('retinal_image',  files[1]);
  if (files[2])     fd.append('tongue_image',   files[2]);
  if (ecgCsvFile)   fd.append('ecg_file',       ecgCsvFile);
  if (ecgImageFile) fd.append('ecg_image_file', ecgImageFile);
  fd.append('ecg_type', 'normal');

  try {
    const res  = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await res.json();
    clearInterval(ticker);

    if (data.status === 'ok') {
      window.location.href = '/results';
    } else {
      document.getElementById('processingOverlay').classList.remove('show');
      btn.disabled = false;
      alert('Analysis failed: ' + (data.message || 'Unknown error'));
    }
  } catch(err) {
    clearInterval(ticker);
    document.getElementById('processingOverlay').classList.remove('show');
    btn.disabled = false;
    alert('Network error. Please check your connection and try again.');
  }
}
