/**
 * IDENTI-SKIN app: Home, Image workspace, Analysis results (patients and clinicians),
 * Benchmark and Research views (clinicians).
 *
 * Every number shown comes from the on-device pipeline (ondevice.js), from
 * benchmark_data.json (backend/export_app_benchmark.py: the paper's test-set results) or
 * from assets/gradcam/gradcam.json (ablation_training/final/gradcam_analysis.py).
 */

// ------------------------------------------------------------------ study definitions

// Eight diseases in model output order (models/manifest.json).
const CLASSES = ['Warts', 'Molluscum', 'Varicella', 'HFMD', 'Tinea versicolor', 'Tinea corporis', 'Tinea pedis', 'Impetigo'];

// Two morphological clusters of four diseases (paper: analytical groupings, not medical categories).
const CLUSTERS = [
  { name: 'Vesiculopapular / Eruptive', classes: ['Varicella', 'HFMD', 'Molluscum', 'Impetigo'] },
  { name: 'Papulosquamous / Verrucous', classes: ['Tinea corporis', 'Tinea versicolor', 'Warts', 'Tinea pedis'] },
];

// Pathogen categories from the paper's scope (four viral, three fungal, one bacterial).
const CATEGORY = {
  Molluscum: 'Viral', Varicella: 'Viral', HFMD: 'Viral', Warts: 'Viral',
  'Tinea corporis': 'Fungal', 'Tinea versicolor': 'Fungal', 'Tinea pedis': 'Fungal',
  Impetigo: 'Bacterial',
};

const DISPLAY_NAME = {
  Warts: 'Kulugo (Warts)', Molluscum: 'Molluscum contagiosum', Varicella: 'Bulutong-tubig (Chickenpox)',
  HFMD: 'Hand, foot and mouth disease', 'Tinea versicolor': 'An-an (Tinea versicolor)',
  'Tinea corporis': 'Buni (Ringworm)', 'Tinea pedis': "Alipunga (Athlete's foot)", Impetigo: 'Mamaso (Impetigo)',
};

// Typical appearance of each disease: reference text (literature), not measured from the photo.
// Sources cited in the paper's review: Chauhan et al. (2023), Leung et al. (2022), Rahim et al. (2025).
const MORPHOLOGY = {
  Warts: { texture: 'Rough, raised, cauliflower-like surface; tiny black dots (clotted capillaries)',
    crust: 'Usually none; thickened keratin instead', border: 'Well-circumscribed papules or plaques',
    apart: 'No central dimple, unlike Molluscum' },
  Molluscum: { texture: 'Smooth, firm, pearly dome-shaped papules', crust: 'None',
    border: 'Discrete papules, often in clusters', apart: 'Central dimple (umbilication), unlike Warts' },
  Varicella: { texture: 'Clear vesicles on a red base', crust: 'Vesicles dry into crusts; lesions at different stages',
    border: 'Scattered and widespread, mainly the trunk', apart: 'Mixed stages over the body, unlike HFMD' },
  HFMD: { texture: 'Small oval vesicles or papules with a red halo', crust: 'Rarely crusts',
    border: 'Palms, soles and mouth', apart: 'Hand, foot and mouth distribution, unlike Varicella' },
  'Tinea versicolor': { texture: 'Fine, powdery scale on flat patches', crust: 'None',
    border: 'Lighter or darker patches that merge, mainly the trunk', apart: 'Flat patches without a raised ring, unlike Buni' },
  'Tinea corporis': { texture: 'Scaly, red ring-shaped plaque', crust: 'Scale rather than crust',
    border: 'Raised active edge with central clearing', apart: 'Ring with a clear centre, unlike Mamaso' },
  'Tinea pedis': { texture: 'Scaling, peeling and white softened skin', crust: 'None; cracks (fissures) may appear',
    border: 'Between the toes or along the sole', apart: 'Location on the feet, between the toes' },
  Impetigo: { texture: 'Red erosions and small blisters', crust: 'Honey-coloured crusts',
    border: 'Irregular, spreading, often around the nose and mouth', apart: 'Honey-coloured crust, unlike Buni' },
};

// The four configurations of SOP 1 and SOP 2.
const CONFIGS = {
  A: { short: 'Baseline', text: 'Baseline YOLOv26, raw images' },
  B: { short: 'Fixed L*-CLAHE', text: 'Fixed-parameter L*-CLAHE, clip limit 2.0' },
  C: { short: 'Focal Loss', text: 'Focal Loss optimization, raw images' },
  D: { short: 'Proposed', text: 'ITA-guided adaptive L*-CLAHE + two-stage decoupled training + Focal Loss' },
};

// ITA brackets of the Phase 0 calibration and the SOP 3 skin-type groups (ITA proxy).
const BRACKET_RULE = 'Darkest: ITA < 28° · Medium: 28° ≤ ITA ≤ 41° · Lightest: ITA > 41°';
const BRACKET_EDGES = [28, 41];

// ------------------------------------------------------------------ state

const State = {
  user: null,
  tab: 'home',
  current: null,        // { source, name, result, at }
  history: [],
  compare: null,        // runCompare result for the current photo
  consistency: null,
  deviceRuntime: null,
  sessionTimings: [],
  benchModel: 'D',
  gradcamExample: 0,
  split: 50,
  showBoxes: true,
};
let BENCH = null;
let GRADCAM = null;

// ------------------------------------------------------------------ helpers

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
// Scores: small values keep two significant decimals instead of rounding to 0.0%.
const pctS = x => (x >= 0.01 ? pct(x) : x >= 0.0001 ? `${(x * 100).toFixed(2)}%` : '< 0.01%');
const num = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? '—' : Number(x).toFixed(d));
const ms = x => `${Math.round(x)} ms`;
const pText = p => (p < 0.001 ? 'p < 0.001' : `p = ${p.toFixed(3)}`);
const clusterOf = label => CLUSTERS.find(c => c.classes.includes(label));
const logit = p => { const q = Math.min(Math.max(p, 1e-12), 1 - 1e-12); return Math.log(q / (1 - q)); };
// Sigmoid values: 4 decimals, or 3 significant digits when very small.
const sig = p => (p >= 0.0001 ? p.toFixed(4) : p.toExponential(2));
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const sd = xs => (xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - mean(xs)) ** 2, 0) / (xs.length - 1)) : 0);
const isClinician = () => State.user?.role === 'clinician';

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

// ------------------------------------------------------------------ progress dialog

// Modal shown while the models run. Steps are driven by the real progress events of
// ondevice.js; each finished step shows its measured time.
const Progress = (() => {
  let job = null;
  let lastFocus = null;

  const el = {
    root: () => $('progress-dialog'), steps: () => $('progress-steps'), title: () => $('progress-title'),
    sub: () => $('progress-sub'), photo: () => $('progress-photo'), message: () => $('progress-message'),
    cancel: () => $('progress-cancel'), primary: () => $('progress-primary'),
  };

  function render() {
    if (!job) return;
    const done = job.steps.filter(s => s.state === 'done').length;
    if (!job.steps.length) { el.steps().innerHTML = ''; el.sub().textContent = ''; return; }
    el.steps().innerHTML = job.steps.map(s => `
      <li class="step ${s.state}">
        <span class="step-icon" aria-hidden="true"></span>
        <span class="step-label">${esc(s.label)}${s.hint && s.state === 'active' ? `<small>${esc(s.hint)}</small>` : ''}</span>
        <span class="step-time">${s.state === 'done' && s.ms !== undefined ? ms(s.ms) : s.state === 'active' ? 'running' : ''}</span>
      </li>`).join('') + `
      <li class="step-count" aria-hidden="true"><span style="width:${(100 * done / job.steps.length).toFixed(0)}%"></span></li>`;
    el.sub().textContent = job.finished ? job.sub : `${done} of ${job.steps.length} steps done${job.sub ? ` · ${job.sub}` : ''}`;
  }

  function trapFocus(e) {
    if (!job) return;
    if (e.key === 'Escape') { e.preventDefault(); (job.finished ? close : cancel)(); return; }
    if (e.key !== 'Tab') return;
    const buttons = [el.cancel(), el.primary()].filter(b => !b.hidden);
    if (!buttons.length) return;
    const i = buttons.indexOf(document.activeElement);
    e.preventDefault();
    buttons[(i + (e.shiftKey ? buttons.length - 1 : 1)) % buttons.length].focus();
  }

  function open({ title, sub = '', photo = null, steps }) {
    lastFocus = document.activeElement;
    job = { steps: steps.map(s => ({ ...s, state: 'pending' })), sub, cancelled: false, finished: false };
    el.title().textContent = title;
    el.photo().src = photo || '';
    el.root().querySelector('.dialog-photo').hidden = !photo;
    el.root().classList.remove('finished', 'failed', 'warn');
    el.message().hidden = true;
    el.cancel().hidden = false;
    el.cancel().textContent = 'Cancel';
    el.cancel().onclick = cancel;
    el.primary().hidden = true;
    el.root().hidden = false;
    document.addEventListener('keydown', trapFocus);
    render();
    el.cancel().focus();
    return job;
  }

  function update(id, state, time) {
    if (!job) return;
    const s = job.steps.find(x => x.id === id);
    if (!s) return;
    s.state = state;
    if (time !== undefined) s.ms = time;
    render();
  }

  function finish({ title, message, primaryLabel = 'OK', onPrimary = null, failed = false, tone = 'ok' }) {
    if (!job) return;
    job.finished = true;
    job.steps.forEach(s => { if (s.state === 'active') s.state = failed ? 'error' : 'done'; });
    const total = job.steps.reduce((a, s) => a + (s.ms || 0), 0);
    job.sub = failed ? '' : `Finished in ${ms(total)} on this device`;
    el.root().classList.add(failed ? 'failed' : 'finished');
    el.root().classList.toggle('warn', tone === 'warn');
    if (title) el.title().textContent = title;
    el.message().innerHTML = message;
    el.message().hidden = !message;
    el.cancel().hidden = true;
    el.primary().hidden = false;
    el.primary().textContent = primaryLabel;
    el.primary().onclick = () => { close(); if (onPrimary) onPrimary(); };
    render();
    el.primary().focus();
  }

  function cancel() {
    if (job) job.cancelled = true;
    close();
  }

  function close() {
    el.root().hidden = true;
    document.removeEventListener('keydown', trapFocus);
    job = null;
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  return { open, update, finish, close };
})();

const ANALYSIS_STEPS = [
  { id: 'load', label: 'Loading the model and photo', hint: 'The first analysis also loads the model.' },
  { id: 'ita', label: 'Skin mask and ITA (Phase 0)' },
  { id: 'clahe', label: 'L*-CLAHE enhancement' },
  { id: 'detect', label: 'Detecting lesions (YOLOv26, Model D)' },
  { id: 'gradcam', label: 'Grad-CAM' },
  { id: 'features', label: 'Measuring lesion features' },
];

// ------------------------------------------------------------------ analysis

async function analyzePhoto(source, name) {
  const job = Progress.open({ title: 'Analyzing on device', photo: source, steps: ANALYSIS_STEPS });
  let result;
  try {
    result = await OnDevice.runDetect(source, { onStep: (id, state, t) => { if (!job.cancelled) Progress.update(id, state, t); } });
  } catch (err) {
    if (!job.cancelled) {
      Progress.finish({ title: 'Analysis failed', failed: true,
        message: `<p>${esc(err.message || err)}</p><p>Try another photo, or take it again in natural light.</p>` });
    }
    return;
  }
  if (job.cancelled) return;

  State.current = { source, name, result, at: new Date() };
  State.compare = null;
  State.consistency = null;
  State.history.unshift(State.current);
  State.history = State.history.slice(0, 6);
  State.sessionTimings.push(result.timings);
  renderAll();

  const top = result.detections[0];
  const message = top
    ? `<p>Top prediction: <b>${esc(DISPLAY_NAME[top.label])}</b>, model confidence ${pct(top.confidence)}.</p>
       <p>${result.detections.length} lesion box(es) at or above the ${pct(result.settings.confidence, 0)} cutoff.</p>`
    : `<p><b>No lesion reached the ${pct(result.settings.confidence, 0)} confidence cutoff.</b></p>
       ${result.below_cutoff ? `<p>Strongest candidate below the cutoff: ${esc(DISPLAY_NAME[result.below_cutoff.label])}, ${pct(result.below_cutoff.confidence)} (not counted).</p>` : ''}
       <p>Retake the photo closer to the lesion, in focus and in natural light.</p>`;
  Progress.finish({ title: top ? 'Analysis complete' : 'No lesion detected', message, tone: top ? 'ok' : 'warn',
    primaryLabel: 'View result', onPrimary: () => switchTab('results') });
}

async function handleFile(file) {
  if (!file) return;
  const source = await readFileAsDataUrl(file);
  analyzePhoto(source, file.name || 'Photo');
}

function openFilePicker() { $('file-input').click(); }
function openCamera() { $('camera-input').click(); }

// ------------------------------------------------------------------ navigation

const TAB_ALIASES = { patient: 'results', panel: 'benchmark', xai: 'research' };
const CLINICIAN_TABS = ['benchmark', 'research'];

function switchTab(tab) {
  tab = TAB_ALIASES[tab] || tab;
  if (!['home', 'workspace', 'results', 'benchmark', 'research'].includes(tab)) tab = 'home';
  if (CLINICIAN_TABS.includes(tab) && !isClinician()) tab = 'results';
  State.tab = tab;
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === `tab-${tab}`));
  document.querySelectorAll('.nav-btn').forEach(b => {
    const active = b.dataset.tab === tab;
    b.classList.toggle('active', active);
    if (active) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
  });
  $('app-content').scrollTop = 0;
  window.scrollTo(0, 0);
  if (tab === 'workspace') renderWorkspace();
}

function toggleDrawer(open) { $('drawer').hidden = !open; }

// ------------------------------------------------------------------ users

function initUser() {
  try {
    const saved = localStorage.getItem('identi_skin_user');
    if (saved) State.user = { ...JSON.parse(saved), isLoggedIn: true };
  } catch (err) {
    State.user = null;
  }
  if (!State.user) State.user = { isLoggedIn: false, name: 'Guest', role: 'patient' };
  const clinician = isClinician();
  document.body.classList.toggle('role-patient', !clinician);
  document.body.classList.toggle('role-clinician', clinician);
  const initials = (State.user.name || 'G').replace(/^Dr\.\s*/, '').split(/\s+/).map(w => w[0]).join('').slice(0, 2).toUpperCase();
  $('user-avatar').textContent = initials;
  $('user-name').textContent = State.user.name;
  $('user-role').textContent = clinician ? 'Clinician / researcher view' : 'Patient view';
  $('auth-btn').textContent = State.user.isLoggedIn ? 'Sign out' : 'Sign in';
}

function authAction() {
  if (State.user?.isLoggedIn) {
    try { localStorage.removeItem('identi_skin_user'); } catch (err) { /* storage unavailable */ }
  }
  window.location.href = 'login.html';
}

// ------------------------------------------------------------------ rendering: home

function renderHome() {
  $('recent-list').innerHTML = State.history.length ? State.history.map((h, i) => {
    const top = h.result.detections[0];
    return `
      <button type="button" class="recent" onclick="openHistory(${i})">
        <img src="${h.source}" alt="">
        <span class="recent-text"><b>${top ? esc(DISPLAY_NAME[top.label]) : 'No lesion detected'}</b>
          <small>${top ? `Model confidence ${pct(top.confidence)}` : 'No box at or above the cutoff'} · ${h.at.toLocaleTimeString()}</small></span>
      </button>`;
  }).join('') : '<p class="note">No analysis yet in this session.</p>';

  const box = $('home-benchmark');
  if (!BENCH) { box.innerHTML = ''; return; }
  const d = BENCH.models.find(m => m.id === 'D');
  const a = BENCH.models.find(m => m.id === 'A');
  box.innerHTML = `
    <div class="card-head"><h2>Benchmark summary</h2><span class="tag">${BENCH.testImages} test images</span></div>
    <p>Model D (proposed) on the held-out test set: mAP@50 <b>${d.map50.toFixed(1)}%</b>, F1 <b>${d.f1.toFixed(3)}</b>,
      within-cluster errors (eruptive) <b>${d.withinEruptive.toFixed(1)}%</b>. Baseline (Model A): mAP@50 ${a.map50.toFixed(1)}%, F1 ${a.f1.toFixed(3)}.</p>
    <p class="note">${isClinician() ? 'Full results, computations and p-values: Benchmark tab.' : 'Full results are in the clinician Benchmark view.'}</p>`;
}

function openHistory(i) {
  State.current = State.history[i];
  State.compare = null;
  State.consistency = null;
  renderAll();
  switchTab('results');
}

// ------------------------------------------------------------------ rendering: workspace

function renderWorkspace() {
  const cur = State.current;
  $('workspace-empty').hidden = Boolean(cur);
  $('workspace-body').hidden = !cur;
  renderCompareKeys();
  renderCompare();
  if (!cur) return;
  const r = cur.result;
  const W = r.image.width, H = r.image.height;
  const viewer = $('viewer');
  viewer.style.aspectRatio = `${W} / ${H}`;
  viewer.style.width = `min(100%, calc(62vh * ${(W / H).toFixed(4)}))`;  // keep the photo's shape under the height limit
  $('viewer-original').src = cur.source;
  $('viewer-enhanced').src = r.enhanced_image;
  setSplit(State.split);

  const svg = $('viewer-boxes');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const font = Math.max(10, Math.round(Math.max(W, H) / 45));
  svg.innerHTML = State.showBoxes ? r.detections.map((d, i) => {
    const [x1, y1, x2, y2] = [d.box[0] * W, d.box[1] * H, d.box[2] * W, d.box[3] * H];
    const label = `${i + 1}. ${d.label} ${pct(d.confidence)}`;
    return `<g class="det${i === 0 ? ' top' : ''}">
      <rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" stroke-width="${font / 5}"></rect>
      <rect class="det-tag" x="${x1}" y="${Math.max(0, y1 - font * 1.4)}" width="${label.length * font * 0.58 + font * 0.6}" height="${font * 1.4}"></rect>
      <text x="${x1 + font * 0.3}" y="${Math.max(0, y1 - font * 1.4) + font * 1.05}" font-size="${font}">${esc(label)}</text>
    </g>`;
  }).join('') : '';

  $('workspace-boxes').innerHTML = r.detections.length ? `
    <table class="data-table">
      <thead><tr><th>#</th><th>Disease (box label)</th><th>Model confidence</th><th>Box (x1, y1, x2, y2) px</th></tr></thead>
      <tbody>${r.detections.map((d, i) => `<tr><td>${i + 1}</td><td>${esc(d.label)}</td><td>${pct(d.confidence)}</td><td class="mono">${d.box_px.join(', ')}</td></tr>`).join('')}</tbody>
    </table>
    <p class="note">Boxes with confidence ≥ ${pct(r.settings.confidence, 0)} after per-class non-maximum suppression (IoU ${r.settings.nms_iou}).
      Pixel coordinates of the analysed image (${W} × ${H}).</p>`
    : `<p class="note">No box reached the ${pct(r.settings.confidence, 0)} cutoff.${r.below_cutoff ? ` Strongest candidate below it: ${esc(r.below_cutoff.label)} ${pct(r.below_cutoff.confidence)} (not counted).` : ''}</p>`;
}

function setSplit(p) {
  State.split = p;
  $('viewer-enhanced').style.clipPath = `inset(0 0 0 ${p}%)`;
  $('viewer-split').style.left = `${p}%`;
}

function renderCompareKeys() {
  $('compare-keys').innerHTML = Object.entries(CONFIGS).map(([id, c]) => `
    <div class="model-key${id === 'D' ? ' proposed' : ''}"><b>${id}</b><span>${esc(c.text)}</span></div>`).join('');
}

const BETA_TEXT = (model, cmp) => {
  if (model.preprocessing === 'raw') return 'Raw photo (no enhancement)';
  if (model.preprocessing === 'l_clahe_fixed') return `L*-CLAHE, fixed clip limit β = ${model.beta}`;
  return `L*-CLAHE, ITA-guided clip limit β = ${model.beta}${cmp.ita === null ? ' (no ITA: Medium fallback)' : ` (${cmp.bracket} bracket)`}`;
};

function renderCompare() {
  const grid = $('compare-grid');
  const status = $('compare-status');
  const cmp = State.compare;
  if (!State.current) { status.textContent = 'Analyse a photo first.'; grid.innerHTML = ''; return; }
  if (!cmp) { status.textContent = ''; grid.innerHTML = ''; return; }
  status.textContent = 'Single-photo comparison. The comparison on the 200 test images is in the Benchmark view.';
  grid.innerHTML = cmp.models.map(m => {
    const top = m.detections[0];
    const verdict = top
      ? `<b>${esc(top.label)}</b> · model confidence ${pct(top.confidence)} · ${m.boxes} box(es)`
      : `No box ≥ ${pct(cmp.settings.confidence, 0)}${m.below_cutoff ? `; strongest below: ${esc(m.below_cutoff.label)} ${pct(m.below_cutoff.confidence)} (not counted)` : ''}`;
    return `
      <article class="compare-card${m.id === 'D' ? ' proposed' : ''}">
        <div class="compare-head"><b>Model ${m.id}</b><span>${esc(CONFIGS[m.id].short)}</span></div>
        <img src="${m.image}" alt="Model ${m.id} input image with its detections">
        <div class="compare-pre">${esc(BETA_TEXT(m, cmp))}</div>
        <div class="compare-verdict">${verdict}</div>
        <div class="compare-time">Inference ${ms(m.inference_ms)} · total ${ms(m.total_ms)}</div>
      </article>`;
  }).join('');
}

async function runComparison() {
  const cur = State.current;
  if (!cur) { $('compare-status').textContent = 'Analyse a photo first.'; return; }
  const steps = [{ id: 'ita', label: 'Skin mask and ITA (Phase 0)' },
    ...Object.entries(CONFIGS).map(([id, c]) => ({ id: `model-${id}`, label: `Model ${id}: ${c.short}` }))];
  const job = Progress.open({ title: 'Comparing models on device', photo: cur.source, steps });
  try {
    const cmp = await OnDevice.runCompare(cur.source, { onStep: (id, state, t) => { if (!job.cancelled) Progress.update(id, state, t); } });
    if (job.cancelled) return;
    State.compare = cmp;
    renderCompare();
    renderResearch();
    Progress.finish({ title: 'Comparison complete', message: '<p>Each model saw its own preprocessed image. Results are below the button.</p>' });
  } catch (err) {
    if (!job.cancelled) Progress.finish({ title: 'Comparison failed', failed: true, message: `<p>${esc(err.message || err)}</p>` });
  }
}

// ------------------------------------------------------------------ rendering: results

function scoreBars(scores, labels, highlight) {
  return labels.map(label => {
    const s = scores[CLASSES.indexOf(label)];
    return `
      <div class="bar-row${label === highlight ? ' top' : ''}">
        <div class="bar-label"><span>${esc(DISPLAY_NAME[label])}</span><b>${pctS(s)}</b></div>
        <div class="bar"><span style="width:${(s * 100).toFixed(1)}%"></span></div>
      </div>`;
  }).join('');
}

function candidateCard(det, counted) {
  const cluster = clusterOf(det.label);
  const ranked = [...cluster.classes].sort((a, b) => det.scores[CLASSES.indexOf(b)] - det.scores[CLASSES.indexOf(a)]);
  const s = l => det.scores[CLASSES.indexOf(l)];
  const outside = CLASSES.filter(c => !cluster.classes.includes(c)).sort((a, b) => s(b) - s(a))[0];
  return `
    <div class="card">
      <div class="card-head"><h2>Candidate diseases</h2><span class="tag">${counted ? 'Top box' : 'Not counted'}</span></div>
      <p class="note">Model scores of the four diseases in the <b>${esc(cluster.name)}</b> group, read from the same box.
        Each is an independent sigmoid score, so they do not add up to 100%.</p>
      ${scoreBars(det.scores, ranked, det.label)}
      <p class="explain">${esc(DISPLAY_NAME[ranked[0]])} ranked first: ${pctS(s(ranked[0]))} versus ${esc(DISPLAY_NAME[ranked[1]])} ${pctS(s(ranked[1]))}
        (difference ${((s(ranked[0]) - s(ranked[1])) * 100).toFixed(1)} percentage points).
        Highest score outside this group: ${esc(DISPLAY_NAME[outside])} ${pctS(s(outside))}.</p>
    </div>`;
}

function featureCard(r) {
  const crops = r.lesion_crops || [];
  if (!crops.length) return '';
  const W = r.image.width, H = r.image.height;
  const sign = v => (v > 0 ? `+${v}` : `${v}`);
  return `
    <div class="card">
      <div class="card-head"><h2>Lesion feature extraction</h2><span class="tag">${crops.length} lesion(s)</span></div>
      <p class="note">Each detected box, cropped from the photo, with values measured inside the box and in the ring of skin around it
        (box enlarged by 50%). These measurements describe the lesion; they are not inputs to the model.</p>
      <div class="crops">
        ${crops.map((c, i) => {
          const [lL, la, lb] = c.lesion_lab;
          const [sL, sa, sb] = c.skin_lab;
          const dL = +(lL - sL).toFixed(1), da = +(la - sa).toFixed(1), db = +(lb - sb).toFixed(1);
          const z = logit(c.confidence);
          return `
          <article class="crop">
            <div class="crop-top">
              <img src="${c.image}" alt="Lesion ${i + 1}">
              <div>
                <div class="crop-conf">${(c.confidence * 100).toFixed(2)}</div>
                <div class="crop-label">${esc(c.label)}</div>
                <div class="muted">model confidence (%)</div>
              </div>
            </div>
            <table class="kv">
              <tr><td>Box size</td><td>${c.box_px[0]} × ${c.box_px[1]} px (${(100 * c.box_px[0] * c.box_px[1] / (W * H)).toFixed(1)}% of the photo)</td></tr>
              <tr><td>Colour difference ΔE*ab</td><td>${c.delta_e}</td></tr>
              <tr><td>Δa* (redness)</td><td>${sign(da)}</td></tr>
              <tr><td>Δb* (yellowness)</td><td>${sign(db)}</td></tr>
              <tr><td>ΔL* (lightness)</td><td>${sign(dL)}</td></tr>
              <tr><td>Texture ratio</td><td>${c.roughness_ratio}</td></tr>
            </table>
            <details class="calc">
              <summary>How these were calculated</summary>
              <p><b>Confidence</b> = σ(z) = 1 / (1 + e<sup>−z</sup>) with z = ${z.toFixed(3)} → ${c.confidence.toFixed(4)}.</p>
              <p><b>CIELAB</b> means: lesion (L*, a*, b*) = (${lL}, ${la}, ${lb}); surrounding skin = (${sL}, ${sa}, ${sb}).</p>
              <p><b>ΔE*ab</b> = √(ΔL*² + Δa*² + Δb*²) = √(${dL}² + ${da}² + ${db}²) = ${c.delta_e}.</p>
              <p><b>Δa*</b> = ${la} − ${sa} = ${sign(da)} (positive = redder than the skin); <b>Δb*</b> = ${lb} − ${sb} = ${sign(db)} (positive = more yellow);
                <b>ΔL*</b> = ${lL} − ${sL} = ${sign(dL)} (negative = darker).</p>
              <p><b>Texture ratio</b> = σ(Laplacian of grey levels) in the box ÷ σ in the skin ring = ${c.lesion_rough} ÷ ${c.skin_rough} = ${c.roughness_ratio}
                (above 1 = more local intensity variation than the surrounding skin).</p>
            </details>
          </article>`;
        }).join('')}
      </div>
    </div>`;
}

function morphologyCard(label) {
  const m = MORPHOLOGY[label];
  return `
    <div class="card morphology">
      <div class="card-head"><h2>Morphological view</h2><span class="tag">Reference</span></div>
      <p class="note">Typical appearance of <b>${esc(DISPLAY_NAME[label])}</b> from the literature, to compare with the photo.
        <b>Not measured from your photo</b> and not used by the model; the measured values are in Lesion feature extraction.</p>
      <table class="kv">
        <tr><td>Surface texture</td><td>${esc(m.texture)}</td></tr>
        <tr><td>Crust and exudate</td><td>${esc(m.crust)}</td></tr>
        <tr><td>Border and distribution</td><td>${esc(m.border)}</td></tr>
        <tr><td>How to tell apart</td><td>${esc(m.apart)}</td></tr>
      </table>
      <p class="note">Sources cited in the study: Chauhan et al. (2023); Leung et al. (2022); Rahim et al. (2025).</p>
    </div>`;
}

function calculationCard(r) {
  const top = r.detections[0] || r.below_cutoff;
  const scoreTable = top ? `
    <table class="data-table">
      <thead><tr><th>Disease</th><th>z (logit)</th><th>σ(z)</th></tr></thead>
      <tbody>${CLASSES.map((c, k) => `<tr class="${c === top.label ? 'hl' : ''}"><td>${esc(c)}</td><td class="mono">${logit(top.scores[k]).toFixed(3)}</td><td class="mono">${sig(top.scores[k])}</td></tr>`).join('')}</tbody>
    </table>` : '';
  return `
    <details class="card calc-card">
      <summary>How this result was calculated</summary>
      <ol class="calc-steps">
        <li><b>Photo.</b> ${r.image.sourceWidth} × ${r.image.sourceHeight} px, reduced to ${r.image.width} × ${r.image.height} (longest side at most ${r.settings.max_side} px).</li>
        <li><b>Enhancement (Phase 0).</b> L*-CLAHE on the lightness channel only, clip limit β = ${r.beta},
          chosen from the measured skin tone with the study's calibration (β = ${r.settings.calibration.beta_high}, ${r.settings.calibration.beta_mid}, ${r.settings.calibration.beta_low}
          for darkest, medium, lightest skin)${r.ita === null ? '; the skin could not be isolated, so the medium value was used' : ''}.</li>
        <li><b>Detection.</b> Model D (YOLOv26) reads the enhanced photo at ${r.model_input[1]} × ${r.model_input[0]} px and gives every candidate box
          a score for each of the 8 diseases. Boxes whose highest score is at least ${r.settings.confidence} are kept, then overlapping boxes of the same
          disease are merged (non-maximum suppression, IoU ${r.settings.nms_iou}): <b>${r.detections.length} box(es)</b>.</li>
        ${top ? `<li><b>Scores of the ${r.detections.length ? 'top' : 'strongest (not counted)'} box.</b> Model confidence = σ(z) = 1 / (1 + e<sup>−z</sup>):
          ${scoreTable}
          The box is labelled with the disease of the highest score: <b>${esc(top.label)}</b>, σ = ${top.confidence.toFixed(4)}.</li>` : ''}
        <li><b>Top prediction.</b> ${r.detections.length
          ? `The label of the box with the highest confidence (${esc(r.detections[0].label)}). Its group (${esc(clusterOf(r.detections[0].label).name)}) and category (${CATEGORY[r.detections[0].label]}) are the study's definitions, not model outputs.`
          : 'None, because no box reached the cutoff. The strongest candidate above is shown for information only.'}</li>
      </ol>
      <p class="note"><b>Data sources:</b> model output of D.onnx (exported from ${esc(BENCH_D_WEIGHTS)}); cutoff ${r.settings.confidence} and NMS IoU ${r.settings.nms_iou}
        from models/manifest.json (Ultralytics defaults, also used for the test-set confusion matrices); clip limits from
        phase0_calibration_d2.json. Measured on this device in ${ms(r.timings.total)}.</p>
    </details>`;
}
const BENCH_D_WEIGHTS = 'best_ModelD_g1.0_f11_seed42.pt';

function renderResults() {
  const cur = State.current;
  $('results-empty').hidden = Boolean(cur);
  $('results-body').hidden = !cur;
  if (!cur) return;
  const r = cur.result;
  const top = r.detections[0];
  let html = '';
  if (top) {
    const cluster = clusterOf(top.label);
    html += `
      <div class="card prediction">
        <div class="pill ${CATEGORY[top.label].toLowerCase()}">${CATEGORY[top.label]} infection</div>
        <div class="muted">Top prediction</div>
        <h2>${esc(DISPLAY_NAME[top.label])}</h2>
        <div class="confidence"><span>Model confidence</span><b>${pct(top.confidence)}</b></div>
        <p class="note">The detector's score for this disease in its highest-confidence box (sigmoid output, 0–100%). It is not the
          probability that you have the disease. Group: ${esc(cluster.name)} (analysis grouping used in the study).</p>
      </div>`;
    html += candidateCard(top, true);
    if (r.detections.length > 1) {
      html += `
        <div class="card">
          <div class="card-head"><h2>All detected boxes</h2><span class="tag">${r.detections.length}</span></div>
          ${r.detections.slice(0, 12).map((d, i) => `
            <div class="bar-row"><div class="bar-label"><span>${i + 1}. ${esc(DISPLAY_NAME[d.label])}</span><b>${pct(d.confidence)}</b></div>
            <div class="bar"><span style="width:${(d.confidence * 100).toFixed(1)}%"></span></div></div>`).join('')}
          <p class="note">Model confidence of each box (≥ ${pct(r.settings.confidence, 0)}). Positions are shown in the Workspace.</p>
        </div>`;
    }
  } else {
    html += `
      <div class="card prediction none">
        <div class="muted">Result</div>
        <h2>No lesion detected</h2>
        <p>No box reached the ${pct(r.settings.confidence, 0)} confidence cutoff.</p>
        ${r.below_cutoff ? `<p class="note">Strongest candidate below the cutoff (not counted): <b>${esc(DISPLAY_NAME[r.below_cutoff.label])}</b>, ${pct(r.below_cutoff.confidence)}.</p>` : ''}
        <p class="note">Retake the photo closer to the lesion, in focus and in natural light, without filters.</p>
      </div>`;
    if (r.below_cutoff) html += candidateCard(r.below_cutoff, false);
  }
  html += featureCard(r);
  if (top) html += morphologyCard(top.label);
  html += calculationCard(r);
  html += `<button type="button" class="btn wide" onclick="openReport()">Print / save report</button>`;
  $('results-body').innerHTML = html;
}

// ------------------------------------------------------------------ rendering: benchmark

function selectBenchModel(id) {
  State.benchModel = id;
  renderBenchmark();
}

function renderBenchmark() {
  const body = $('bench-body');
  if (!BENCH) { $('bench-note').textContent = 'benchmark_data.json not found.'; body.innerHTML = ''; return; }
  $('bench-note').textContent = BENCH.note;
  const models = BENCH.models;
  const sel = models.find(m => m.id === State.benchModel) || models[3];
  const cell = (m, v) => `<td class="${m.id === sel.id ? 'sel' : ''}">${v}</td>`;
  const head = `<tr><th></th>${models.map(m => `<th class="${m.id === sel.id ? 'sel' : ''}">${m.id}</th>`).join('')}</tr>`;
  const ex = sel.exact;
  const pc = sel.perClass;
  const sum = key => pc.reduce((a, c) => a + c[key], 0);

  const sopMap = `
    <div class="card">
      <div class="card-head"><h2>What each SOP is answered by</h2></div>
      <table class="data-table">
        <tr><td><b>SOP 1</b> Localization of A–D: mAP@50, mAP@50–95</td><td>Section 1 below</td></tr>
        <tr><td><b>SOP 2</b> Classification of A–D: precision, recall, F1, within-cluster misclassification</td><td>Section 2</td></tr>
        <tr><td><b>SOP 3</b> ΔAP50 (D − A), Fitzpatrick I–II vs III–V (ITA proxy)</td><td>Section 3</td></tr>
        <tr><td><b>H01</b> Differences among A–D (Friedman, Wilcoxon, Bonferroni, rank-biserial)</td><td>Section 4</td></tr>
        <tr><td><b>H02</b> Skin-type difference (Mann-Whitney U)</td><td>Section 3</td></tr>
        <tr><td>System check: inference time</td><td>Section 5</td></tr>
        <tr><td>Framework on one photo</td><td>Workspace, Results, Research</td></tr>
      </table>
    </div>`;

  const keys = `
    <div class="model-keys selectable">${models.map(m => `
      <button type="button" class="model-key${m.id === sel.id ? ' selected' : ''}" onclick="selectBenchModel('${m.id}')" aria-pressed="${m.id === sel.id}">
        <b>${m.id}</b><span>${esc(CONFIGS[m.id].text)}</span></button>`).join('')}</div>
    <p class="note">Tap a configuration to show its computations, per-disease values and confusion matrix.</p>`;

  const sop1 = `
    <div class="card">
      <div class="card-head"><h2>1 · Localization (SOP 1)</h2></div>
      <table class="data-table center"><thead>${head}</thead><tbody>
        <tr><th>mAP@50</th>${models.map(m => cell(m, `${m.map50.toFixed(1)}%`)).join('')}</tr>
        <tr><th>mAP@50–95</th>${models.map(m => cell(m, `${m.map5095.toFixed(1)}%`)).join('')}</tr>
      </tbody></table>
      <h3>Per disease, Model ${sel.id}</h3>
      <table class="data-table"><thead><tr><th>Disease</th><th>Boxes</th><th>AP@50</th><th>AP@50–95</th></tr></thead><tbody>
        ${pc.map(c => `<tr><td>${esc(c.class)}</td><td>${c.instances}</td><td class="mono">${c.AP50.toFixed(4)}</td><td class="mono">${c.AP50_95.toFixed(4)}</td></tr>`).join('')}
      </tbody></table>
      <details class="calc"><summary>Computation for Model ${sel.id}</summary>
        <p>AP@50 of a disease = area under its precision–recall curve, a predicted box counting as correct when IoU ≥ 0.50 with a labelled box of the same disease.
          AP@50–95 averages AP over IoU 0.50, 0.55, …, 0.95.</p>
        <p><b>mAP@50</b> = (1/8) Σ AP@50 = (${pc.map(c => c.AP50.toFixed(4)).join(' + ')}) / 8 = ${(sum('AP50') / 8).toFixed(4)} = ${(ex.mAP50 * 100).toFixed(1)}%</p>
        <p><b>mAP@50–95</b> = (1/8) Σ AP@50–95 = ${(sum('AP50_95')).toFixed(4)} / 8 = ${(sum('AP50_95') / 8).toFixed(4)} = ${(ex.mAP50_95 * 100).toFixed(1)}%</p>
      </details>
    </div>`;

  const cl = sel.clusters;
  const sop2 = `
    <div class="card">
      <div class="card-head"><h2>2 · Classification (SOP 2)</h2></div>
      <table class="data-table center"><thead>${head}</thead><tbody>
        <tr><th>Precision</th>${models.map(m => cell(m, `${m.precision.toFixed(1)}%`)).join('')}</tr>
        <tr><th>Recall</th>${models.map(m => cell(m, `${m.recall.toFixed(1)}%`)).join('')}</tr>
        <tr><th>F1</th>${models.map(m => cell(m, m.f1.toFixed(3))).join('')}</tr>
        <tr><th>Within-cluster errors, eruptive</th>${models.map(m => cell(m, `${m.withinEruptive.toFixed(1)}%`)).join('')}</tr>
        <tr><th>Within-cluster errors, scaly/verrucous</th>${models.map(m => cell(m, `${m.withinScaly.toFixed(1)}%`)).join('')}</tr>
      </tbody></table>
      <h3>Per disease, Model ${sel.id}</h3>
      <table class="data-table"><thead><tr><th>Disease</th><th>P</th><th>R</th><th>F1</th></tr></thead><tbody>
        ${pc.map(c => `<tr><td>${esc(c.class)}</td><td class="mono">${c.precision.toFixed(4)}</td><td class="mono">${c.recall.toFixed(4)}</td><td class="mono">${c.F1.toFixed(4)}</td></tr>`).join('')}
      </tbody></table>
      <details class="calc"><summary>Computation for Model ${sel.id}</summary>
        <p>Per disease: precision = TP / (TP + FP), recall = TP / (TP + FN), at the confidence that maximizes the mean F1 (Ultralytics evaluation).</p>
        <p><b>Precision</b> = (1/8) Σ P = (${pc.map(c => c.precision.toFixed(4)).join(' + ')}) / 8 = ${ex.precision.toFixed(4)}</p>
        <p><b>Recall</b> = (1/8) Σ R = (${pc.map(c => c.recall.toFixed(4)).join(' + ')}) / 8 = ${ex.recall.toFixed(4)}</p>
        <p><b>F1</b> = 2PR / (P + R) = 2 × ${ex.precision.toFixed(4)} × ${ex.recall.toFixed(4)} / (${ex.precision.toFixed(4)} + ${ex.recall.toFixed(4)}) = ${ex.F1.toFixed(4)}</p>
        <p><b>Within-cluster misclassification rate</b> = boxes given another disease of the same cluster ÷ detected boxes of that cluster
          (confidence ≥ 0.25, IoU ≥ 0.5):</p>
        <table class="data-table"><thead><tr><th>Cluster</th><th>Labelled</th><th>Detected</th><th>Correct</th><th>Within</th><th>Cross</th><th>Rate</th></tr></thead><tbody>
          ${cl.map(c => `<tr><td>${esc(c.cluster)}</td><td>${c.gt}</td><td>${c.detected}</td><td>${c.correct}</td><td>${c.within}</td><td>${c.cross}</td><td class="mono">${c.within} / ${c.detected} = ${(100 * c.within / c.detected).toFixed(1)}%</td></tr>`).join('')}
        </tbody></table>
      </details>
      <h3>Confusion matrix, Model ${sel.id}</h3>
      ${confusionMatrix(sel)}
    </div>`;

  const ft = BENCH.fairnessTest;
  const sop3 = `
    <div class="card">
      <div class="card-head"><h2>3 · Skin-type groups (SOP 3, H02)</h2></div>
      <p class="note">Groups from the ITA of each test image (ITA proxy of Fitzpatrick type; ITA ≤ −30°, type VI, excluded: ${ft.excludedVI} images; no ITA: ${ft.excludedNoIta}).</p>
      <table class="data-table center"><thead><tr><th>Group</th><th>Images</th>${models.map(m => `<th>${m.id}</th>`).join('')}<th>ΔAP50 (D − A)</th></tr></thead><tbody>
        ${BENCH.fairness.map(g => `<tr><td>${esc(g.group)}<small>${esc(g.range)}</small></td><td>${g.images}</td>${models.map(m => `<td>${g.scores[m.id].toFixed(1)}%</td>`).join('')}<td class="mono">${g.dAP50 >= 0 ? '+' : ''}${(g.dAP50 * 100).toFixed(1)} pts</td></tr>`).join('')}
      </tbody></table>
      <p class="explain">Mann-Whitney U on the per-image ΔAP50 (D − A), III–V (n = ${ft.n2}) versus I–II (n = ${ft.n1}):
        U = ${ft.U}, <b>${pText(ft.p)}</b>, rank-biserial r = ${ft.r.toFixed(3)}; bootstrap 95% CI of the difference ${ft.ciLow.toFixed(3)} to ${ft.ciHigh.toFixed(3)}.
        ${ft.p < 0.05 ? 'H02 is rejected at α = 0.05.' : 'H02 is not rejected at α = 0.05.'}</p>
      <details class="calc"><summary>Computation</summary>
        <p>For each test image: ΔAP50 = AP50 of Model D − AP50 of Model A. Group mAP@50 values above are computed on the images of each group.</p>
        ${BENCH.fairness.map(g => `<p>${esc(g.group)}: mAP@50 D − A = ${g.scores.D.toFixed(1)}% − ${g.scores.A.toFixed(1)}% = ${(g.dAP50 * 100).toFixed(1)} points; mean per-image ΔAP50 = ${g.meanPerImageDAP50.toFixed(4)}, median = ${g.medianPerImageDAP50.toFixed(4)}.</p>`).join('')}
      </details>
    </div>`;

  const h01 = `
    <div class="card">
      <div class="card-head"><h2>4 · Differences among A–D (H01)</h2></div>
      <p class="note">Per-image scores on the ${BENCH.testImages} test images. Friedman test across A–D, then Wilcoxon signed-rank D versus A, B, C with Bonferroni correction (× 3); r = rank-biserial correlation.</p>
      <div class="table-scroll"><table class="data-table center"><thead><tr><th>Metric</th><th>Friedman χ²</th><th>p</th><th>D vs A</th><th>D vs B</th><th>D vs C</th></tr></thead><tbody>
        ${BENCH.tests.map(t => `<tr><td>${esc(t.metric)}</td><td>${t.friedmanChi2}</td><td class="${t.friedmanP < 0.05 ? 'sig' : ''}">${pText(t.friedmanP)}</td>
          ${t.pairs.map(p => `<td class="${p.pBonf < 0.05 ? 'sig' : ''}">${pText(p.pBonf)}<small>r = ${p.r.toFixed(2)}</small></td>`).join('')}</tr>`).join('')}
      </tbody></table></div>
      <p class="note">Highlighted: p < 0.05 after correction.</p>
    </div>`;

  const rt = BENCH.runtime;
  const sess = State.sessionTimings;
  const dev = State.deviceRuntime;
  const runtime = `
    <div class="card">
      <div class="card-head"><h2>5 · Inference time</h2></div>
      ${rt ? `
      <h3>Training hardware (${esc(rt.hardware.device)})</h3>
      <table class="data-table center"><thead><tr><th>Model</th><th>GFLOPs</th><th>Params (M)</th><th>Inference (ms)</th><th>Total (ms)</th></tr></thead><tbody>
        ${Object.entries(rt.models).map(([id, m]) => `<tr><td>${id}</td><td>${m.gflops_640}</td><td>${m.params_million}</td><td>${m.inference_ms_mean} ± ${m.inference_ms_sd}</td><td>${m.total_ms_mean} ± ${m.total_ms_sd}</td></tr>`).join('')}
      </tbody></table>
      <p class="note">${esc(rt.hardware.framework)}; batch ${rt.hardware.batch}, input ${rt.hardware.imgsz} px (${esc(rt.hardware.letterbox)}), ${rt.models.D.images_timed} test images after ${rt.hardware.warmup_images} warm-up images; mean ± SD.
        Total = preprocess + inference + postprocess. ${esc(rt.hardware.note)}</p>` : '<p class="note">Runtime file not found.</p>'}
      <h3>This device (on-device pipeline, Model D)</h3>
      ${dev ? runtimeTable(dev.runs, `${dev.runs.length} timed runs on a ${dev.image.width} × ${dev.image.height} px photo, after one warm-up run`) : ''}
      ${sess.length ? runtimeTable(sess, `${sess.length} analyses in this session (first one includes model loading)`) : ''}
      <button type="button" class="btn wide" onclick="measureDeviceRuntime()">Measure on this device (current photo, 5 runs)</button>
      <p class="note">Device: ${esc(deviceText())}. On-device settings: WebAssembly, 1 thread, longest side ≤ 1280 px, model input 640 px.</p>
    </div>`;

  body.innerHTML = sopMap + keys + sop1 + sop2 + sop3 + h01 + runtime;
}

function deviceText() {
  const n = navigator;
  return [n.userAgent.match(/\(([^)]+)\)/)?.[1], n.hardwareConcurrency ? `${n.hardwareConcurrency} logical cores` : null,
    n.deviceMemory ? `${n.deviceMemory} GB memory (approx.)` : null].filter(Boolean).join(' · ');
}

function runtimeTable(runs, caption) {
  const keys = [['load', 'Load'], ['ita', 'Skin mask + ITA'], ['clahe', 'L*-CLAHE'], ['detect', 'Detection'], ['inference_ms', 'of which model inference'],
    ['gradcam', 'Grad-CAM'], ['features', 'Features'], ['total', 'Total']];
  return `
    <table class="data-table center"><caption>${esc(caption)}</caption><thead><tr><th>Step</th><th>${runs.length > 1 ? 'Mean (ms)' : 'Time (ms)'}</th>${runs.length > 1 ? '<th>SD</th>' : ''}</tr></thead><tbody>
      ${keys.map(([k, label]) => { const xs = runs.map(r => r[k]).filter(v => v !== undefined); return xs.length ? `<tr><td>${label}</td><td>${mean(xs).toFixed(0)}</td>${runs.length > 1 ? `<td>${sd(xs).toFixed(0)}</td>` : ''}</tr>` : ''; }).join('')}
    </tbody></table>`;
}

function confusionMatrix(model) {
  const codes = BENCH.classes;
  const rows = BENCH.confusion[model.id];
  const max = Math.max(1, ...rows.flat());
  const correct = rows.reduce((a, row, i) => a + row[i], 0);
  const matched = rows.flat().reduce((a, b) => a + b, 0);
  return `
    <div class="table-scroll"><table class="confusion">
      <thead><tr><th>Actual ↓ / Predicted →</th>${codes.map(c => `<th>${c}</th>`).join('')}</tr></thead>
      <tbody>${rows.map((row, i) => `<tr><th>${codes[i]}</th>${row.map((v, j) => `<td class="${i === j ? 'diag' : v ? 'err' : ''}" style="--a:${(i === j ? 0.15 + 0.85 * v / max : Math.min(v / 10, 0.6)).toFixed(2)}">${v}</td>`).join('')}</tr>`).join('')}</tbody>
    </table></div>
    <p class="note">Matched boxes (confidence ≥ 0.25, IoU ≥ 0.5): ${correct} of ${matched} given the correct disease (${(100 * correct / matched).toFixed(1)}%).
      ${BENCH.classes.map((c, i) => `${c} = ${esc(BENCH.classNames[i])}`).join(', ')}.</p>`;
}

async function measureDeviceRuntime() {
  const cur = State.current;
  if (!cur) { alertInDialog('Analyse a photo first; the measurement repeats Model D on that photo.'); return; }
  const repeats = 5;
  const steps = [{ id: 'warmup', label: 'Warm-up run (not timed)' },
    ...Array.from({ length: repeats }, (_, i) => ({ id: `r${i}`, label: `Timed run ${i + 1} of ${repeats}` }))];
  const job = Progress.open({ title: 'Measuring inference time', photo: cur.source, steps });
  try {
    Progress.update('warmup', 'active');
    const t = performance.now();
    await OnDevice.runDetect(cur.source);
    Progress.update('warmup', 'done', performance.now() - t);
    const runs = [];
    for (let i = 0; i < repeats; i++) {
      if (job.cancelled) return;
      Progress.update(`r${i}`, 'active');
      const r = await OnDevice.runDetect(cur.source);
      runs.push(r.timings);
      Progress.update(`r${i}`, 'done', r.timings.total);
    }
    if (job.cancelled) return;
    State.deviceRuntime = { runs, at: new Date(), image: cur.result.image };
    renderBenchmark();
    const totals = runs.map(r => r.total);
    Progress.finish({ title: 'Measurement complete', message: `<p>Mean total time per analysis: <b>${mean(totals).toFixed(0)} ± ${sd(totals).toFixed(0)} ms</b> (n = ${runs.length}).</p>` });
  } catch (err) {
    if (!job.cancelled) Progress.finish({ title: 'Measurement failed', failed: true, message: `<p>${esc(err.message || err)}</p>` });
  }
}

function alertInDialog(text) {
  Progress.open({ title: 'IDENTI-SKIN', steps: [] });
  Progress.finish({ title: 'IDENTI-SKIN', message: `<p>${esc(text)}</p>`, tone: 'warn' });
}

// ------------------------------------------------------------------ rendering: research

function itaGroup(ita) {
  if (ita === null) return 'not available (no ITA)';
  if (ita > 41) return 'I–II proxy (ITA > 41°)';
  if (ita > -30) return 'III–V proxy (−30° < ITA ≤ 41°)';
  return 'VI proxy (ITA ≤ −30°), excluded from SOP 3';
}

function renderResearch() {
  const body = $('research-body');
  const cur = State.current;
  let html = '';
  if (!cur) {
    html += '<div class="card empty-state">Analyse a photo to see how its result was generated.</div>';
  } else {
    const r = cur.result;
    const it = r.ita_inputs;
    const nearest = r.ita === null ? null : BRACKET_EDGES.reduce((a, e) => (Math.abs(r.ita - e) < Math.abs(r.ita - a) ? e : a));
    const top = r.detections[0] || r.below_cutoff;
    html += `
      <div class="card">
        <div class="card-head"><h2>How this result was generated</h2><span class="tag">${esc(cur.name)}</span></div>
        <h3>Phase 0: skin tone and enhancement</h3>
        <table class="kv">
          <tr><td>Skin mask</td><td>${it ? `K-means in CIELAB (k = ${it.k}) on the photo reduced to ${it.width} × ${it.height} px; skin cluster = ${it.maskPixels} of ${it.pixels} pixels (${(100 * it.maskPixels / it.pixels).toFixed(1)}%), status ${esc(r.mask_status)}` : `failed (${esc(r.mask_status)})`}</td></tr>
          ${it ? `<tr><td>Mean skin colour</td><td>L* = ${it.meanL.toFixed(2)}, b* = ${it.meanB.toFixed(2)}</td></tr>
          <tr><td>ITA</td><td>arctan((L* − 50) / b*) × 180/π = arctan((${it.meanL.toFixed(2)} − 50) / ${it.meanB.toFixed(2)}) × 180/π = <b>${r.ita.toFixed(1)}°</b></td></tr>
          <tr><td>ITA bracket</td><td>${esc(r.bracket)} (${BRACKET_RULE}); ${Math.abs(r.ita - nearest).toFixed(1)}° from the ${nearest}° edge</td></tr>` : ''}
          <tr><td>Clip limit β</td><td><b>${r.beta}</b> (${r.beta_source === 'ita_bracket' ? 'from the ITA bracket' : 'medium-bracket fallback, no ITA'}; calibration: Darkest ${r.settings.calibration.beta_high}, Medium ${r.settings.calibration.beta_mid}, Lightest ${r.settings.calibration.beta_low})</td></tr>
          <tr><td>L*-CLAHE</td><td>lightness channel only, tile grid ${r.settings.tile_grid_size.join(' × ')}</td></tr>
          <tr><td>SOP 3 group</td><td>${itaGroup(r.ita)}. ITA is a colour measurement, not a clinical Fitzpatrick assessment.</td></tr>
        </table>
        <h3>Localization and classification</h3>
        <table class="kv">
          <tr><td>Model</td><td>Model D, D.onnx (${esc(BENCH_D_WEIGHTS)}), one-to-many head + NMS (as in the test-set evaluation)</td></tr>
          <tr><td>Input</td><td>${r.image.width} × ${r.image.height} px → letterbox to ${r.model_input[1]} × ${r.model_input[0]} (640, stride 32, grey padding)</td></tr>
          <tr><td>Cutoff / NMS</td><td>confidence ≥ ${r.settings.confidence}; per-class NMS, IoU ${r.settings.nms_iou}</td></tr>
          <tr><td>Boxes</td><td>${r.detections.length}${r.below_cutoff ? `; strongest below the cutoff: ${esc(r.below_cutoff.label)} ${r.below_cutoff.confidence.toFixed(4)}` : ''}</td></tr>
        </table>
        ${r.detections.length ? `
        <div class="table-scroll"><table class="data-table"><thead><tr><th>#</th><th>Label</th><th>σ</th><th>x1, y1, x2, y2 (px)</th>${CLASSES.map(c => `<th title="${esc(c)}">${esc(c.split(' ').map(w => w[0]).join(''))}</th>`).join('')}</tr></thead><tbody>
          ${r.detections.slice(0, 15).map((d, i) => `<tr><td>${i + 1}</td><td>${esc(d.label)}</td><td class="mono">${d.confidence.toFixed(4)}</td><td class="mono">${d.box_px.join(', ')}</td>${d.scores.map(s => `<td class="mono">${s.toFixed(3)}</td>`).join('')}</tr>`).join('')}
        </tbody></table></div>
        <p class="note">All 8 sigmoid class scores of each box (column initials: ${CLASSES.map(c => `${c.split(' ').map(w => w[0]).join('')} = ${c}`).join(', ')}).</p>` : ''}
        <h3>Measured time of this run</h3>
        ${runtimeTable([r.timings], r.first_load ? 'This run (includes loading the model)' : 'This run')}
      </div>`;

    const c = State.consistency;
    html += `
      <div class="card">
        <div class="card-head"><h2>Run consistency</h2></div>
        <p class="note">Repeats the analysis of this photo with identical settings and compares ITA, β, every box, its label and all class scores.</p>
        ${c ? `<p class="explain ${c.identical === c.runs ? 'ok' : 'bad'}">${c.identical} of ${c.runs} repeated runs identical to the first run.
          ${c.identical === c.runs ? 'The result is stable.' : `Largest difference: ${esc(c.difference)}.`}</p>
          ${runtimeTable(c.timings, `${c.runs} repeated runs`)}` : ''}
        <button type="button" class="btn wide" onclick="runConsistency()">Repeat this analysis 5 times</button>
      </div>`;

    html += `
      <div class="card">
        <div class="card-head"><h2>Grad-CAM of this photo</h2><span class="tag">Research only</span></div>
        ${r.gradcam_image ? `<img class="figure" src="${r.gradcam_image}" alt="Grad-CAM of Model D on this photo">` : '<p class="note">Not available.</p>'}
        <p class="note">Shows where Model D's class evidence is strongest (Grad-CAM at the last feature layer of the class head; target anchors with score ≥ 0.25, up to 10:
          ${r.gradcam_targets.length ? r.gradcam_targets.map(t => `${esc(t.label)} ${t.score.toFixed(3)}`).join(', ') : 'none'}).
          ${r.lesion_measures && r.lesion_measures.gradcam_in_box_percent !== null ? `Share of the heatmap inside the detected boxes: ${r.lesion_measures.gradcam_in_box_percent}%.` : ''}
          A heatmap shows where the model focused. It is not a measure of accuracy and not a validated explanation of the diagnosis.</p>
        ${State.compare ? `
          <h3>Per configuration (same photo)</h3>
          <div class="cam-grid">${State.compare.models.map(m => m.gradcam_image ? `<figure><img src="${m.gradcam_image}" alt="Grad-CAM of Model ${m.id}"><figcaption>Model ${m.id}</figcaption></figure>` : '').join('')}</div>` : ''}
      </div>`;
  }
  html += gradcamAudit();
  body.innerHTML = html;
}

function gradcamAudit() {
  if (!GRADCAM) return '';
  const s = GRADCAM.summary, st = GRADCAM.statistics;
  const ex = GRADCAM.examples[Math.min(State.gradcamExample, GRADCAM.examples.length - 1)];
  return `
    <div class="card">
      <div class="card-head"><h2>Grad-CAM on the test set</h2><span class="tag">Research only</span></div>
      <table class="data-table center"><thead><tr><th>Model</th><th>IoU with labelled boxes</th><th>Heatmap energy inside boxes</th></tr></thead><tbody>
        ${Object.keys(CONFIGS).map(k => `<tr><td>${k}</td><td>${s[k].iou_mean.toFixed(3)}</td><td>${(s[k].energy_mean * 100).toFixed(1)}%</td></tr>`).join('')}
      </tbody></table>
      <p class="note">${esc(GRADCAM.method)}. IoU uses the heatmap region ≥ ${GRADCAM.iou_threshold * 100}% of its maximum.
        IoU: Friedman ${pText(st.gradcam_iou.friedman_p)}; D vs A ${pText(st.gradcam_iou.D_vs_A_p_bonf)}, D vs B ${pText(st.gradcam_iou.D_vs_B_p_bonf)}, D vs C ${pText(st.gradcam_iou.D_vs_C_p_bonf)} (Bonferroni).
        These describe where each model looks; they are not accuracy measures. Examples are pending review by the adviser/panel.</p>
      <div class="chips">${GRADCAM.examples.map((e, i) => `<button type="button" class="chip${e === ex ? ' active' : ''}" onclick="State.gradcamExample=${i}; renderResearch()">${esc(e.disease)}</button>`).join('')}</div>
      <div class="cam-grid">
        <figure><img src="assets/gradcam/${ex.image_id}_original.jpg" alt="Original test image"><figcaption>Original (${esc(ex.disease)})</figcaption></figure>
        ${Object.keys(CONFIGS).map(k => `<figure><img src="assets/gradcam/${ex.image_id}_${k}.jpg" alt="Grad-CAM of Model ${k}"><figcaption>${k} · IoU ${ex.models[k].iou.toFixed(2)}</figcaption></figure>`).join('')}
      </div>
    </div>`;
}

// Signature of a result for the consistency check (everything except timings and images).
function signature(r) {
  return {
    ita: r.ita, beta: r.beta,
    boxes: r.detections.map(d => ({ label: d.label, confidence: d.confidence, box: d.box, scores: d.scores })),
    below: r.below_cutoff && { label: r.below_cutoff.label, confidence: r.below_cutoff.confidence },
  };
}

function firstDifference(a, b) {
  if (a.ita !== b.ita) return `ITA ${a.ita} vs ${b.ita}`;
  if (a.beta !== b.beta) return `β ${a.beta} vs ${b.beta}`;
  if (a.boxes.length !== b.boxes.length) return `${a.boxes.length} vs ${b.boxes.length} boxes`;
  for (let i = 0; i < a.boxes.length; i++) {
    const x = a.boxes[i], y = b.boxes[i];
    if (x.label !== y.label) return `box ${i + 1}: ${x.label} vs ${y.label}`;
    if (x.confidence !== y.confidence) return `box ${i + 1}: confidence ${x.confidence} vs ${y.confidence}`;
    if (JSON.stringify(x.box) !== JSON.stringify(y.box)) return `box ${i + 1}: position differs`;
    if (JSON.stringify(x.scores) !== JSON.stringify(y.scores)) return `box ${i + 1}: class scores differ`;
  }
  return 'below-cutoff candidate differs';
}

async function runConsistency() {
  const cur = State.current;
  if (!cur) return;
  const runs = 5;
  const job = Progress.open({ title: 'Checking run consistency', photo: cur.source,
    steps: Array.from({ length: runs }, (_, i) => ({ id: `run${i}`, label: `Repeat ${i + 1} of ${runs}` })) });
  const reference = JSON.stringify(signature(cur.result));
  let identical = 0, difference = null;
  const timings = [];
  try {
    for (let i = 0; i < runs; i++) {
      if (job.cancelled) return;
      Progress.update(`run${i}`, 'active');
      const r = await OnDevice.runDetect(cur.source);
      timings.push(r.timings);
      const sig = signature(r);
      if (JSON.stringify(sig) === reference) identical++;
      else if (!difference) difference = firstDifference(signature(cur.result), sig);
      Progress.update(`run${i}`, 'done', r.timings.total);
    }
  } catch (err) {
    if (!job.cancelled) Progress.finish({ title: 'Check failed', failed: true, message: `<p>${esc(err.message || err)}</p>` });
    return;
  }
  if (job.cancelled) return;
  State.consistency = { runs, identical, difference, timings };
  renderResearch();
  Progress.finish({ title: identical === runs ? 'Results are identical' : 'Results differ',
    message: `<p>${identical} of ${runs} repeated runs gave exactly the same ITA, β, boxes and scores as the first run.${difference ? ` First difference: ${esc(difference)}.` : ''}</p>` });
}

// ------------------------------------------------------------------ report

function openReport() {
  const cur = State.current;
  const body = $('report-body');
  if (!cur) {
    body.innerHTML = '<h2>Report</h2><p>No photo analysed yet.</p>';
  } else {
    const r = cur.result;
    const top = r.detections[0];
    const d = BENCH && BENCH.models.find(m => m.id === 'D');
    body.innerHTML = `
      <div class="report-head"><div><b>IDENTI-SKIN screening summary</b><div class="muted">ITA-guided adaptive L*-CLAHE YOLOv26 framework (Model D), on-device</div></div>
        <div class="muted">${cur.at.toLocaleString()}</div></div>
      <img class="report-photo" src="${r.boxed_image}" alt="Analysed photo with detected boxes">
      <table class="kv">
        <tr><td>Top prediction</td><td>${top ? `<b>${esc(DISPLAY_NAME[top.label])}</b> (${CATEGORY[top.label]}; ${esc(clusterOf(top.label).name)} group)` : 'No lesion detected'}</td></tr>
        <tr><td>Model confidence</td><td>${top ? `${pct(top.confidence)} (sigmoid score of the top box; not a probability of disease)` : `no box ≥ ${r.settings.confidence}`}</td></tr>
        ${top ? `<tr><td>Candidate scores (same group)</td><td>${clusterOf(top.label).classes.map(c => `${esc(c)} ${pctS(top.scores[CLASSES.indexOf(c)])}`).join(' · ')}</td></tr>` : ''}
        <tr><td>Detected boxes</td><td>${r.detections.length}</td></tr>
        <tr><td>Enhancement</td><td>L*-CLAHE, clip limit β = ${r.beta}</td></tr>
        ${isClinician() ? `<tr><td>ITA</td><td>${r.ita === null ? 'not available' : `${r.ita.toFixed(1)}° (${esc(r.bracket)} bracket)`}</td></tr>` : ''}
        ${d ? `<tr><td>Model D on the test set</td><td>mAP@50 ${d.map50.toFixed(1)}%, F1 ${d.f1.toFixed(3)} (${BENCH.testImages} held-out images; not specific to this photo)</td></tr>` : ''}
      </table>
      <p class="note">Academic screening aid (PUP CCIS thesis 2026). Not a diagnosis; consult a licensed physician or dermatologist. Analysed on this device (RA 10173).</p>
      <div class="signature">Attending physician / dermatologist</div>`;
  }
  $('report-dialog').hidden = false;
}

function closeReport() { $('report-dialog').hidden = true; }

// ------------------------------------------------------------------ init

function renderAll() {
  renderHome();
  renderResults();
  renderWorkspace();
  renderBenchmark();
  renderResearch();
}

async function loadData() {
  try {
    const res = await fetch('benchmark_data.json', { cache: 'no-store' });
    if (res.ok) BENCH = await res.json();
  } catch (err) { BENCH = null; }
  try {
    const res = await fetch('assets/gradcam/gradcam.json', { cache: 'no-store' });
    if (res.ok) GRADCAM = await res.json();
  } catch (err) { GRADCAM = null; }
  renderAll();
}

document.addEventListener('DOMContentLoaded', () => {
  initUser();
  $('file-input').addEventListener('change', e => { handleFile(e.target.files[0]); e.target.value = ''; });
  $('camera-input').addEventListener('change', e => { handleFile(e.target.files[0]); e.target.value = ''; });
  $('split-slider').addEventListener('input', e => setSplit(+e.target.value));
  $('toggle-boxes').addEventListener('change', e => { State.showBoxes = e.target.checked; renderWorkspace(); });
  renderAll();
  loadData();
  const hash = window.location.hash.replace('#', '');
  if (hash) switchTab(hash);
  window.addEventListener('hashchange', () => switchTab(window.location.hash.replace('#', '')));
});
