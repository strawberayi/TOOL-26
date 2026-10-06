const CLASSES = ['Warts', 'Molluscum', 'Varicella', 'HFMD', 'Tinea versicolor', 'Tinea corporis', 'Tinea pedis', 'Impetigo'];


const CLUSTERS = [
  { name: 'Vesiculopapular / Eruptive', classes: ['Varicella', 'HFMD', 'Molluscum', 'Impetigo'] },
  { name: 'Papulosquamous / Verrucous', classes: ['Tinea corporis', 'Tinea versicolor', 'Warts', 'Tinea pedis'] },
];


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


const CONFIGS = {
  A: { short: 'Baseline', text: 'Baseline YOLOv26, raw images' },
  B: { short: 'Fixed L*-CLAHE', text: 'Fixed-parameter L*-CLAHE, clip limit 2.0' },
  C: { short: 'Focal Loss', text: 'Focal Loss optimization, raw images' },
  D: { short: 'Proposed', text: 'ITA-guided adaptive L*-CLAHE + two-stage decoupled training + Focal Loss' },
};


const BRACKET_RULE = 'Darkest: ITA < 28° · Medium: 28° ≤ ITA ≤ 41° · Lightest: ITA > 41°';
const BRACKET_EDGES = [28, 41];


const State = {
  user: null,
  tab: 'home',
  current: null,
  history: [],
  compare: null,
  consistency: null,
  deviceRuntime: null,
  sessionTimings: [],
  benchModel: 'D',
  gradcamExample: 0,
  split: 100,
  showBoxes: true,
  showHeatmap: true,
  heatmapOpacity: 100,
};
let BENCH = null;
let GRADCAM = null;
let MORPH_REF = null;


const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

const pctS = x => (x >= 0.01 ? pct(x) : x >= 0.0001 ? `${(x * 100).toFixed(2)}%` : '< 0.01%');
const num = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? '—' : Number(x).toFixed(d));
const ms = x => `${Math.round(x)} ms`;
const pText = p => (p < 0.001 ? 'p < 0.001' : `p = ${p.toFixed(3)}`);
const clusterOf = label => CLUSTERS.find(c => c.classes.includes(label));
const logit = p => { const q = Math.min(Math.max(p, 1e-12), 1 - 1e-12); return Math.log(q / (1 - q)); };

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
  { id: 'features', label: 'Cropping the detected lesions' },
];


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
    try { localStorage.removeItem('identi_skin_user'); } catch (err) {  }
  }
  window.location.href = 'login.html';
}


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
  viewer.style.width = `min(100%, calc(62vh * ${(W / H).toFixed(4)}))`;
  $('viewer-original').src = cur.source;
  $('viewer-enhanced').src = r.enhanced_image;
  setSplit(State.split);

  const heat = $('viewer-heatmap');
  heat.src = r.gradcam_heatmap || '';
  heat.hidden = !State.showHeatmap || !r.gradcam_heatmap;
  heat.style.opacity = State.heatmapOpacity / 100;
  const below = r.below_cutoff;
  $('heatmap-note').textContent = !r.gradcam_heatmap ? 'Grad-CAM is not available for this photo.'
    : r.detections.length
      ? `Grad-CAM (same as in Research): red and yellow are where Model D found the evidence for the ${r.detections.length} detected box(es). It shows where the model focused, not whether it is correct. Move the slider left to compare with the enhanced photo.`
      : `No box reached the ${pct(r.settings.confidence, 0)} cutoff, so nothing is counted as a lesion. The warm areas are where the model saw weak signs of ${below ? below.label : 'a disease'}; the strongest one (dashed box, ${below ? pct(below.confidence) : '—'}) was still below the cutoff.`;

  const svg = $('viewer-boxes');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const font = Math.max(10, Math.round(Math.max(W, H) / 45));
  const boxSvg = (d, label, cls) => {
    const [x1, y1, x2, y2] = [d.box[0] * W, d.box[1] * H, d.box[2] * W, d.box[3] * H];
    const ty = Math.max(0, y1 - font * 1.4);
    const tw = label.length * font * 0.58 + font * 0.6;
    const tx = Math.max(0, Math.min(x1, W - tw));
    return `<g class="det ${cls}">
      <rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" stroke-width="${font / 5}"></rect>
      <rect class="det-tag" x="${tx}" y="${ty}" width="${tw}" height="${font * 1.4}"></rect>
      <text x="${tx + font * 0.3}" y="${ty + font * 1.05}" font-size="${font}">${esc(label)}</text>
    </g>`;
  };
  svg.innerHTML = !State.showBoxes ? ''
    : r.detections.length
      ? r.detections.map((d, i) => boxSvg(d, `${i + 1}. ${d.label} ${pct(d.confidence)}`, i === 0 ? 'top' : '')).join('')
      : below ? boxSvg(below, `Not counted: ${below.label} ${pct(below.confidence)}`, 'below') : '';

  const top = r.detections[0];
  $('workspace-morph').innerHTML = top
    ? morphologyCard(top, r, true)
    : '<p class="note">Morphological features are shown when a lesion is detected.</p>';
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
      <p class="note">Model scores of the four diseases in the <b>${esc(cluster.name)}</b> group, read from the same box.</p>
      ${scoreBars(det.scores, ranked, det.label)}
      <p class="explain">${esc(DISPLAY_NAME[ranked[0]])} ranked first: ${pctS(s(ranked[0]))} versus ${esc(DISPLAY_NAME[ranked[1]])} ${pctS(s(ranked[1]))}
        (difference ${((s(ranked[0]) - s(ranked[1])) * 100).toFixed(1)} percentage points).
        Highest score outside this group: ${esc(DISPLAY_NAME[outside])} ${pctS(s(outside))}.</p>
      <details class="calc">
        <summary>Why don't these add up to 100%?</summary>
        <p>The model asks a separate yes-or-no question for each disease: "Is this ${esc(DISPLAY_NAME[ranked[0]])}?", "Is this ${esc(DISPLAY_NAME[ranked[1]])}?", and so on.
          Each answer is its own score from 0% to 100% (a sigmoid output), so the scores are not shares of one whole.</p>
        <p>${pctS(s(ranked[0]))} for ${esc(DISPLAY_NAME[ranked[0]])} means the model is that sure this lesion is ${esc(DISPLAY_NAME[ranked[0]])}. The remaining
          ${((1 - s(ranked[0])) * 100).toFixed(1)}% is its doubt about that answer; it is not given to the other diseases. Each of the other diseases got its own,
          separate low score (for example ${esc(DISPLAY_NAME[ranked[1]])} ${pctS(s(ranked[1]))}), which means the model says "no" to them.</p>
        <p>This is how YOLO detectors are trained (one yes/no loss per disease, binary cross-entropy / Focal Loss), so a lesion can also score low for every disease;
          that is when "No lesion detected" appears.</p>
      </details>
    </div>`;
}

function featureCard(r) {
  const crops = r.lesion_crops || [];
  if (!crops.length) return '';
  return `
    <div class="card">
      <div class="card-head"><h2>Detected lesions</h2><span class="tag">${crops.length}</span></div>
      <p class="note">Each lesion found by the model, cropped from the photo, with its model confidence.</p>
      <div class="crops">
        ${crops.map((c, i) => `
          <article class="crop">
            <div class="crop-top">
              <img src="${c.image}" alt="Lesion ${i + 1}">
              <div>
                <div class="crop-conf">${(c.confidence * 100).toFixed(2)}</div>
                <div class="crop-label">${esc(DISPLAY_NAME[c.label])}</div>
                <div class="muted">model confidence (%)</div>
              </div>
            </div>
          </article>`).join('')}
      </div>
    </div>`;
}

function percentile(feature, x) {
  const ref = MORPH_REF && MORPH_REF.features[feature];
  if (!ref || x === null || x === undefined || !Number.isFinite(x)) return null;
  const q = ref.q;
  if (x <= q[0]) return 0;
  if (x >= q[100]) return 100;
  let i = 0;
  while (i < 99 && q[i + 1] < x) i++;
  const span = q[i + 1] - q[i];
  return i + (span > 0 ? (x - q[i]) / span : 0);
}

function morphologyRows(det, r) {
  const m = r.morphology;
  const cluster = clusterOf(det.label);
  const s = l => det.scores[CLASSES.indexOf(l)];
  const runner = cluster.classes.filter(c => c !== det.label).sort((a, b) => s(b) - s(a))[0];
  const margin = s(det.label) - s(runner);
  const r1 = v => (Math.round(v * 100) / 100).toFixed(2);
  const rows = [
    { key: 'texture', name: 'Surface texture', ref: 'texture', value: m && m.texture,
      shown: m && m.texture !== null ? `roughness ${r1(m.texture)}× the surrounding skin` : '—',
      how: m ? `σ of the Laplacian (local light-dark changes) inside the lesion boxes ÷ in the skin around them = ${r1(m.lesion_rough)} ÷ ${r1(m.skin_rough)} = ${r1(m.texture)}` : '' },
    { key: 'crust', name: 'Crust and exudate', ref: 'crust', value: m && m.crust,
      shown: m ? `yellowness Δb* ${m.crust >= 0 ? '+' : ''}${r1(m.crust)} versus the skin` : '—',
      how: m ? `Δb* = b* of the lesions − b* of the skin = ${r1(m.lesion_lab[2])} − ${r1(m.skin_lab[2])} = ${r1(m.crust)} (crusts and exudate are yellow to brown, so they raise b*)` : '' },
    { key: 'edge', name: 'Boundary / edge', ref: 'border', value: m && m.edge,
      shown: m ? `colour contrast ΔE ${r1(m.edge)} with the skin` : '—',
      how: m ? `ΔE*ab = √(ΔL*² + Δa*² + Δb*²) between the lesions (${m.lesion_lab.map(r1).join(', ')}) and the skin (${m.skin_lab.map(r1).join(', ')}) = ${r1(m.edge)}; a sharper, clearer edge gives a larger contrast` : '' },
    { key: 'differential', name: 'Differential key', ref: 'apart', value: margin,
      shown: `${esc(det.label)} ahead of ${esc(runner)} by ${(margin * 100).toFixed(1)} points`,
      how: `model score of ${esc(det.label)} − score of its closest look-alike ${esc(runner)} = ${pctS(s(det.label))} − ${pctS(s(runner))} = ${(margin * 100).toFixed(1)} percentage points` },
  ];
  rows.forEach(row => { row.strength = percentile(row.key, row.value); });
  const total = rows.reduce((a, row) => a + (row.strength || 0), 0);
  rows.forEach(row => { row.split = total > 0 && row.strength !== null ? 100 * row.strength / total : null; });
  return rows;
}

function morphologyCard(det, r, embedded = false) {
  const ref = MORPHOLOGY[det.label];
  const rows = morphologyRows(det, r);
  const hasSplit = rows.some(row => row.split !== null);
  return `
    <div class="${embedded ? 'morphology embedded' : 'card morphology'}">
      <div class="card-head"><h2>Morphological features</h2><span class="tag">Whole photo</span></div>
      <p class="note">The four features, measured over all detected lesions in the photo. The split shows which features stand out most in this photo.</p>
      ${hasSplit ? `
      <div class="split-bar" aria-label="Feature split">${rows.map(row => row.split ? `<span class="f-${row.key}" style="width:${row.split.toFixed(1)}%" title="${row.name} ${row.split.toFixed(1)}%"></span>` : '').join('')}</div>` : ''}
      ${rows.map(row => `
        <div class="feature-row">
          <div class="feature-head"><span><i class="dot f-${row.key}"></i>${row.name}</span><b>${row.split === null ? '—' : `${row.split.toFixed(1)}%`}</b></div>
          <div class="feature-measure">${row.shown}${row.strength === null ? '' : ` · stronger than ${Math.round(row.strength)}% of training lesions`}</div>
          <div class="feature-ref">Typical for ${esc(DISPLAY_NAME[det.label])}: ${esc(ref[row.ref])}</div>
        </div>`).join('')}
      <details class="calc">
        <summary>How the feature percentages are computed</summary>
        <p>1. Each feature is measured on the photo, inside the lesion boxes and in the ring of skin around them (each box enlarged by 50%):</p>
        <ul class="calc-list">${rows.map(row => `<li><b>${row.name}:</b> ${row.how}.</li>`).join('')}</ul>
        <p>2. <b>Strength</b> = the percentile of that value among the ${MORPH_REF ? MORPH_REF.features.texture.n : ''} training images
          (for example 80 means stronger than 80% of the labelled training lesions; for the differential key, Model D's own top boxes on the training images).</p>
        <p>3. <b>Split</b> = strength of the feature ÷ sum of the four strengths × 100, so the four add up to 100%:
          ${rows.map(row => row.strength === null ? '' : `${row.name} ${Math.round(row.strength)}`).filter(Boolean).join(' + ')} = ${Math.round(rows.reduce((a, row) => a + (row.strength || 0), 0))};
          ${rows.map(row => row.split === null ? '' : `${row.name} ${Math.round(row.strength)} ÷ ${Math.round(rows.reduce((a, x) => a + (x.strength || 0), 0))} = ${row.split.toFixed(1)}%`).filter(Boolean).join('; ')}.</p>
        <p>These measurements describe the lesions; the model's decision comes from its own scores. "Typical" lines are reference text from the literature
          (Chauhan et al., 2023; Leung et al., 2022; Rahim et al., 2025).</p>
      </details>
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
const PLAIN_METRIC = { 'AP@50': 'Box accuracy', 'AP@50–95': 'Strict box accuracy', Precision: 'Precision', Recall: 'Recall', F1: 'F1' };

function sopCard(r) {
  const top = r.detections[0];
  const cluster = top && clusterOf(top.label);
  const s = l => top.scores[CLASSES.indexOf(l)];
  const runner = top && cluster.classes.filter(c => c !== top.label).sort((a, b) => s(b) - s(a))[0];
  return `
    <div class="card clinician-only">
      <div class="card-head"><h2>This photo and the SOPs</h2></div>
      <table class="kv">
        <tr><td><b>SOP 1</b> Localization</td><td>${r.detections.length} lesion box(es) found by Model D; the top box covers ${top ? `${(100 * (top.box[2] - top.box[0]) * (top.box[3] - top.box[1])).toFixed(1)}% of the photo` : '—'} (see Workspace).</td></tr>
        <tr><td><b>SOP 2</b> Classification</td><td>${top ? `${esc(top.label)} (${pct(top.confidence)}) in the ${esc(cluster.name)} cluster; closest look-alike ${esc(runner)} (${pctS(s(runner))}).` : 'No box above the cutoff.'}</td></tr>
        <tr><td><b>SOP 3</b> Skin tone</td><td>${r.ita === null ? 'ITA not available.' : `ITA ${r.ita.toFixed(1)}° → ${itaGroup(r.ita)}; L*-CLAHE clip limit β = ${r.beta}.`}</td></tr>
      </table>
      <p class="note">Test-set results for each SOP (200 images, Models A–D) are in the Benchmark tab.</p>
    </div>`;
}

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
        <div class="remaining">
          <div class="remaining-bar"><span class="yes" style="width:${(top.confidence * 100).toFixed(1)}%">${pct(top.confidence, 0)} "yes, ${esc(top.label)}"</span><span class="no">${pct(1 - top.confidence, 0)} doubt</span></div>
          <p><b>Where is the remaining ${pct(1 - top.confidence)}?</b> The model answers one yes-or-no question per disease. For "Is this ${esc(DISPLAY_NAME[top.label])}?"
            it is ${pct(top.confidence)} sure the answer is yes, so ${pct(1 - top.confidence)} is its doubt about that answer (the chance it is not ${esc(DISPLAY_NAME[top.label])}).
            The remaining ${pct(1 - top.confidence)} is not given to another disease: each of the other diseases got its own separate score
            (shown below), and they are all low, which means "no".</p>
          <p class="note">Formula: score = 1 / (1 + e<sup>−z</sup>) (sigmoid), where z is the model's raw output for that disease; here z = ${logit(top.confidence).toFixed(2)}.
            Doubt = 1 − score = ${pct(1 - top.confidence)}.</p>
        </div>
        <p class="note">This is the detector's score in its highest-confidence box, not a medical probability that you have the disease.
          Group: ${esc(cluster.name)} (analysis grouping used in the study).</p>
        ${r.ita !== null && r.ita <= -30 ? `<p class="note warn">This skin is very dark (ITA ${r.ita.toFixed(1)}°, Fitzpatrick VI range). The study did not test this range, so the result may be less reliable.</p>` : ''}
      </div>`;
    html += sopCard(r);
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
    html += sopCard(r);
  }
  html += featureCard(r);
  if (top) html += morphologyCard(top, r);
  html += calculationCard(r);
  html += `<button type="button" class="btn wide" onclick="openReport()">Print / save report</button>`;
  $('results-body').innerHTML = html;
}


function selectBenchModel(id) {
  State.benchModel = id;
  renderBenchmark();
}

function renderBenchmark() {
  const body = $('bench-body');
  if (!BENCH) { $('bench-note').textContent = 'benchmark_data.json not found.'; body.innerHTML = ''; return; }
  $('bench-note').textContent = `The four models (A–D) tested on the same ${BENCH.testImages} photos that were never used in training.`;
  const models = BENCH.models;
  const sel = models.find(m => m.id === State.benchModel) || models[3];
  const cell = (m, v) => `<td class="${m.id === sel.id ? 'sel' : ''}">${v}</td>`;
  const head = `<tr><th></th>${models.map(m => `<th class="${m.id === sel.id ? 'sel' : ''}">${m.id}</th>`).join('')}</tr>`;
  const ex = sel.exact;
  const pc = sel.perClass;
  const sum = key => pc.reduce((a, c) => a + c[key], 0);

  const sopMap = `
    <div class="card">
      <div class="card-head"><h2>What this page answers</h2></div>
      <table class="data-table">
        <tr><td><b>SOP 1</b> How well does each model find and box the lesions?</td><td>Section 1</td></tr>
        <tr><td><b>SOP 2</b> How well does each model name the right disease, especially look-alike diseases?</td><td>Section 2</td></tr>
        <tr><td><b>SOP 3</b> Does Model D help darker skin (Fitzpatrick III–V) as much as lighter skin (I–II)?</td><td>Section 3</td></tr>
        <tr><td><b>H01, H02</b> Are the differences real or just chance?</td><td>Sections 3 and 4</td></tr>
        <tr><td>How fast is it?</td><td>Section 5</td></tr>
      </table>
      <details class="calc"><summary>Words used on this page</summary>
        <p><b>Box accuracy (mAP@50)</b>: how often the model's box lands on the real lesion and has the right disease, averaged over the 8 diseases.
          The box counts if it overlaps the real lesion by at least half (IoU ≥ 0.5).</p>
        <p><b>Strict box accuracy (mAP@50–95)</b>: the same, but the box must fit the lesion more tightly (overlap from 50% up to 95%).</p>
        <p><b>Precision</b>: of the lesions the model reported, how many were really that disease. Low precision = many false alarms.</p>
        <p><b>Recall</b>: of the real lesions, how many the model found. Low recall = many missed lesions.</p>
        <p><b>F1</b>: one score that balances precision and recall (0 to 1, higher is better).</p>
        <p><b>Look-alike errors (within-cluster rate)</b>: how often a lesion was named as a different disease from the same look-alike group.
          Lower is better.</p>
        <p><b>Confusion matrix</b>: a table of real disease (rows) versus the model's answer (columns). The diagonal is correct.</p>
        <p><b>Skin-tone group</b>: from the ITA, a colour measure of the skin in the photo (an estimate of Fitzpatrick type, not a clinical assessment).</p>
        <p><b>p-value</b>: below 0.05 means the difference is unlikely to be due to chance (statistically significant).</p>
        <p class="muted">Technical note: ${esc(BENCH.note)}</p>
      </details>
    </div>`;

  const keys = `
    <div class="model-keys selectable">${models.map(m => `
      <button type="button" class="model-key${m.id === sel.id ? ' selected' : ''}" onclick="selectBenchModel('${m.id}')" aria-pressed="${m.id === sel.id}">
        <b>${m.id}</b><span>${esc(CONFIGS[m.id].text)}</span></button>`).join('')}</div>
    <p class="note">Tap a model to see its details: results per disease, how each number was computed, and which diseases it mixes up.</p>`;

  const sop1 = `
    <div class="card">
      <div class="card-head"><h2>1 · Finding the lesions (SOP 1)</h2></div>
      <p class="note">How well each model puts a box on the real lesion with the right disease, on the 200 test images. Higher is better.</p>
      <table class="data-table center"><thead>${head}</thead><tbody>
        <tr><th>Box accuracy<small>mAP@50</small></th>${models.map(m => cell(m, `${m.map50.toFixed(1)}%`)).join('')}</tr>
        <tr><th>Strict box accuracy<small>mAP@50–95</small></th>${models.map(m => cell(m, `${m.map5095.toFixed(1)}%`)).join('')}</tr>
      </tbody></table>
      <h3>Per disease, Model ${sel.id}</h3>
      <table class="data-table"><thead><tr><th>Disease</th><th>Boxes</th><th>AP@50</th><th>AP@50–95</th></tr></thead><tbody>
        ${pc.map(c => `<tr><td>${esc(c.class)}</td><td>${c.instances}</td><td class="mono">${c.AP50.toFixed(4)}</td><td class="mono">${c.AP50_95.toFixed(4)}</td></tr>`).join('')}
      </tbody></table>
      <details class="calc"><summary>How it was computed (Model ${sel.id})</summary>
        <p>AP@50 of a disease = area under its precision–recall curve, a predicted box counting as correct when IoU ≥ 0.50 with a labelled box of the same disease.
          AP@50–95 averages AP over IoU 0.50, 0.55, …, 0.95.</p>
        <p><b>mAP@50</b> = (1/8) Σ AP@50 = (${pc.map(c => c.AP50.toFixed(4)).join(' + ')}) / 8 = ${(sum('AP50') / 8).toFixed(4)} = ${(ex.mAP50 * 100).toFixed(1)}%</p>
        <p><b>mAP@50–95</b> = (1/8) Σ AP@50–95 = ${(sum('AP50_95')).toFixed(4)} / 8 = ${(sum('AP50_95') / 8).toFixed(4)} = ${(ex.mAP50_95 * 100).toFixed(1)}%</p>
      </details>
    </div>`;

  const cl = sel.clusters;
  const sop2 = `
    <div class="card">
      <div class="card-head"><h2>2 · Naming the right disease (SOP 2)</h2></div>
      <p class="note">Higher precision, recall and F1 are better; fewer look-alike errors are better.</p>
      <table class="data-table center"><thead>${head}</thead><tbody>
        <tr><th>Precision<small>reported lesions that were right</small></th>${models.map(m => cell(m, `${m.precision.toFixed(1)}%`)).join('')}</tr>
        <tr><th>Recall<small>real lesions that were found</small></th>${models.map(m => cell(m, `${m.recall.toFixed(1)}%`)).join('')}</tr>
        <tr><th>F1<small>balance of the two (0–1)</small></th>${models.map(m => cell(m, m.f1.toFixed(3))).join('')}</tr>
        <tr><th>Look-alike errors<small>bumps and blisters group</small></th>${models.map(m => cell(m, `${m.withinEruptive.toFixed(1)}%`)).join('')}</tr>
        <tr><th>Look-alike errors<small>scaly patches and warts group</small></th>${models.map(m => cell(m, `${m.withinScaly.toFixed(1)}%`)).join('')}</tr>
      </tbody></table>
      <h3>Per disease, Model ${sel.id}</h3>
      <table class="data-table"><thead><tr><th>Disease</th><th>P</th><th>R</th><th>F1</th></tr></thead><tbody>
        ${pc.map(c => `<tr><td>${esc(c.class)}</td><td class="mono">${c.precision.toFixed(4)}</td><td class="mono">${c.recall.toFixed(4)}</td><td class="mono">${c.F1.toFixed(4)}</td></tr>`).join('')}
      </tbody></table>
      <details class="calc"><summary>How it was computed (Model ${sel.id})</summary>
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
      <h3>Which diseases get mixed up (confusion matrix), Model ${sel.id}</h3>
      ${confusionMatrix(sel)}
    </div>`;

  const ft = BENCH.fairnessTest;
  const sop3 = `
    <div class="card">
      <div class="card-head"><h2>3 · Lighter vs darker skin (SOP 3)</h2></div>
      <p class="note">Box accuracy (mAP@50) of each model for lighter and darker skin, and how much Model D changed it compared with the baseline (Model A).
        Skin tone comes from the ITA of each test image (very dark, type VI, excluded: ${ft.excludedVI} images; no ITA: ${ft.excludedNoIta}).</p>
      <table class="data-table center"><thead><tr><th>Group</th><th>Images</th>${models.map(m => `<th>${m.id}</th>`).join('')}<th>Change, D vs A</th></tr></thead><tbody>
        ${BENCH.fairness.map(g => `<tr><td>${esc(g.group)}<small>${esc(g.range)}</small></td><td>${g.images}</td>${models.map(m => `<td>${g.scores[m.id].toFixed(1)}%</td>`).join('')}<td class="mono">${g.dAP50 >= 0 ? '+' : ''}${(g.dAP50 * 100).toFixed(1)} pts</td></tr>`).join('')}
      </tbody></table>
      <p class="explain">${ft.p < 0.05
        ? `Model D's change compared with Model A was significantly different between the two groups (${pText(ft.p)}): ${BENCH.fairness.map(g => `${g.dAP50 >= 0 ? '+' : ''}${(g.dAP50 * 100).toFixed(1)} points for ${g.group.replace(' (ITA proxy)', '')}`).join(', ')}. H02 is rejected. The effect is small (r = ${ft.r.toFixed(2)}, and the bootstrap interval includes 0).`
        : `No significant difference between the two skin groups (${pText(ft.p)}). H02 is not rejected.`}</p>
      <details class="calc"><summary>How it was computed</summary>
        <p>Mann-Whitney U on the per-image ΔAP50 (D − A), III–V (n = ${ft.n2}) versus I–II (n = ${ft.n1}): U = ${ft.U}, ${pText(ft.p)},
          rank-biserial r = ${ft.r.toFixed(3)}; bootstrap 95% CI of the difference ${ft.ciLow.toFixed(3)} to ${ft.ciHigh.toFixed(3)}.</p>
        <p>For each test image: ΔAP50 = AP50 of Model D − AP50 of Model A. Group mAP@50 values above are computed on the images of each group.</p>
        ${BENCH.fairness.map(g => `<p>${esc(g.group)}: mAP@50 D − A = ${g.scores.D.toFixed(1)}% − ${g.scores.A.toFixed(1)}% = ${(g.dAP50 * 100).toFixed(1)} points; mean per-image ΔAP50 = ${g.meanPerImageDAP50.toFixed(4)}, median = ${g.medianPerImageDAP50.toFixed(4)}.</p>`).join('')}
      </details>
    </div>`;

  const h01 = `
    <div class="card">
      <div class="card-head"><h2>4 · Are the differences real? (H01)</h2></div>
      <p class="note">"Yes" = statistically significant (p < 0.05): the difference is unlikely to be chance. First across all four models, then Model D against each other model.</p>
      <div class="table-scroll"><table class="data-table center"><thead><tr><th>Score</th><th>Any difference among A–D?</th><th>D vs A</th><th>D vs B</th><th>D vs C</th></tr></thead><tbody>
        ${BENCH.tests.map(t => `<tr><td>${esc(PLAIN_METRIC[t.metric] || t.metric)}</td><td class="${t.friedmanP < 0.05 ? 'sig' : ''}">${t.friedmanP < 0.05 ? 'Yes' : 'No'}<small>${pText(t.friedmanP)}</small></td>
          ${t.pairs.map(p => `<td class="${p.pBonf < 0.05 ? 'sig' : ''}">${p.pBonf < 0.05 ? (t.means.D > t.means[p.vs] ? 'D better' : 'D worse') : 'No'}<small>${pText(p.pBonf)}</small></td>`).join('')}</tr>`).join('')}
      </tbody></table></div>
      <details class="calc"><summary>How it was computed</summary>
        <p>Per-image scores on the ${BENCH.testImages} test images. Friedman test across A–D, then Wilcoxon signed-rank test of D versus A, B and C
          with Bonferroni correction (p × 3); r = rank-biserial correlation (effect size).</p>
        <table class="data-table center"><thead><tr><th>Score</th><th>Friedman χ²</th><th>r, D vs A</th><th>r, D vs B</th><th>r, D vs C</th></tr></thead><tbody>
          ${BENCH.tests.map(t => `<tr><td>${esc(t.metric)}</td><td>${t.friedmanChi2}</td>${t.pairs.map(p => `<td>${p.r.toFixed(2)}</td>`).join('')}</tr>`).join('')}
        </tbody></table>
      </details>
    </div>`;

  const rt = BENCH.runtime;
  const sess = State.sessionTimings;
  const dev = State.deviceRuntime;
  const runtime = `
    <div class="card">
      <div class="card-head"><h2>5 · Speed</h2></div>
      <p class="note">How long one photo takes. GFLOPs = amount of computation per photo (same for all four models).</p>
      ${rt ? `
      <h3>On the training computer (${esc(rt.hardware.device)})</h3>
      <table class="data-table center"><thead><tr><th>Model</th><th>GFLOPs</th><th>Parameters (millions)</th><th>Model time (ms)</th><th>Total time (ms)</th></tr></thead><tbody>
        ${Object.entries(rt.models).map(([id, m]) => `<tr><td>${id}</td><td>${m.gflops_640}</td><td>${m.params_million}</td><td>${m.inference_ms_mean} ± ${m.inference_ms_sd}</td><td>${m.total_ms_mean} ± ${m.total_ms_sd}</td></tr>`).join('')}
      </tbody></table>
      <p class="note">${esc(rt.hardware.framework)}; batch ${rt.hardware.batch}, input ${rt.hardware.imgsz} px (${esc(rt.hardware.letterbox)}), ${rt.models.D.images_timed} test images after ${rt.hardware.warmup_images} warm-up images; mean ± SD.
        Total = preprocess + inference + postprocess. ${esc(rt.hardware.note)}</p>` : '<p class="note">Runtime file not found.</p>'}
      <h3>On this phone (Model D, whole analysis)</h3>
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
        <p class="note">${r.detections.length
          ? `The heatmap explains the ${r.detections.length} detected box(es) (targets: ${r.gradcam_targets.map(t => `${esc(t.label)} ${pct(t.score)}`).join(', ')}).`
          : `No box reached the ${pct(r.settings.confidence, 0)} cutoff, so nothing is counted as a lesion. The heatmap shows where the model saw weak signs of ${r.below_cutoff ? esc(r.below_cutoff.label) : 'a disease'}; the strongest (${r.below_cutoff ? pct(r.below_cutoff.confidence) : '—'}) was still below the cutoff.`}
          ${r.detections.length && r.lesion_measures && r.lesion_measures.gradcam_in_box_percent !== null ? `Share of the heatmap inside the detected boxes: ${r.lesion_measures.gradcam_in_box_percent}%.` : ''}
          A heatmap shows where the model focused. It is not a measure of accuracy and not a validated explanation of the diagnosis.</p>
        <details class="calc">
          <summary>Why does the heatmap focus on the lesion?</summary>
          <p>Grad-CAM is computed from the part of the model that names the disease (the last layer of the class head) and only for the boxes the model detected.
            For each box, it weights the model's feature maps by how much each map raised that disease's score, then keeps the positive part.</p>
          <p>The model was trained only on the labelled lesion boxes, so the features that raise a disease score are the lesion's own colour, texture and
            shape. Normal skin and background do not raise any disease score, so they stay cold. That is why the heat sits on the lesion; a heatmap
            spreading outside the lesion would mean the model also used the surroundings.</p>
        </details>
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
  try {
    const res = await fetch('assets/morphology_reference.json', { cache: 'no-store' });
    if (res.ok) MORPH_REF = await res.json();
  } catch (err) { MORPH_REF = null; }
  renderAll();
}

document.addEventListener('DOMContentLoaded', () => {
  initUser();
  $('file-input').addEventListener('change', e => { handleFile(e.target.files[0]); e.target.value = ''; });
  $('camera-input').addEventListener('change', e => { handleFile(e.target.files[0]); e.target.value = ''; });
  $('split-slider').addEventListener('input', e => setSplit(+e.target.value));
  $('toggle-boxes').addEventListener('change', e => { State.showBoxes = e.target.checked; renderWorkspace(); });
  $('toggle-heatmap').addEventListener('change', e => {
    State.showHeatmap = e.target.checked;
    if (State.showHeatmap) { State.split = 100; $('split-slider').value = 100; }
    renderWorkspace();
  });
  $('heatmap-opacity').addEventListener('input', e => { State.heatmapOpacity = +e.target.value; $('viewer-heatmap').style.opacity = State.heatmapOpacity / 100; });
  $('split-slider').value = State.split;
  renderAll();
  loadData();
  const hash = window.location.hash.replace('#', '');
  if (hash) switchTab(hash);
  window.addEventListener('hashchange', () => switchTab(window.location.hash.replace('#', '')));
});
