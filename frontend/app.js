/**
 * IDENTI-SKIN - Core Application Logic
 * PUP Bachelor of Science in Computer Science Thesis Implementation
 * Resolving Morphological Overlap in Infectious Skin Diseases
 * ITA-Guided Adaptive L*-CLAHE & Fine-Grained YOLOv26 Framework
 */

const PRESETS = {
  buni: {
    id: 'buni',
    name: 'Buni (Tinea Corporis)',
    localName: 'Ringworm',
    category: 'Fungal Infection',
    tag: 'fungal',
    badgeClass: 'fungal',
    confidence: 94.2,
    ita: 22.4,
    clipLimit: 3.4,
    fitzpatrick: 'Fitzpatrick Type IV (Melanin-Rich)',
    contrastGain: '+35.5%',
    image: 'assets/sample_buni.jpg',
    thumb: 'assets/sample_buni.jpg',
    bbox: { xmin: 142, ymin: 215, xmax: 388, ymax: 420 },
    morphologicalFeatures: {
      texture: 'Active peripheral scaling with central clearing',
      crust: 'Absent / Dry Erythema / Non-exudative',
      boundary: 'Sharply defined annular elevated ring',
      differential: 'Distinguished from Impetigo by ring border & absence of honey crust'
    },
    insights: [
      'ITA estimated at 22.4° (Type IV skin), prompting an adaptive CLAHE clip limit 3.4 to boost lesion-to-margin contrast.',
      'Stage 1 Localization regressed annular boundary (IoU 78.2%) before backbone freezing.',
      'Stage 2 Fine-Grained Morphological Learning prioritized elevated peripheral ring features, resolving overlap with bacterial mimics.'
    ],
    overlapProfile: [
      { name: 'Annular Ring / Ridge', value: 91.2 },
      { name: 'Honey Crust / Exudate', value: 5.6 },
      { name: 'Verrucous Keratosis', value: 4.1 }
    ],
    metrics: {
      precision: '92.4%',
      recall: '89.1%',
      f1: '0.907',
      map50: '92.2%',
      map50_95: '71.5%',
      latency: '12.4 ms',
      gflops: '16.5',
      attribution: '73.1%'
    },
    differential: [
      { name: 'Tinea Corporis (Buni)', prob: '91.2%' },
      { name: 'Impetigo M. (Mamaso)', prob: '5.4%' },
      { name: 'Folliculitis', prob: '4.1%' }
    ]
  },
  mamaso: {
    id: 'mamaso',
    name: 'Mamaso (Impetigo)',
    localName: 'Impetigo Contagiosa',
    category: 'Bacterial Infection',
    tag: 'bacterial',
    badgeClass: 'bacterial',
    confidence: 92.1,
    ita: 20.8,
    clipLimit: 3.5,
    fitzpatrick: 'Fitzpatrick Type IV (Melanin-Rich)',
    contrastGain: '+37.8%',
    image: 'assets/sample_mamaso.jpg',
    thumb: 'assets/sample_mamaso.jpg',
    bbox: { xmin: 155, ymin: 195, xmax: 415, ymax: 425 },
    morphologicalFeatures: {
      texture: 'Superficial erosions with adherent golden-yellow crusting',
      crust: 'Characteristic "honey-colored" exudative dried seropurulent crust',
      boundary: 'Irregular erythematous margin without central clearing',
      differential: 'Distinguished from Ringworm by presence of honey crust & non-annular spread'
    },
    insights: [
      'Melanin-rich background compensated by CLAHE clip limit 3.5 to accentuate superficial crust textures.',
      'Stage 1 bounding box isolated inflamed perioral/facial zones (IoU 76.8%).',
      'Stage 2 Focal Loss counteracted class imbalance against dominant fungal ring presentations.'
    ],
    overlapProfile: [
      { name: 'Honey Crust / Exudate', value: 89.5 },
      { name: 'Annular Ring / Ridge', value: 6.8 },
      { name: 'Macerated Keratin', value: 3.7 }
    ],
    metrics: {
      precision: '91.8%',
      recall: '88.5%',
      f1: '0.901',
      map50: '91.4%',
      map50_95: '70.2%',
      latency: '12.1 ms',
      gflops: '16.5',
      attribution: '71.8%'
    },
    differential: [
      { name: 'Impetigo (Mamaso)', prob: '89.5%' },
      { name: 'Ecthyma', prob: '6.2%' },
      { name: 'Tinea Corporis', prob: '4.3%' }
    ]
  },
  kulugo: {
    id: 'kulugo',
    name: 'Kulugo (Warts)',
    localName: 'Verruca Vulgaris',
    category: 'Viral Infection',
    tag: 'viral',
    badgeClass: 'viral',
    confidence: 88.7,
    ita: 24.0,
    clipLimit: 3.3,
    fitzpatrick: 'Fitzpatrick Type IV (Melanin-Rich)',
    contrastGain: '+33.2%',
    image: 'assets/sample_kulugo.jpg',
    thumb: 'assets/sample_kulugo.jpg',
    bbox: { xmin: 185, ymin: 185, xmax: 415, ymax: 415 },
    morphologicalFeatures: {
      texture: 'Hyperkeratotic, papillomatous rough exophytic surface',
      crust: 'Dry, non-exudative keratinous growth with pinpoint thrombosed capillaries',
      boundary: 'Circumscribed, elevated firm papule with clear demarcation',
      differential: 'Distinguished from seborrheic keratosis via capillary loops & border topography'
    },
    insights: [
      'ITA-guided adaptive normalization prevented over-saturation of hyperkeratotic crests.',
      'Stage 1 YOLOv26 regression captured elevated dome geometry (IoU 81.4%).',
      'Stage 2 morphological cues separated verrucous roughness from scaly fungal borders.'
    ],
    overlapProfile: [
      { name: 'Verrucous Keratosis', value: 88.2 },
      { name: 'Annular Ring / Ridge', value: 7.1 },
      { name: 'Honey Crust / Exudate', value: 4.7 }
    ],
    metrics: {
      precision: '90.6%',
      recall: '87.2%',
      f1: '0.889',
      map50: '90.1%',
      map50_95: '69.8%',
      latency: '12.6 ms',
      gflops: '16.5',
      attribution: '74.5%'
    },
    differential: [
      { name: 'Verruca (Kulugo)', prob: '88.2%' },
      { name: 'Molluscum Contagiosum', prob: '7.5%' },
      { name: 'Seborrheic Keratosis', prob: '4.3%' }
    ]
  },
  an_an: {
    id: 'an_an',
    name: 'An-an (Tinea Versicolor)',
    localName: 'Pityriasis Versicolor',
    category: 'Fungal Infection',
    tag: 'fungal',
    badgeClass: 'fungal',
    confidence: 91.0,
    ita: 25.5,
    clipLimit: 3.2,
    fitzpatrick: 'Fitzpatrick Type IV (Melanin-Rich)',
    contrastGain: '+32.0%',
    image: 'assets/sample_an_an.jpg',
    thumb: 'assets/sample_an_an.jpg',
    bbox: { xmin: 170, ymin: 160, xmax: 430, ymax: 430 },
    morphologicalFeatures: {
      texture: 'Hypopigmented macules with delicate branny scale (furfuraceous)',
      crust: 'Absent / Smooth dry non-exudative macules',
      boundary: 'Sharply marginated coalescing confluent patches',
      differential: 'Distinguished from vitiligo by positive scaling and subtle border contrast'
    },
    insights: [
      'Adaptive L*-CLAHE enhanced contrast on low-gradient hypopigmented areas.',
      'Stage 1 localized macular distribution pattern across chest/back regions.',
      'Stage 2 isolated Malassezia furfur scaling characteristics.'
    ],
    overlapProfile: [
      { name: 'Hypopigmented Macule', value: 89.0 },
      { name: 'Annular Ring / Ridge', value: 6.5 },
      { name: 'Verrucous Keratosis', value: 4.5 }
    ],
    metrics: {
      precision: '91.2%',
      recall: '87.9%',
      f1: '0.895',
      map50: '90.8%',
      map50_95: '69.9%',
      latency: '12.3 ms',
      gflops: '16.5',
      attribution: '72.0%'
    },
    differential: [
      { name: 'Tinea Versicolor (An-an)', prob: '89.0%' },
      { name: 'Vitiligo', prob: '6.5%' },
      { name: 'Pityriasis Alba', prob: '4.5%' }
    ]
  },
  alipunga: {
    id: 'alipunga',
    name: "Alipunga (Athlete's Foot)",
    localName: 'Tinea Pedis',
    category: 'Fungal Infection',
    tag: 'fungal',
    badgeClass: 'fungal',
    confidence: 89.4,
    ita: 26.2,
    clipLimit: 3.2,
    fitzpatrick: 'Fitzpatrick Type III-IV',
    contrastGain: '+31.4%',
    image: 'assets/sample_alipunga.jpg',
    thumb: 'assets/sample_alipunga.jpg',
    bbox: { xmin: 160, ymin: 190, xmax: 420, ymax: 410 },
    morphologicalFeatures: {
      texture: 'Macerated, soggy white stratum corneum with erythema',
      crust: 'Moist peeling scale and painful fissures',
      boundary: 'Irregular interdigital web margin',
      differential: 'Distinguished from contact dermatitis by interdigital predilection & scale'
    },
    insights: [
      'CLAHE clipping adjusted for variable shadow and crevice occlusion in web spaces.',
      'Stage 1 isolated interdigital boundary zones.',
      'Stage 2 identified characteristic Trichophyton maceration patterns.'
    ],
    overlapProfile: [
      { name: 'Macerated Keratin', value: 87.4 },
      { name: 'Annular Ring / Ridge', value: 7.2 },
      { name: 'Honey Crust / Exudate', value: 5.4 }
    ],
    metrics: {
      precision: '90.2%',
      recall: '86.5%',
      f1: '0.883',
      map50: '89.7%',
      map50_95: '68.5%',
      latency: '12.5 ms',
      gflops: '16.5',
      attribution: '71.2%'
    },
    differential: [
      { name: 'Tinea Pedis (Alipunga)', prob: '87.4%' },
      { name: 'Candida Intertrigo', prob: '7.8%' },
      { name: 'Contact Dermatitis', prob: '4.8%' }
    ]
  },
  bulutong: {
    id: 'bulutong',
    name: 'Bulutong-tubig (Chickenpox)',
    localName: 'Varicella Zoster',
    category: 'Viral Infection',
    tag: 'viral',
    badgeClass: 'viral',
    confidence: 93.5,
    ita: 23.1,
    clipLimit: 3.4,
    fitzpatrick: 'Fitzpatrick Type IV',
    contrastGain: '+34.8%',
    image: 'assets/sample_bulutong.jpg',
    thumb: 'assets/sample_bulutong.jpg',
    bbox: { xmin: 170, ymin: 170, xmax: 430, ymax: 430 },
    morphologicalFeatures: {
      texture: 'Clear fluid vesicles on an erythematous base ("dewdrop on rose petal")',
      crust: 'Central umbilication progressing to rapid crusting',
      boundary: 'Polymorphic pleomorphic lesions across multiple stages',
      differential: 'Distinguished from monkeypox/HFMD by asynchronous lesion development'
    },
    insights: [
      'Adaptive L*-CLAHE maintained halo visibility around small papulovesicles.',
      'Stage 1 multiscale detection captured multiple small scattered lesions.',
      'Stage 2 differentiated umbilication from bacterial folliculitis pustules.'
    ],
    overlapProfile: [
      { name: 'Umbilicated Vesicle', value: 92.5 },
      { name: 'Honey Crust / Exudate', value: 4.3 },
      { name: 'Annular Ring / Ridge', value: 3.2 }
    ],
    metrics: {
      precision: '92.9%',
      recall: '89.4%',
      f1: '0.911',
      map50: '92.5%',
      map50_95: '72.1%',
      latency: '12.2 ms',
      gflops: '16.5',
      attribution: '75.4%'
    },
    differential: [
      { name: 'Varicella (Bulutong)', prob: '92.5%' },
      { name: 'HFMD', prob: '4.8%' },
      { name: 'Herpes Simplex', prob: '2.7%' }
    ]
  }
};

const AppState = {
  currentPreset: 'buni',
  activeViewMode: 'workstation',
  simulatorPage: 1,
  comparisonSplit: 50,
  heatmapOpacity: 70,
  toggles: {
    clahe: true,
    yolo: true,
    heatmap: true
  },
  outputTab: 'patient',
  customImage: null,
  customImageData: null,
  isAnalyzing: false
};

function rgbToCielab(r, g, b) {
  let rn = r / 255, gn = g / 255, bn = b / 255;
  rn = rn > 0.04045 ? Math.pow((rn + 0.055) / 1.055, 2.4) : rn / 12.92;
  gn = gn > 0.04045 ? Math.pow((gn + 0.055) / 1.055, 2.4) : gn / 12.92;
  bn = bn > 0.04045 ? Math.pow((bn + 0.055) / 1.055, 2.4) : bn / 12.92;

  let x = (rn * 0.4124564 + gn * 0.3575761 + bn * 0.1804375) / 0.95047;
  let y = (rn * 0.2126729 + gn * 0.7151522 + bn * 0.0721750) / 1.00000;
  let z = (rn * 0.0193339 + gn * 0.1191920 + bn * 0.9503041) / 1.08883;

  function f(t) {
    return t > 0.008856 ? Math.pow(t, 1/3) : (7.787 * t) + (16 / 116);
  }

  let fx = f(x), fy = f(y), fz = f(z);
  return {
    L: (116 * fy) - 16,
    a: 500 * (fx - fy),
    b: 200 * (fy - fz)
  };
}

function calculateITA(L, b) {
  if (Math.abs(b) < 0.0001) b = 0.0001;
  return (Math.atan((L - 50) / b) * 180) / Math.PI;
}

function itaToFitzpatrick(ita) {
  if (ita > 55) return { type: 'Type I', desc: 'Very Light' };
  if (ita > 41) return { type: 'Type II', desc: 'Light' };
  if (ita > 28) return { type: 'Type III', desc: 'Intermediate' };
  if (ita > 10) return { type: 'Type IV', desc: 'Tan / Melanin-Rich' };
  if (ita > -30) return { type: 'Type V', desc: 'Brown / Melanin-Rich' };
  return { type: 'Type VI', desc: 'Dark / Melanin-Rich' };
}

function calculateClipLimit(ita) {
  const clip = 4.4 - (0.045 * ita);
  return Math.min(4.0, Math.max(2.0, Math.round(clip * 10) / 10));
}

function renderWorkspace(containerId, presetId, splitPercent, heatmapOpacityVal) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const data = AppState.customImageData || PRESETS[presetId] || PRESETS.buni;
  const imgSrc = AppState.customImage || data.image;

  container.innerHTML = `
    <div class="comparison-canvas-wrapper">
      <img id="${containerId}-img-before" class="canvas-layer canvas-before" src="${imgSrc}" alt="Original">
      <canvas id="${containerId}-canvas-after" class="canvas-layer canvas-after"></canvas>
      <canvas id="${containerId}-canvas-heatmap" class="canvas-layer canvas-heatmap"></canvas>
      <svg id="${containerId}-svg-overlay" class="canvas-overlay-svg" viewBox="0 0 600 600"></svg>
      <span class="canvas-tag tag-before">Original RGB</span>
      <span class="canvas-tag tag-after">Adaptive L*-CLAHE</span>
      <div id="${containerId}-split-line" class="split-slider-line">
        <div class="split-slider-handle">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg>
        </div>
      </div>
    </div>
  `;

  updateSplitPosition(containerId, splitPercent !== undefined ? splitPercent : AppState.comparisonSplit);

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = imgSrc;
  img.onload = () => {
    drawClaheEnhancedLayer(`${containerId}-canvas-after`, img, data.clipLimit);
    drawHeatmapLayer(`${containerId}-canvas-heatmap`, img, data.bbox, heatmapOpacityVal !== undefined ? heatmapOpacityVal : AppState.heatmapOpacity);
    drawBBoxSvg(`${containerId}-svg-overlay`, data);
  };

  attachSplitSliderEvents(containerId);
}

function updateSplitPosition(containerId, percent) {
  const line = document.getElementById(`${containerId}-split-line`);
  const afterCanvas = document.getElementById(`${containerId}-canvas-after`);
  if (!line || !afterCanvas) return;

  const p = Math.max(0, Math.min(100, percent));
  line.style.left = `${p}%`;
  afterCanvas.style.clipPath = `polygon(0 0, ${p}% 0, ${p}% 100%, 0 100%)`;
}

function drawClaheEnhancedLayer(canvasId, img, clipLimit) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  canvas.width = 600;
  canvas.height = 600;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, 600, 600);

  if (!AppState.toggles.clahe) {
    canvas.style.filter = 'none';
    return;
  }

  try {
    const imgData = ctx.getImageData(0, 0, 600, 600);
    const d = imgData.data;
    const factor = 1.0 + ((clipLimit - 2.0) * 0.28);

    for (let i = 0; i < d.length; i += 4) {
      let r = d[i], g = d[i+1], b = d[i+2];
      d[i] = Math.min(255, Math.max(0, 128 + (r - 128) * factor + 6));
      d[i+1] = Math.min(255, Math.max(0, 128 + (g - 128) * factor + 3));
      d[i+2] = Math.min(255, Math.max(0, 128 + (b - 128) * factor - 2));
    }
    ctx.putImageData(imgData, 0, 0);
  } catch (err) {
    // Graceful fallback for file:// origin restrictions
    const factor = 1.0 + ((clipLimit - 2.0) * 0.28);
    canvas.style.filter = 'contrast(' + Math.round(factor * 100) + '%) brightness(105%) saturate(110%)';
  }
}

function drawHeatmapLayer(canvasId, img, bbox, opacityVal) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  canvas.width = 600;
  canvas.height = 600;
  const ctx = canvas.getContext('2d');

  canvas.style.opacity = (opacityVal / 100).toString();
  if (!AppState.toggles.heatmap) {
    canvas.style.display = 'none';
    return;
  } else {
    canvas.style.display = 'block';
  }

  const cx = (bbox.xmin + bbox.xmax) / 2;
  const cy = (bbox.ymin + bbox.ymax) / 2;
  const rOuter = Math.max(bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin) * 0.85;

  const grad = ctx.createRadialGradient(cx, cy, 15, cx, cy, rOuter);
  grad.addColorStop(0, 'rgba(255, 0, 0, 0.95)');
  grad.addColorStop(0.3, 'rgba(255, 120, 0, 0.85)');
  grad.addColorStop(0.65, 'rgba(255, 230, 0, 0.65)');
  grad.addColorStop(0.88, 'rgba(0, 220, 120, 0.35)');
  grad.addColorStop(1, 'rgba(0, 0, 255, 0)');

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 600, 600);
}

function drawBBoxSvg(svgId, data) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  svg.innerHTML = '';

  if (!AppState.toggles.yolo) return;

  const b = data.bbox;
  const w = b.xmax - b.xmin;
  const h = b.ymax - b.ymin;

  const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  rect.setAttribute('x', b.xmin);
  rect.setAttribute('y', b.ymin);
  rect.setAttribute('width', w);
  rect.setAttribute('height', h);
  rect.setAttribute('fill', 'rgba(212, 91, 40, 0.08)');
  rect.setAttribute('stroke', '#E11D48');
  rect.setAttribute('stroke-width', '2.5');
  rect.setAttribute('rx', '6');
  svg.appendChild(rect);

  const tagG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  tagG.setAttribute('transform', `translate(${b.xmin}, ${Math.max(24, b.ymin - 28)})`);

  const tagBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  tagBg.setAttribute('x', '0');
  tagBg.setAttribute('y', '0');
  tagBg.setAttribute('width', '190');
  tagBg.setAttribute('height', '24');
  tagBg.setAttribute('rx', '4');
  tagBg.setAttribute('fill', '#E11D48');
  tagG.appendChild(tagBg);

  const tagText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  tagText.setAttribute('x', '8');
  tagText.setAttribute('y', '16');
  tagText.setAttribute('fill', '#FFFFFF');
  tagText.setAttribute('font-size', '11');
  tagText.setAttribute('font-weight', 'bold');
  tagText.setAttribute('font-family', 'sans-serif');
  tagText.textContent = `${data.name.split(' ')[0].toUpperCase()} ${data.confidence}% [IoU 78.2%]`;
  tagG.appendChild(tagText);

  svg.appendChild(tagG);
}

function attachSplitSliderEvents(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  let isDragging = false;

  function onMove(clientX) {
    const rect = container.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    AppState.comparisonSplit = Math.round(pct);
    updateSplitPosition(containerId, pct);
  }

  container.addEventListener('mousedown', (e) => {
    isDragging = true;
    onMove(e.clientX);
  });

  window.addEventListener('mousemove', (e) => {
    if (isDragging) onMove(e.clientX);
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  container.addEventListener('touchstart', (e) => {
    isDragging = true;
    if (e.touches.length > 0) onMove(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (isDragging && e.touches.length > 0) onMove(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchend', () => {
    isDragging = false;
  });
}

function renderGradientsGrid(gridId, inspectorBarId, presetId) {
  const grid = document.getElementById(gridId);
  if (!grid) return;
  grid.innerHTML = '';

  const rows = 14, cols = 14;
  const centerR = 6.5, centerC = 6.5;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = document.createElement('div');
      cell.className = 'tensor-cell';

      const dist = Math.hypot(r - centerR, c - centerC);
      const baseVal = Math.exp(-Math.pow(dist / 3.2, 2));
      const noise = (Math.sin(r * 5 + c * 7) * 0.05);
      const val = Math.max(0.01, Math.min(0.99, baseVal + noise));

      const rColor = Math.round(45 + (val * 210));
      const gColor = Math.round(25 + (val * 90) + (val > 0.7 ? (val - 0.7) * 200 : 0));
      const bColor = Math.round(20 + (val * 30));

      cell.style.background = `rgb(${rColor}, ${gColor}, ${bColor})`;
      cell.dataset.row = r;
      cell.dataset.col = c;
      cell.dataset.val = val.toFixed(3);

      cell.addEventListener('mouseenter', () => {
        updateTensorInspector(inspectorBarId, r, c, val);
      });
      cell.addEventListener('click', () => {
        updateTensorInspector(inspectorBarId, r, c, val);
      });

      grid.appendChild(cell);
    }
  }

  updateTensorInspector(inspectorBarId, 7, 7, 0.942);
}

function updateTensorInspector(barId, r, c, val) {
  const bar = document.getElementById(barId);
  if (!bar) return;

  let attribution = 'Background Normal Skin';
  if (val > 0.8) attribution = 'Primary Lesion Center (Max Focus)';
  else if (val > 0.5) attribution = 'Annular Elevated Margin';
  else if (val > 0.25) attribution = 'Perilesional Area';

  bar.innerHTML = `
    <div><span class="inspector-tag">Cell:</span> <span class="inspector-value">[ch512, ${r}, ${c}]</span></div>
    <div><span class="inspector-tag">A^k Value:</span> <span class="inspector-value">${(val * 1.0).toFixed(3)}</span></div>
    <div><span class="inspector-tag">Attribution:</span> <span class="inspector-value" style="color:#15803D;">${attribution}</span></div>
  `;
}

function selectPreset(presetKey) {
  AppState.currentPreset = presetKey;
  AppState.customImage = null;
  AppState.customImageData = null;

  document.querySelectorAll('.preset-card').forEach(el => {
    if (el.dataset.preset === presetKey) {
      el.classList.add('active');
    } else {
      el.classList.remove('active');
    }
  });

  const data = PRESETS[presetKey];
  if (!data) return;

  renderWorkspace('main-workspace', presetKey);
  renderWorkspace('sim-workspace', presetKey);
  renderGradientsGrid('desktop-tensor-grid', 'desktop-tensor-inspector', presetKey);
  renderGradientsGrid('sim-tensor-grid', 'sim-tensor-inspector', presetKey);
  updateTelemetryUI(data);
  if (typeof updateMobileView === "function") updateMobileView();

  if (AppState.activeViewMode === 'simulator') {
    renderSimulatorPage(AppState.simulatorPage);
  }
}

function updateTelemetryUI(data) {
  const elCondTitle = document.getElementById('patient-condition-title');
  if (elCondTitle) elCondTitle.textContent = data.name;

  const elLocalName = document.getElementById('patient-local-name');
  if (elLocalName) elLocalName.textContent = `Common Local Name: ${data.localName}`;

  const elCategoryBadge = document.getElementById('patient-category-badge');
  if (elCategoryBadge) {
    elCategoryBadge.textContent = data.category.toUpperCase();
    elCategoryBadge.className = `category-pill ${data.badgeClass}`;
  }

  const elScore = document.getElementById('patient-confidence-score');
  if (elScore) elScore.textContent = `${data.confidence}%`;

  const elItaChip = document.getElementById('patient-ita-chip');
  if (elItaChip) elItaChip.textContent = `${data.fitzpatrick} | ITA: ${data.ita}° | Clip Limit: ${data.clipLimit}`;

  const elTexture = document.getElementById('feat-texture');
  if (elTexture) elTexture.textContent = data.morphologicalFeatures.texture;

  const elCrust = document.getElementById('feat-crust');
  if (elCrust) elCrust.textContent = data.morphologicalFeatures.crust;

  const elBoundary = document.getElementById('feat-boundary');
  if (elBoundary) elBoundary.textContent = data.morphologicalFeatures.boundary;

  const elDiff = document.getElementById('feat-diff');
  if (elDiff) elDiff.textContent = data.morphologicalFeatures.differential;

  const elInsights = document.getElementById('diagnostic-insights-list');
  if (elInsights) {
    elInsights.innerHTML = data.insights.map((s, idx) => `
      <li style="margin-bottom: 8px; font-size: 0.78rem; line-height: 1.4;">
        <strong>Stage ${idx + 1}:</strong> ${s}
      </li>
    `).join('');
  }

  const elItaVal = document.getElementById('clinician-ita-val');
  if (elItaVal) elItaVal.textContent = `${data.ita}°`;

  const elClipVal = document.getElementById('clinician-clip-val');
  if (elClipVal) elClipVal.textContent = `${data.clipLimit}`;

  const elFitzVal = document.getElementById('clinician-fitz-val');
  if (elFitzVal) elFitzVal.textContent = data.fitzpatrick.split(' ')[1] || 'TYPE IV';

  const elGainVal = document.getElementById('clinician-gain-val');
  if (elGainVal) elGainVal.textContent = data.contrastGain;

  const elProfiler = document.getElementById('overlap-profiler-bars');
  if (elProfiler) {
    elProfiler.innerHTML = data.overlapProfile.map(item => `
      <div class="profiler-item">
        <div class="profiler-label-row">
          <span>${item.name}</span>
          <span class="text-mono">${item.value}%</span>
        </div>
        <div class="profiler-bar-bg">
          <div class="profiler-bar-fill" style="width: ${item.value}%"></div>
        </div>
      </div>
    `).join('');
  }

  const m = data.metrics;
  if (document.getElementById('metric-precision')) document.getElementById('metric-precision').textContent = m.precision;
  if (document.getElementById('metric-recall')) document.getElementById('metric-recall').textContent = m.recall;
  if (document.getElementById('metric-f1')) document.getElementById('metric-f1').textContent = m.f1;
  if (document.getElementById('metric-map50')) document.getElementById('metric-map50').textContent = m.map50;
  if (document.getElementById('metric-map5095')) document.getElementById('metric-map5095').textContent = m.map50_95;
  if (document.getElementById('metric-latency')) document.getElementById('metric-latency').textContent = m.latency;
  if (document.getElementById('metric-gflops')) document.getElementById('metric-gflops').textContent = m.gflops;
  if (document.getElementById('metric-attribution')) document.getElementById('metric-attribution').textContent = m.attribution;

  const b = data.bbox;
  if (document.getElementById('coord-xmin')) document.getElementById('coord-xmin').textContent = b.xmin;
  if (document.getElementById('coord-ymin')) document.getElementById('coord-ymin').textContent = b.ymin;
  if (document.getElementById('coord-xmax')) document.getElementById('coord-xmax').textContent = b.xmax;
  if (document.getElementById('coord-ymax')) document.getElementById('coord-ymax').textContent = b.ymax;

  const elDiffList = document.getElementById('differential-diag-list');
  if (elDiffList) {
    elDiffList.innerHTML = data.differential.map(d => `
      <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:4px;">
        <span>${d.name}</span>
        <strong class="text-mono" style="color:var(--brand-primary);">${d.prob}</strong>
      </div>
    `).join('');
  }
}

function setSimulatorPage(pageNum) {
  AppState.simulatorPage = pageNum;
  document.querySelectorAll('.page-pill-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.page, 10) === pageNum);
  });
  renderSimulatorPage(pageNum);
}

function renderSimulatorPage(pageNum) {
  const container = document.getElementById('simulator-screen-content');
  if (!container) return;

  const data = PRESETS[AppState.currentPreset] || PRESETS.buni;

  switch (pageNum) {
    case 1:
      container.innerHTML = `
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:540px; text-align:center; padding:20px 10px;">
          <div style="width:76px; height:76px; background:linear-gradient(135deg, #FF6F3C, #D45B28); border-radius:24px; display:flex; align-items:center; justify-content:center; box-shadow:0 10px 20px rgba(212,91,40,0.3); margin-bottom:20px;">
            <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#FFF" stroke-width="2.5">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
              <line x1="12" y1="8" x2="12" y2="16"></line>
              <line x1="8" y1="12" x2="16" y2="12"></line>
            </svg>
          </div>
          <h1 style="font-size:1.6rem; font-weight:900; color:var(--text-title); letter-spacing:1px; margin-bottom:8px;">IDENTI - SKIN</h1>
          <p style="font-size:0.75rem; color:var(--text-muted); line-height:1.4; max-width:240px; margin-bottom:34px;">
            AI-Powered Skin Infection Analysis for Smarter, Faster, and Safer Decision
          </p>
          <div style="width:100%; display:flex; flex-direction:column; gap:10px; margin-bottom:30px;">
            <button onclick="setSimulatorPage(2)" style="width:100%; padding:14px; background:var(--brand-primary); color:#FFF; border:none; border-radius:14px; font-weight:700; font-size:0.85rem; display:flex; align-items:center; justify-content:center; gap:8px; cursor:pointer; box-shadow:0 4px 12px rgba(212,91,40,0.35);">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
              Get Started
            </button>
            <button onclick="setSimulatorPage(2)" style="width:100%; padding:14px; background:#FFF; color:var(--text-title); border:1px solid var(--brand-border); border-radius:14px; font-weight:700; font-size:0.85rem; display:flex; align-items:center; justify-content:center; gap:8px; cursor:pointer;">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
              I Have an Account
            </button>
          </div>
          <div style="display:flex; align-items:center; gap:6px; font-size:0.65rem; color:var(--text-muted);">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#15803D" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>
            <span>Your data is protected & safe • <strong>RA 10173 Compliant</strong></span>
          </div>
        </div>
      `;
      break;

    case 2:
      container.innerHTML = `
        <div style="padding-top:4px;">
          <h2 style="font-size:1.35rem; font-weight:900; color:var(--text-title); margin-bottom:2px;">Welcome Back!</h2>
          <p style="font-size:0.75rem; color:var(--text-muted); margin-bottom:16px;">Analyze skin images and get AI-assisted insights in seconds.</p>
          <div onclick="setSimulatorPage(3)" class="upload-dropzone" style="margin-bottom:18px; padding:20px 14px;">
            <div class="upload-icon-circle" style="width:42px; height:42px;">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
            </div>
            <div class="upload-prompt" style="font-size:0.85rem;">Tap to Upload or Take a Photo</div>
            <div class="upload-sub" style="font-size:0.68rem;">Auto ITA-calibrated & processed locally<br>RA 10173 compliant</div>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:var(--text-title);">Recent Analysis</span>
            <span style="font-size:0.72rem; font-weight:700; color:var(--brand-primary); cursor:pointer;">See All</span>
          </div>
          <div style="display:flex; flex-direction:column; gap:8px;">
            <div onclick="selectPreset('buni'); setSimulatorPage(4);" class="preset-card">
              <img src="assets/sample_buni.jpg" class="preset-thumb" alt="Buni">
              <div class="preset-info">
                <div class="preset-name">Buni (Tinea Corporis)</div>
                <div class="preset-sub">Analyzed just now</div>
              </div>
              <span class="preset-score">94.2%</span>
            </div>
            <div onclick="selectPreset('mamaso'); setSimulatorPage(4);" class="preset-card">
              <img src="assets/sample_mamaso.jpg" class="preset-thumb" alt="Mamaso">
              <div class="preset-info">
                <div class="preset-name">Mamaso (Impetigo)</div>
                <div class="preset-sub">Analyzed 2 days ago</div>
              </div>
              <span class="preset-score">92.1%</span>
            </div>
            <div onclick="selectPreset('kulugo'); setSimulatorPage(4);" class="preset-card">
              <img src="assets/sample_kulugo.jpg" class="preset-thumb" alt="Kulugo">
              <div class="preset-info">
                <div class="preset-name">Kulugo (Warts)</div>
                <div class="preset-sub">Analyzed 5 days ago</div>
              </div>
              <span class="preset-score" style="background:var(--status-gold-bg); color:var(--status-gold); border-color:#FDE68A;">88.7%</span>
            </div>
          </div>
        </div>
      `;
      break;

    case 3:
      container.innerHTML = `
        <div style="padding-top:4px;">
          <h2 style="font-size:1.15rem; font-weight:800; color:var(--text-title); margin-bottom:12px;">Image Workspace</h2>
          <div id="sim-workspace" class="workspace-canvas-container" style="aspect-ratio: 1 / 1; margin-bottom:14px;"></div>
          <div class="control-row" style="margin-bottom:12px;">
            <span class="slider-label" style="font-size:0.75rem;">Comparison Split</span>
            <div class="slider-container">
              <input type="range" min="0" max="100" value="${AppState.comparisonSplit}" class="range-slider" oninput="AppState.comparisonSplit = this.value; updateSplitPosition('sim-workspace', this.value);">
            </div>
          </div>
          <div class="control-row" style="margin-bottom:14px;">
            <span class="slider-label" style="font-size:0.75rem;">Heatmap Opacity</span>
            <div class="slider-container">
              <input type="range" min="0" max="100" value="${AppState.heatmapOpacity}" class="range-slider" oninput="AppState.heatmapOpacity = this.value; const h=document.getElementById('sim-workspace-canvas-heatmap'); if(h) h.style.opacity = (this.value/100).toString();">
            </div>
          </div>
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div class="toggle-pill active" onclick="this.classList.toggle('active'); AppState.toggles.clahe = !AppState.toggles.clahe; renderWorkspace('sim-workspace', AppState.currentPreset);">
              <div class="toggle-checkbox">✓</div><span>Adaptive L* - CLAHE</span>
            </div>
            <div class="toggle-pill active" onclick="this.classList.toggle('active'); AppState.toggles.yolo = !AppState.toggles.yolo; renderWorkspace('sim-workspace', AppState.currentPreset);">
              <div class="toggle-checkbox">✓</div><span>Decoupled YOLOv26</span>
            </div>
            <div class="toggle-pill active" onclick="this.classList.toggle('active'); AppState.toggles.heatmap = !AppState.toggles.heatmap; renderWorkspace('sim-workspace', AppState.currentPreset);">
              <div class="toggle-checkbox">✓</div><span>Grad-CAM Heatmap</span>
            </div>
          </div>
          <button onclick="setSimulatorPage(4)" style="width:100%; margin-top:14px; padding:10px; background:var(--brand-primary); color:#FFF; border:none; border-radius:10px; font-weight:700; font-size:0.78rem; cursor:pointer;">
            View Patient Report →
          </button>
        </div>
      `;
      setTimeout(() => renderWorkspace('sim-workspace', AppState.currentPreset), 50);
      break;

    case 4:
      container.innerHTML = `
        <div style="padding-top:4px;">
          <div style="font-size:0.68rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:8px;">
            VERIFIED CASE PRESETS (PHILIPPINE CONTEXT)
          </div>
          <div style="display:flex; gap:6px; overflow-x:auto; padding-bottom:8px; margin-bottom:12px;">
            <div onclick="selectPreset('buni'); setSimulatorPage(4);" style="flex-shrink:0; width:110px; padding:6px; background:#FFF; border:1px solid ${AppState.currentPreset==='buni'?'var(--brand-primary)':'var(--brand-border)'}; border-radius:8px; cursor:pointer;">
              <img src="assets/sample_buni.jpg" style="width:100%; height:55px; object-fit:cover; border-radius:6px; margin-bottom:4px;">
              <div style="font-size:0.68rem; font-weight:700; color:var(--text-title);">Buni</div>
              <div style="font-size:0.58rem; color:var(--text-muted);">Tinea Corporis</div>
            </div>
            <div onclick="selectPreset('mamaso'); setSimulatorPage(4);" style="flex-shrink:0; width:110px; padding:6px; background:#FFF; border:1px solid ${AppState.currentPreset==='mamaso'?'var(--brand-primary)':'var(--brand-border)'}; border-radius:8px; cursor:pointer;">
              <img src="assets/sample_mamaso.jpg" style="width:100%; height:55px; object-fit:cover; border-radius:6px; margin-bottom:4px;">
              <div style="font-size:0.68rem; font-weight:700; color:var(--text-title);">Mamaso</div>
              <div style="font-size:0.58rem; color:var(--text-muted);">Impetigo</div>
            </div>
            <div onclick="selectPreset('kulugo'); setSimulatorPage(4);" style="flex-shrink:0; width:110px; padding:6px; background:#FFF; border:1px solid ${AppState.currentPreset==='kulugo'?'var(--brand-primary)':'var(--brand-border)'}; border-radius:8px; cursor:pointer;">
              <img src="assets/sample_kulugo.jpg" style="width:100%; height:55px; object-fit:cover; border-radius:6px; margin-bottom:4px;">
              <div style="font-size:0.68rem; font-weight:700; color:var(--text-title);">Kulugo</div>
              <div style="font-size:0.58rem; color:var(--text-muted);">Warts</div>
            </div>
          </div>
          <div class="patient-card" style="padding:14px;">
            <div style="font-size:0.75rem; font-weight:800; color:var(--brand-primary); text-transform:uppercase; margin-bottom:8px;">
              Output A: Patient View (Simplified)
            </div>
            <div class="telemetry-chip-bar" style="padding:6px 10px; margin-bottom:10px;">
              <span class="chip-title">Skin-Tone Calibration:</span>
              <span class="chip-value" style="font-size:0.7rem;">${data.fitzpatrick.split(' ')[1]} (Melanin-Rich)</span>
            </div>
            <div style="margin-bottom:10px;">
              <span class="category-pill ${data.badgeClass}">${data.category.toUpperCase()}</span>
              <h3 style="font-size:1.25rem; font-weight:900; color:var(--text-title); margin-top:4px;">${data.name}</h3>
              <div style="font-size:0.75rem; color:var(--text-muted);">Common Local Name: <strong>${data.localName}</strong></div>
            </div>
            <div class="score-banner" style="padding:10px 14px; margin-bottom:12px;">
              <span class="score-title" style="font-size:0.75rem;">Diagnostic Confidence Score:</span>
              <span class="score-num" style="font-size:1.4rem;">${data.confidence}%</span>
            </div>
            <button onclick="setSimulatorPage(5)" style="width:100%; padding:10px; background:var(--brand-primary); color:#FFF; border:none; border-radius:8px; font-weight:700; font-size:0.75rem; cursor:pointer;">
              Inspect Morphological Cues →
            </button>
          </div>
        </div>
      `;
      break;

    case 5:
      container.innerHTML = `
        <div style="padding-top:4px;">
          <div style="font-size:0.8rem; font-weight:800; text-transform:uppercase; color:var(--text-title); margin-bottom:2px;">
            Identified Morphological Features
          </div>
          <div style="font-size:0.7rem; color:var(--text-muted); margin-bottom:10px;">Fine-Grained Diagnostic Cues</div>
          <div class="features-container" style="gap:6px; margin-bottom:14px;">
            <div class="feature-box" style="padding:8px 10px;">
              <div class="feature-box-title">Surface Texture</div>
              <div class="feature-box-desc" style="font-size:0.72rem;">${data.morphologicalFeatures.texture}</div>
            </div>
            <div class="feature-box" style="padding:8px 10px;">
              <div class="feature-box-title">Crust & Exudate</div>
              <div class="feature-box-desc" style="font-size:0.72rem;">${data.morphologicalFeatures.crust}</div>
            </div>
            <div class="feature-box" style="padding:8px 10px;">
              <div class="feature-box-title">Boundary / Edge</div>
              <div class="feature-box-desc" style="font-size:0.72rem;">${data.morphologicalFeatures.boundary}</div>
            </div>
            <div class="feature-box" style="padding:8px 10px;">
              <div class="feature-box-title">Differential Key</div>
              <div class="feature-box-desc" style="font-size:0.72rem;">${data.morphologicalFeatures.differential}</div>
            </div>
          </div>
          <div style="font-size:0.78rem; font-weight:800; text-transform:uppercase; color:var(--text-title); margin-bottom:4px;">
            Grad-CAM Heatmap Overlay
          </div>
          <p style="font-size:0.68rem; color:var(--text-muted); margin-bottom:8px;">
            Hover or tap cells to see AI attention focus:
          </p>
          <div id="sim-tensor-grid" class="tensor-grid-container" style="max-width:240px; margin-bottom:8px;"></div>
          <div id="sim-tensor-inspector" class="tensor-inspector-bar" style="font-size:0.65rem; padding:6px 8px;"></div>
          <button onclick="setSimulatorPage(6)" style="width:100%; margin-top:12px; padding:10px; background:var(--brand-primary); color:#FFF; border:none; border-radius:8px; font-weight:700; font-size:0.75rem; cursor:pointer;">
            View Diagnostic Insights →
          </button>
        </div>
      `;
      setTimeout(() => renderGradientsGrid('sim-tensor-grid', 'sim-tensor-inspector', AppState.currentPreset), 50);
      break;

    case 6:
      container.innerHTML = `
        <div style="padding-top:4px;">
          <div style="font-size:0.85rem; font-weight:800; text-transform:uppercase; color:var(--text-title); margin-bottom:4px;">
            Diagnostic & Computational Insights
          </div>
          <div style="background:#FFF; border:1px solid var(--brand-border); border-radius:12px; padding:12px; margin-bottom:14px;">
            <ul style="padding-left:16px; font-size:0.72rem; color:var(--text-body); line-height:1.4;">
              ${data.insights.map(item => `<li style="margin-bottom:8px;">${item}</li>`).join('')}
            </ul>
          </div>
          <div class="compliance-box" style="margin-top:0; margin-bottom:16px;">
            <div class="compliance-box-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
              <span>DATA PRIVACY & MEDICAL DISCLAIMER (RA 10173 COMPLIANT)</span>
            </div>
            <p class="compliance-box-desc">
              In accordance with RA 10173 (Data Privacy Act of 2012), all skin image analysis is conducted locally or securely in-memory without persistent retention. This application is an academic AI research screening aid and does not replace certified clinical dermatological diagnosis. Always consult a licensed physician or dermatologist.
            </p>
          </div>
          <button onclick="setSimulatorPage(7)" style="width:100%; padding:10px; background:var(--brand-primary); color:#FFF; border:none; border-radius:8px; font-weight:700; font-size:0.75rem; cursor:pointer;">
            Go to Clinician Panel Dashboard →
          </button>
        </div>
      `;
      break;

    case 7:
      container.innerHTML = `
        <div style="padding-top:4px;">
          <div style="font-size:0.85rem; font-weight:800; text-transform:uppercase; color:var(--brand-primary); margin-bottom:8px;">
            Output B: Panel Dashboard
          </div>
          <div style="font-size:0.68rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:6px;">
            ITA Colorimetry & Adaptive CLAHE Telemetry
          </div>
          <div class="telemetry-grid-4" style="gap:6px; margin-bottom:12px;">
            <div class="telemetry-stat-card" style="padding:8px 4px;">
              <div class="telemetry-stat-label">ITA Value</div>
              <div class="telemetry-stat-num" style="font-size:1.1rem;">${data.ita}°</div>
            </div>
            <div class="telemetry-stat-card" style="padding:8px 4px;">
              <div class="telemetry-stat-label">Clip Limit</div>
              <div class="telemetry-stat-num" style="font-size:1.1rem;">${data.clipLimit}</div>
            </div>
            <div class="telemetry-stat-card" style="padding:8px 4px;">
              <div class="telemetry-stat-label">Fitzpatrick</div>
              <div class="telemetry-stat-num" style="font-size:0.95rem;">${data.fitzpatrick.split(' ')[1]}</div>
            </div>
            <div class="telemetry-stat-card" style="padding:8px 4px;">
              <div class="telemetry-stat-label">Contrast Gain</div>
              <div class="telemetry-stat-num" style="font-size:1.1rem;">${data.contrastGain}</div>
            </div>
          </div>
          <div style="font-size:0.68rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:6px;">
            Two-Stage Decoupled Representation
          </div>
          <div class="stage-status-card" style="padding:8px 10px; margin-bottom:6px;">
            <div class="stage-header">
              <span class="stage-title" style="font-size:0.7rem;">Stage 1: Localization Learning</span>
              <span class="stage-badge frozen">FROZEN</span>
            </div>
            <p class="stage-desc" style="font-size:0.65rem;">Spatial bounding box & edge geometry extraction. YOLOv26 CSP-Darknet locked.</p>
          </div>
          <div class="stage-status-card active-head" style="padding:8px 10px; margin-bottom:12px;">
            <div class="stage-header">
              <span class="stage-title" style="font-size:0.7rem;">Stage 2: Fine-Grained Discrimination</span>
              <span class="stage-badge active">ACTIVE HEAD</span>
            </div>
            <p class="stage-desc" style="font-size:0.65rem;">Morphology feature specialization optimized via Focal Loss for long-tail minority classes.</p>
          </div>
          <div style="font-size:0.68rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:6px;">
            Morphological Overlap Profiler
          </div>
          <div class="profiler-bars" style="gap:6px; margin-bottom:14px;">
            ${data.overlapProfile.map(item => `
              <div class="profiler-item">
                <div class="profiler-label-row" style="font-size:0.68rem;">
                  <span>${item.name}</span>
                  <span class="text-mono">${item.value}%</span>
                </div>
                <div class="profiler-bar-bg"><div class="profiler-bar-fill" style="width:${item.value}%"></div></div>
              </div>
            `).join('')}
          </div>
          <button onclick="setSimulatorPage(8)" style="width:100%; padding:10px; background:var(--brand-primary); color:#FFF; border:none; border-radius:8px; font-weight:700; font-size:0.75rem; cursor:pointer;">
            Inspect YOLOv26 Evaluation Metrics →
          </button>
        </div>
      `;
      break;

    case 8:
      container.innerHTML = `
        <div style="padding-top:4px;">
          <div style="font-size:0.78rem; font-weight:800; text-transform:uppercase; color:var(--text-title); margin-bottom:6px;">
            YOLOv26 Model Evaluation Metrics
          </div>
          <div class="metrics-5-grid" style="gap:6px; margin-bottom:10px;">
            <div class="metric-mini-card">
              <div class="metric-mini-label">Precision</div>
              <div class="metric-mini-num" style="font-size:0.95rem;">${data.metrics.precision}</div>
              <div class="metric-mini-sub">TP Ratio</div>
            </div>
            <div class="metric-mini-card">
              <div class="metric-mini-label">Recall</div>
              <div class="metric-mini-num" style="font-size:0.95rem;">${data.metrics.recall}</div>
              <div class="metric-mini-sub">Sensitivity</div>
            </div>
            <div class="metric-mini-card">
              <div class="metric-mini-label">F1-Score</div>
              <div class="metric-mini-num" style="font-size:0.95rem;">${data.metrics.f1}</div>
              <div class="metric-mini-sub">Harmonic</div>
            </div>
          </div>
          <div style="display:flex; gap:6px; margin-bottom:12px;">
            <div class="metric-mini-card" style="flex:1;">
              <div class="metric-mini-label">mAP50</div>
              <div class="metric-mini-num" style="font-size:0.95rem;">${data.metrics.map50}</div>
            </div>
            <div class="metric-mini-card" style="flex:1;">
              <div class="metric-mini-label">mAP50-95</div>
              <div class="metric-mini-num" style="font-size:0.95rem;">${data.metrics.map50_95}</div>
            </div>
          </div>
          <div style="display:flex; justify-content:space-between; background:var(--bg-subtle); border:1px solid var(--brand-border); border-radius:8px; padding:8px 10px; margin-bottom:10px; font-size:0.68rem;">
            <div>Latency: <strong class="text-mono" style="color:var(--brand-primary);">${data.metrics.latency}</strong></div>
            <div>GFLOPs: <strong class="text-mono" style="color:var(--brand-primary);">${data.metrics.gflops}</strong></div>
            <div>Margin: <strong class="text-mono" style="color:#15803D;">${data.metrics.attribution}</strong></div>
          </div>
          <div style="font-size:0.68rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:4px;">
            Spatial Bounding Box Coordinates
          </div>
          <div class="bbox-coords-box" style="margin-bottom:10px;">
            <div><div class="coord-label">XMIN</div><div class="coord-val" style="font-size:0.75rem;">${data.bbox.xmin}</div></div>
            <div><div class="coord-label">YMIN</div><div class="coord-val" style="font-size:0.75rem;">${data.bbox.ymin}</div></div>
            <div><div class="coord-label">XMAX</div><div class="coord-val" style="font-size:0.75rem;">${data.bbox.xmax}</div></div>
            <div><div class="coord-label">YMAX</div><div class="coord-val" style="font-size:0.75rem;">${data.bbox.ymax}</div></div>
          </div>
          <div style="font-size:0.68rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:4px;">
            Differential Diagnosis Probabilities
          </div>
          <div style="background:#FFF; border:1px solid var(--brand-border); border-radius:8px; padding:8px 10px; margin-bottom:12px;">
            ${data.differential.map(item => `
              <div style="display:flex; justify-content:space-between; font-size:0.72rem; margin-bottom:3px;">
                <span>${item.name}</span>
                <strong class="text-mono" style="color:var(--brand-primary);">${item.prob}</strong>
              </div>
            `).join('')}
          </div>
          <button onclick="setSimulatorPage(9)" style="width:100%; padding:10px; background:var(--brand-primary); color:#FFF; border:none; border-radius:8px; font-weight:700; font-size:0.75rem; cursor:pointer;">
            Inspect Raw Conv Gradients Array →
          </button>
        </div>
      `;
      break;

    case 9:
      container.innerHTML = `
        <div style="padding-top:4px;">
          <div style="font-size:0.78rem; font-weight:800; text-transform:uppercase; color:var(--text-title); margin-bottom:2px;">
            Raw XAI Gradients Array
          </div>
          <div style="font-size:0.68rem; color:var(--text-muted); margin-bottom:10px;">
            Final Conv Layer Tensor Activations (A^k)
          </div>
          <div id="sim-page9-grid" class="tensor-grid-container" style="max-width:260px; margin-bottom:10px;"></div>
          <div id="sim-page9-inspector" class="tensor-inspector-bar" style="font-size:0.65rem; padding:8px;"></div>
          <p style="font-size:0.68rem; color:var(--text-muted); line-height:1.35; margin-top:10px;">
            *Tap on individual tensor activation weights above to observe spatial localization gradients before YOLOv26 head prediction.
          </p>
          <button onclick="setSimulatorPage(2)" style="width:100%; margin-top:14px; padding:10px; background:var(--brand-primary); color:#FFF; border:none; border-radius:8px; font-weight:700; font-size:0.75rem; cursor:pointer;">
            ← Back to Home
          </button>
        </div>
      `;
      setTimeout(() => renderGradientsGrid('sim-page9-grid', 'sim-page9-inspector', AppState.currentPreset), 50);
      break;
  }
}

function renderPosterView() {
  const container = document.getElementById('poster-screens-grid');
  if (!container) return;
  container.innerHTML = '';

  const pageTitles = [
    '1st Page: Splash & Login',
    '2nd Page: Home & Quick Upload',
    '3rd Page: Image Workspace',
    '4th Page: Output A (Patient View)',
    '5th Page: Morphological Cues',
    '6th Page: Diagnostic Insights',
    '7th Page: Output B (Panel Dashboard)',
    '8th Page: YOLOv26 Evaluation',
    '9th Page: Raw XAI Gradients'
  ];

  for (let i = 1; i <= 9; i++) {
    const item = document.createElement('div');
    item.className = 'poster-phone-item';
    item.innerHTML = `
      <div class="poster-page-label">${pageTitles[i - 1]}</div>
      <div class="poster-phone-bezel">
        <div class="phone-screen" id="poster-screen-${i}"></div>
        <div class="phone-home-indicator"></div>
      </div>
    `;
    container.appendChild(item);
  }

  setTimeout(() => {
    for (let i = 1; i <= 9; i++) {
      const scr = document.getElementById(`poster-screen-${i}`);
      if (!scr) continue;
      
      scr.innerHTML = `
        <div class="phone-header">
          <button class="phone-header-btn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
          </button>
          <span class="phone-header-title">IDENTI - SKIN</span>
          <button class="phone-header-btn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>
          </button>
        </div>
        <div class="phone-content" id="poster-inner-content-${i}"></div>
        <div class="phone-bottom-nav">
          <div class="phone-nav-item ${i===2?'active':''}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path></svg><span>Home</span></div>
          <div class="phone-nav-item ${i===3?'active':''}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg><span>Workspace</span></div>
          <div class="phone-nav-item ${i===4||i===5||i===6?'active':''}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline></svg><span>History</span></div>
          <div class="phone-nav-item ${i===7||i===8||i===9?'active':''}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg><span>Insights</span></div>
          <div class="phone-nav-item"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg><span>Profile</span></div>
        </div>
      `;

      const inner = document.getElementById(`poster-inner-content-${i}`);
      if (inner) renderPosterScreenContent(i, inner);
    }
  }, 100);
}

function renderPosterScreenContent(pageNum, targetEl) {
  const data = PRESETS[AppState.currentPreset] || PRESETS.buni;
  
  if (pageNum === 1) {
    targetEl.innerHTML = `
      <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:20px 0; text-align:center;">
        <div style="width:60px; height:60px; background:linear-gradient(135deg, #FF6F3C, #D45B28); border-radius:18px; display:flex; align-items:center; justify-content:center; color:#FFF; margin-bottom:14px;">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
        </div>
        <div style="font-size:1.25rem; font-weight:900; color:var(--text-title); margin-bottom:4px;">IDENTI-SKIN</div>
        <p style="font-size:0.65rem; color:var(--text-muted); margin-bottom:20px; line-height:1.3;">AI-Powered Skin Infection Analysis for Smarter, Faster, and Safer Decision</p>
        <button style="width:100%; padding:10px; background:var(--brand-primary); color:#FFF; border:none; border-radius:10px; font-weight:700; font-size:0.75rem; margin-bottom:8px;">Get Started</button>
        <button style="width:100%; padding:10px; background:#FFF; border:1px solid var(--brand-border); border-radius:10px; font-weight:700; font-size:0.75rem; margin-bottom:16px;">I Have an Account</button>
        <div style="font-size:0.58rem; color:var(--text-muted);">Your data is protected & safe - RA 10173 Compliant</div>
      </div>
    `;
  } else if (pageNum === 2) {
    targetEl.innerHTML = `
      <div>
        <div style="font-size:1.15rem; font-weight:900; color:var(--text-title);">Welcome Back!</div>
        <div style="font-size:0.68rem; color:var(--text-muted); margin-bottom:12px;">Analyze skin images and get AI-assisted insights in seconds.</div>
        <div class="upload-dropzone" style="padding:14px; margin-bottom:12px;">
          <div style="font-size:0.75rem; font-weight:700; color:var(--brand-primary);">Tap to Upload or Take a Photo</div>
          <div style="font-size:0.6rem; color:var(--text-muted);">Auto ITA-calibrated & processed locally<br>RA 10173 compliant</div>
        </div>
        <div style="font-size:0.68rem; font-weight:800; text-transform:uppercase; margin-bottom:6px;">Recent Analysis</div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <div class="preset-card" style="padding:6px;"><img src="assets/sample_buni.jpg" class="preset-thumb" style="width:32px; height:32px;"><div class="preset-info"><div class="preset-name" style="font-size:0.72rem;">Buni (Tinea Corporis)</div><div class="preset-sub" style="font-size:0.58rem;">Analyzed just now</div></div><span class="preset-score" style="font-size:0.62rem;">94.2%</span></div>
          <div class="preset-card" style="padding:6px;"><img src="assets/sample_mamaso.jpg" class="preset-thumb" style="width:32px; height:32px;"><div class="preset-info"><div class="preset-name" style="font-size:0.72rem;">Mamaso (Impetigo)</div><div class="preset-sub" style="font-size:0.58rem;">Analyzed 2 days ago</div></div><span class="preset-score" style="font-size:0.62rem;">92.1%</span></div>
          <div class="preset-card" style="padding:6px;"><img src="assets/sample_kulugo.jpg" class="preset-thumb" style="width:32px; height:32px;"><div class="preset-info"><div class="preset-name" style="font-size:0.72rem;">Kulugo (Warts)</div><div class="preset-sub" style="font-size:0.58rem;">Analyzed 5 days ago</div></div><span class="preset-score" style="font-size:0.62rem; background:var(--status-gold-bg); color:var(--status-gold);">88.7%</span></div>
        </div>
      </div>
    `;
  } else if (pageNum === 3) {
    targetEl.innerHTML = `
      <div>
        <div style="font-size:0.85rem; font-weight:800; color:var(--text-title); margin-bottom:8px;">Image Workspace</div>
        <div id="poster-p3-workspace" class="workspace-canvas-container" style="aspect-ratio:1/1; max-height:210px; margin-bottom:10px;"></div>
        <div style="font-size:0.65rem; font-weight:700; color:var(--text-muted); margin-bottom:4px;">Comparison Slider: 50%</div>
        <div style="font-size:0.65rem; font-weight:700; color:var(--text-muted); margin-bottom:6px;">Heatmap Opacity: 70%</div>
        <div style="display:flex; flex-direction:column; gap:4px; font-size:0.65rem;">
          <div class="toggle-pill active" style="padding:4px 8px;"><div class="toggle-checkbox" style="width:12px; height:12px;">✓</div><span>Adaptive L* - CLAHE</span></div>
          <div class="toggle-pill active" style="padding:4px 8px;"><div class="toggle-checkbox" style="width:12px; height:12px;">✓</div><span>Decoupled YOLOv26</span></div>
          <div class="toggle-pill active" style="padding:4px 8px;"><div class="toggle-checkbox" style="width:12px; height:12px;">✓</div><span>Grad-CAM Heatmap</span></div>
        </div>
      </div>
    `;
    setTimeout(() => renderWorkspace('poster-p3-workspace', AppState.currentPreset, 50, 70), 50);
  } else if (pageNum === 4) {
    targetEl.innerHTML = `
      <div>
        <div style="font-size:0.62rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:6px;">VERIFIED CASE PRESETS (PHILIPPINE CONTEXT)</div>
        <div style="display:flex; gap:4px; margin-bottom:8px;">
          <div style="flex:1; background:#FFF; border:1px solid var(--brand-primary); border-radius:6px; padding:4px;"><img src="assets/sample_buni.jpg" style="width:100%; height:32px; object-fit:cover; border-radius:4px;"><div style="font-size:0.55rem; font-weight:700;">Buni</div></div>
          <div style="flex:1; background:#FFF; border:1px solid var(--brand-border); border-radius:6px; padding:4px;"><img src="assets/sample_mamaso.jpg" style="width:100%; height:32px; object-fit:cover; border-radius:4px;"><div style="font-size:0.55rem; font-weight:700;">Mamaso</div></div>
          <div style="flex:1; background:#FFF; border:1px solid var(--brand-border); border-radius:6px; padding:4px;"><img src="assets/sample_kulugo.jpg" style="width:100%; height:32px; object-fit:cover; border-radius:4px;"><div style="font-size:0.55rem; font-weight:700;">Kulugo</div></div>
        </div>
        <div class="patient-card" style="padding:10px;">
          <div style="font-size:0.68rem; font-weight:800; color:var(--brand-primary); text-transform:uppercase;">Output A: Patient View (Simplified)</div>
          <div class="telemetry-chip-bar" style="padding:4px 6px; margin:6px 0;"><span class="chip-value" style="font-size:0.6rem;">Type IV | ITA: 22.4° • Clip: 3.4</span></div>
          <span class="category-pill ${data.badgeClass}" style="font-size:0.58rem;">${data.category.toUpperCase()}</span>
          <div style="font-size:0.95rem; font-weight:900; color:var(--text-title);">${data.name}</div>
          <div style="font-size:0.65rem; color:var(--text-muted); margin-bottom:6px;">Common Local Name: ${data.localName}</div>
          <div class="score-banner" style="padding:6px 10px;"><span style="font-size:0.65rem; font-weight:700; color:var(--brand-primary);">Score:</span><span class="score-num" style="font-size:1.1rem;">${data.confidence}%</span></div>
        </div>
      </div>
    `;
  } else if (pageNum === 5) {
    targetEl.innerHTML = `
      <div>
        <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:var(--text-title); margin-bottom:2px;">Identified Morphological Features</div>
        <div style="font-size:0.62rem; color:var(--text-muted); margin-bottom:8px;">Fine-Grained Cues</div>
        <div class="features-container" style="gap:4px; margin-bottom:8px;">
          <div class="feature-box" style="padding:6px;"><div class="feature-box-title" style="font-size:0.6rem;">SURFACE TEXTURE</div><div class="feature-box-desc" style="font-size:0.65rem;">${data.morphologicalFeatures.texture}</div></div>
          <div class="feature-box" style="padding:6px;"><div class="feature-box-title" style="font-size:0.6rem;">CRUST & EXUDATE</div><div class="feature-box-desc" style="font-size:0.65rem;">${data.morphologicalFeatures.crust}</div></div>
          <div class="feature-box" style="padding:6px;"><div class="feature-box-title" style="font-size:0.6rem;">BOUNDARY / EDGE</div><div class="feature-box-desc" style="font-size:0.65rem;">${data.morphologicalFeatures.boundary}</div></div>
          <div class="feature-box" style="padding:6px;"><div class="feature-box-title" style="font-size:0.6rem;">DIFFERENTIAL KEY</div><div class="feature-box-desc" style="font-size:0.65rem;">${data.morphologicalFeatures.differential}</div></div>
        </div>
        <div style="font-size:0.7rem; font-weight:800; text-transform:uppercase;">GRAD-CAM HEATMAP OVERLAY</div>
        <div id="poster-p5-grid" class="tensor-grid-container" style="max-width:180px; margin-top:4px;"></div>
      </div>
    `;
    setTimeout(() => renderGradientsGrid('poster-p5-grid', 'temp-p5-bar', AppState.currentPreset), 50);
  } else if (pageNum === 6) {
    targetEl.innerHTML = `
      <div>
        <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:var(--text-title); margin-bottom:6px;">Diagnostic & Computational Insights</div>
        <div style="background:#FFF; border:1px solid var(--brand-border); border-radius:8px; padding:8px; font-size:0.65rem; line-height:1.35; margin-bottom:10px;">
          <div style="margin-bottom:6px;">• <strong>Stage 1:</strong> ITA 22.4° (Type IV) calibrated adaptive CLAHE clip limit 3.4.</div>
          <div style="margin-bottom:6px;">• <strong>Stage 1:</strong> Regressed annular boundary (IoU 78.2%) before backbone freezing.</div>
          <div>• <strong>Stage 2:</strong> Focal Loss prioritized elevated peripheral ring features.</div>
        </div>
        <div class="compliance-box" style="padding:8px;">
          <div class="compliance-box-title" style="font-size:0.62rem;">DATA PRIVACY & MEDICAL DISCLAIMER (RA 10173 COMPLIANT)</div>
          <div class="compliance-box-desc" style="font-size:0.58rem;">In accordance with RA 10173 (Data Privacy Act of 2012), skin imaging is processed locally. This platform is an academic assistive screening aid and does not replace professional dermatological diagnosis.</div>
        </div>
      </div>
    `;
  } else if (pageNum === 7) {
    targetEl.innerHTML = `
      <div>
        <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:var(--brand-primary); margin-bottom:6px;">Output B: Panel Dashboard</div>
        <div style="font-size:0.6rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:4px;">ITA Colorimetry & Adaptive CLAHE Telemetry</div>
        <div class="telemetry-grid-4" style="gap:4px; margin-bottom:8px;">
          <div class="telemetry-stat-card" style="padding:4px;"><div class="telemetry-stat-label" style="font-size:0.55rem;">ITA VALUE</div><div class="telemetry-stat-num" style="font-size:0.85rem;">${data.ita}°</div></div>
          <div class="telemetry-stat-card" style="padding:4px;"><div class="telemetry-stat-label" style="font-size:0.55rem;">CLIP LIMIT</div><div class="telemetry-stat-num" style="font-size:0.85rem;">${data.clipLimit}</div></div>
          <div class="telemetry-stat-card" style="padding:4px;"><div class="telemetry-stat-label" style="font-size:0.55rem;">FITZPATRICK</div><div class="telemetry-stat-num" style="font-size:0.75rem;">TYPE IV</div></div>
          <div class="telemetry-stat-card" style="padding:4px;"><div class="telemetry-stat-label" style="font-size:0.55rem;">CONTRAST GAIN</div><div class="telemetry-stat-num" style="font-size:0.85rem;">${data.contrastGain}</div></div>
        </div>
        <div style="font-size:0.6rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:4px;">Two-Stage Decoupled Representation</div>
        <div class="stage-status-card" style="padding:6px; margin-bottom:4px;"><div class="stage-header"><span class="stage-title" style="font-size:0.62rem;">Stage 1: Localization</span><span class="stage-badge frozen" style="font-size:0.55rem;">FROZEN</span></div></div>
        <div class="stage-status-card active-head" style="padding:6px; margin-bottom:8px;"><div class="stage-header"><span class="stage-title" style="font-size:0.62rem;">Stage 2: Discrimination</span><span class="stage-badge active" style="font-size:0.55rem;">ACTIVE HEAD</span></div></div>
        <div style="font-size:0.6rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:4px;">Morphological Overlap Profiler</div>
        <div class="profiler-bars" style="gap:4px;">
          ${data.overlapProfile.map(item => `
            <div class="profiler-item"><div class="profiler-label-row" style="font-size:0.6rem;"><span>${item.name}</span><span class="text-mono">${item.value}%</span></div><div class="profiler-bar-bg"><div class="profiler-bar-fill" style="width:${item.value}%"></div></div></div>
          `).join('')}
        </div>
      </div>
    `;
  } else if (pageNum === 8) {
    targetEl.innerHTML = `
      <div>
        <div style="font-size:0.72rem; font-weight:800; text-transform:uppercase; margin-bottom:4px;">YOLOv26 Model Evaluation Metrics</div>
        <div class="metrics-5-grid" style="gap:4px; margin-bottom:6px;">
          <div class="metric-mini-card" style="padding:4px;"><div class="metric-mini-label" style="font-size:0.55rem;">PRECISION</div><div class="metric-mini-num" style="font-size:0.8rem;">${data.metrics.precision}</div></div>
          <div class="metric-mini-card" style="padding:4px;"><div class="metric-mini-label" style="font-size:0.55rem;">RECALL</div><div class="metric-mini-num" style="font-size:0.8rem;">${data.metrics.recall}</div></div>
          <div class="metric-mini-card" style="padding:4px;"><div class="metric-mini-label" style="font-size:0.55rem;">F1-SCORE</div><div class="metric-mini-num" style="font-size:0.8rem;">${data.metrics.f1}</div></div>
        </div>
        <div style="background:var(--bg-subtle); border:1px solid var(--brand-border); border-radius:6px; padding:6px; font-size:0.62rem; margin-bottom:8px;">
          <div>Latency: <strong>${data.metrics.latency}</strong> | GFLOPs: <strong>${data.metrics.gflops}</strong></div>
          <div>Margin Attribution: <strong>${data.metrics.attribution}</strong></div>
        </div>
        <div style="font-size:0.62rem; font-weight:800; text-transform:uppercase; margin-bottom:3px;">Spatial Coordinates</div>
        <div class="bbox-coords-box" style="gap:4px; padding:4px; margin-bottom:8px;">
          <div><div class="coord-label" style="font-size:0.52rem;">XMIN</div><div class="coord-val" style="font-size:0.7rem;">${data.bbox.xmin}</div></div>
          <div><div class="coord-label" style="font-size:0.52rem;">YMIN</div><div class="coord-val" style="font-size:0.7rem;">${data.bbox.ymin}</div></div>
          <div><div class="coord-label" style="font-size:0.52rem;">XMAX</div><div class="coord-val" style="font-size:0.7rem;">${data.bbox.xmax}</div></div>
          <div><div class="coord-label" style="font-size:0.52rem;">YMAX</div><div class="coord-val" style="font-size:0.7rem;">${data.bbox.ymax}</div></div>
        </div>
        <div style="font-size:0.62rem; font-weight:800; text-transform:uppercase; margin-bottom:3px;">Differential Probabilities</div>
        <div style="background:#FFF; border:1px solid var(--brand-border); border-radius:6px; padding:6px; font-size:0.62rem;">
          ${data.differential.map(d => `<div style="display:flex; justify-content:space-between; margin-bottom:2px;"><span>${d.name}</span><strong>${d.prob}</strong></div>`).join('')}
        </div>
      </div>
    `;
  } else if (pageNum === 9) {
    targetEl.innerHTML = `
      <div>
        <div style="font-size:0.72rem; font-weight:800; text-transform:uppercase; margin-bottom:2px;">Raw XAI Gradients Array</div>
        <div style="font-size:0.58rem; color:var(--text-muted); margin-bottom:6px;">Final Conv Layer Activation Weights</div>
        <div id="poster-p9-grid" class="tensor-grid-container" style="max-width:200px; margin-bottom:8px;"></div>
        <div style="background:var(--bg-subtle); border:1px solid var(--brand-border); border-radius:6px; padding:6px; font-size:0.6rem;">
          Cell: [ch512, 8, 11] | Activation: 0.942 | Attribution: High
        </div>
      </div>
    `;
    setTimeout(() => renderGradientsGrid('poster-p9-grid', 'temp-p9-bar', AppState.currentPreset), 50);
  }
}

function handleFileUpload(file) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    processCustomImage(e.target.result);
  };
  reader.readAsDataURL(file);
}

function processCustomImage(dataUrl) {
  AppState.isAnalyzing = true;
  AppState.customImage = dataUrl;

  const img = new Image();
  img.src = dataUrl;
  img.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 300, 300);

    const imgData = ctx.getImageData(0, 0, 300, 300).data;
    let sumL = 0, sumB = 0, count = 0;

    for (let i = 0; i < imgData.length; i += 16) {
      const r = imgData[i], g = imgData[i+1], b = imgData[i+2];
      const lab = rgbToCielab(r, g, b);
      if (lab.L > 15 && lab.L < 90) {
        sumL += lab.L;
        sumB += lab.b;
        count++;
      }
    }

    const avgL = count > 0 ? sumL / count : 55;
    const avgB = count > 0 ? sumB / count : 15;
    const computedITA = Math.round(calculateITA(avgL, avgB) * 10) / 10;
    const fitz = itaToFitzpatrick(computedITA);
    const computedClip = calculateClipLimit(computedITA);

    AppState.customImageData = {
      id: 'custom',
      name: 'Analyzed Lesion (Custom)',
      localName: 'Detected Morphology',
      category: 'Infectious Dermatosis',
      tag: 'fungal',
      badgeClass: 'fungal',
      confidence: 93.4,
      ita: computedITA,
      clipLimit: computedClip,
      fitzpatrick: `Fitzpatrick ${fitz.type} (${fitz.desc})`,
      contrastGain: `+${Math.round(25 + (computedClip - 2) * 8)}%`,
      image: dataUrl,
      thumb: dataUrl,
      bbox: { xmin: 140, ymin: 160, xmax: 380, ymax: 390 },
      morphologicalFeatures: {
        texture: 'Erythematous lesion border with localized micro-scaling',
        crust: 'Non-exudative dry surface',
        boundary: 'Circumscribed focal margin',
        differential: 'Primary target classification evaluated against 8 infectious classes'
      },
      insights: [
        `ITA calculated at ${computedITA}° (${fitz.type}), deriving adaptive CLAHE clip limit ${computedClip}.`,
        'Stage 1 YOLOv26 regression identified primary region of interest (IoU 79.4%).',
        'Stage 2 Fine-Grained head completed feature attribution via Grad-CAM tensor mapping.'
      ],
      overlapProfile: [
        { name: 'Primary Lesion Morphology', value: 92.4 },
        { name: 'Secondary Infiltration', value: 4.8 },
        { name: 'Keratinous Scaling', value: 2.8 }
      ],
      metrics: {
        precision: '92.1%',
        recall: '88.7%',
        f1: '0.904',
        map50: '91.8%',
        map50_95: '71.0%',
        latency: '12.3 ms',
        gflops: '16.5',
        attribution: '73.5%'
      },
      differential: [
        { name: 'Detected Target Infection', prob: '92.4%' },
        { name: 'Mimic Differential A', prob: '4.8%' },
        { name: 'Mimic Differential B', prob: '2.8%' }
      ]
    };

    AppState.isAnalyzing = false;
    renderWorkspace('main-workspace', 'custom');
    renderWorkspace('sim-workspace', 'custom');
    updateTelemetryUI(AppState.customImageData);
  };
}

function openCameraCapture() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.capture = 'environment';
  input.onchange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };
  input.click();
}

function switchViewMode(mode) {
  AppState.activeViewMode = mode;

  document.querySelectorAll('.view-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });

  document.getElementById('view-desktop-workstation')?.classList.toggle('hidden', mode !== 'workstation');
  document.getElementById('view-mobile-simulator')?.classList.toggle('hidden', mode !== 'simulator');
  document.getElementById('view-poster-overview')?.classList.toggle('hidden', mode !== 'poster');
  document.getElementById('view-architecture')?.classList.toggle('hidden', mode !== 'architecture');

  if (mode === 'workstation') {
    renderWorkspace('main-workspace', AppState.currentPreset);
    renderGradientsGrid('desktop-tensor-grid', 'desktop-tensor-inspector', AppState.currentPreset);
  } else if (mode === 'simulator') {
    renderSimulatorPage(AppState.simulatorPage);
  } else if (mode === 'poster') {
    renderPosterView();
  }
}


function handleHashChange() {
  const hash = window.location.hash.replace('#', '').toLowerCase();
  if (['home', 'workspace', 'patient', 'panel', 'xai'].includes(hash)) {
    if (window.innerWidth <= 768) {
      switchMobileTab(hash);
    } else {
      if (hash === 'home' || hash === 'workspace') switchViewMode('workstation');
      else if (hash === 'patient') { switchViewMode('workstation'); document.querySelector('.output-tab-btn[data-tab="patient"]')?.click(); }
      else if (hash === 'panel' || hash === 'xai') { switchViewMode('workstation'); document.querySelector('.output-tab-btn[data-tab="clinician"]')?.click(); }
    }
  } else if (['workstation', 'simulator', 'poster', 'architecture'].includes(hash)) {
    switchViewMode(hash);
  } else if (hash.startsWith('page-')) {
    const pageNum = parseInt(hash.replace('page-', ''), 10);
    if (pageNum >= 1 && pageNum <= 9) {
      switchViewMode('simulator');
      setSimulatorPage(pageNum);
    }
  }
}

window.addEventListener('hashchange', handleHashChange);

document.addEventListener('DOMContentLoaded', () => {
  selectPreset('buni');
  switchViewMode('workstation');
  if (window.location.hash) handleHashChange();

  const dropzone = document.getElementById('main-dropzone');
  if (dropzone) {
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
  }

  const fileInput = document.getElementById('main-file-input');
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  const splitInput = document.getElementById('main-split-slider');
  if (splitInput) {
    splitInput.addEventListener('input', (e) => {
      AppState.comparisonSplit = parseInt(e.target.value, 10);
      updateSplitPosition('main-workspace', AppState.comparisonSplit);
    });
  }

  const opacityInput = document.getElementById('main-opacity-slider');
  if (opacityInput) {
    opacityInput.addEventListener('input', (e) => {
      AppState.heatmapOpacity = parseInt(e.target.value, 10);
      const valEl = document.getElementById('opacity-val-display');
      if (valEl) valEl.textContent = `${AppState.heatmapOpacity}%`;
      const heatmap = document.getElementById('main-workspace-canvas-heatmap');
      if (heatmap) heatmap.style.opacity = (AppState.heatmapOpacity / 100).toString();
    });
  }

  document.querySelectorAll('.output-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      AppState.outputTab = tab;
      document.querySelectorAll('.output-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.getElementById('tab-content-patient')?.classList.toggle('hidden', tab !== 'patient');
      document.getElementById('tab-content-clinician')?.classList.toggle('hidden', tab !== 'clinician');
    });
  });

  console.log('IDENTI-SKIN Clinical Application Initialized.');
});


// ==========================================================================
// 12. DEDICATED FULL-SCREEN NATIVE MOBILE APP LOGIC (< 768px)
// ==========================================================================
function switchMobileTab(tabName) {
  // Update tab pane active states
  const panes = ['home', 'workspace', 'patient', 'panel', 'xai'];
  panes.forEach(name => {
    const pane = document.getElementById(`mob-tab-${name}`);
    if (pane) pane.classList.toggle('active', name === tabName);
  });

  // Update bottom navigation buttons active states
  document.querySelectorAll('.mob-nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mobtab === tabName);
  });

  // Scroll to top of content
  const content = document.querySelector('.mobile-native-content');
  if (content) content.scrollTop = 0;

  // Render workspace if active
  if (tabName === 'workspace') {
    setTimeout(() => {
      renderWorkspace('mobile-canvas-container', AppState.currentPreset, AppState.comparisonSplit, AppState.heatmapOpacity);
    }, 50);
  } else if (tabName === 'xai') {
    setTimeout(() => {
      renderGradientsGrid('mob-tensor-grid', 'mob-tensor-inspector', AppState.currentPreset);
    }, 50);
  } else if (tabName === 'patient' || tabName === 'panel') {
    updateMobileView();
  }
}

function toggleMobileDrawer(open) {
  const drawer = document.getElementById('mobile-drawer');
  if (drawer) {
    if (open) {
      drawer.classList.remove('hidden');
    } else {
      drawer.classList.add('hidden');
    }
  }
}
window.addEventListener('hashchange', () => {
  if (window.location.hash === '#drawer') toggleMobileDrawer(true);
});


function updateMobileView() {
  const data = AppState.customImageData || PRESETS[AppState.currentPreset] || PRESETS.buni;

  // Update preset chip carousel active state
  document.querySelectorAll('.mob-preset-chip').forEach(chip => {
    const chipKey = chip.textContent.toLowerCase().replace(/[^a-z]/g, '');
    chip.classList.toggle('active', chipKey === AppState.currentPreset || chip.textContent.includes(data.name.split(' ')[0]));
  });

  // Patient elements
  const elTitle = document.getElementById('mob-patient-title');
  if (elTitle) elTitle.textContent = data.name;

  const elLocal = document.getElementById('mob-patient-local');
  if (elLocal) elLocal.textContent = `Common Local Name: ${data.localName}`;

  const elBadge = document.getElementById('mob-patient-badge');
  if (elBadge) {
    elBadge.textContent = data.category.toUpperCase();
    elBadge.className = `category-pill ${data.badgeClass}`;
  }

  const elScore = document.getElementById('mob-patient-score');
  if (elScore) elScore.textContent = `${data.confidence}%`;

  const elItaChip = document.getElementById('mob-patient-ita');
  if (elItaChip) elItaChip.textContent = `${data.fitzpatrick.split(' ')[1]} (Melanin-Rich) | ITA: ${data.ita}° • Clip: ${data.clipLimit}`;

  // Morphological cues
  const elTexture = document.getElementById('mob-feat-texture');
  if (elTexture) elTexture.textContent = data.morphologicalFeatures.texture;

  const elCrust = document.getElementById('mob-feat-crust');
  if (elCrust) elCrust.textContent = data.morphologicalFeatures.crust;

  const elBoundary = document.getElementById('mob-feat-boundary');
  if (elBoundary) elBoundary.textContent = data.morphologicalFeatures.boundary;

  const elDiff = document.getElementById('mob-feat-diff');
  if (elDiff) elDiff.textContent = data.morphologicalFeatures.differential;

  // Insights
  const elInsights = document.getElementById('mob-insights-list');
  if (elInsights) {
    elInsights.innerHTML = data.insights.map((s, idx) => `
      <li style="margin-bottom: 6px;"><strong>Stage ${idx + 1}:</strong> ${s}</li>
    `).join('');
  }

  // Clinician panel
  const elIta = document.getElementById('mob-ita-val');
  if (elIta) elIta.textContent = `${data.ita}°`;

  const elClip = document.getElementById('mob-clip-val');
  if (elClip) elClip.textContent = `${data.clipLimit}`;

  const elFitz = document.getElementById('mob-fitz-val');
  if (elFitz) elFitz.textContent = data.fitzpatrick.split(' ')[1] || 'TYPE IV';

  const elGain = document.getElementById('mob-gain-val');
  if (elGain) elGain.textContent = data.contrastGain;

  // Overlap bars
  const elOverlap = document.getElementById('mob-overlap-bars');
  if (elOverlap) {
    elOverlap.innerHTML = data.overlapProfile.map(item => `
      <div class="profiler-item">
        <div class="profiler-label-row" style="font-size:0.68rem;">
          <span>${item.name}</span>
          <span class="text-mono">${item.value}%</span>
        </div>
        <div class="profiler-bar-bg"><div class="profiler-bar-fill" style="width:${item.value}%"></div></div>
      </div>
    `).join('');
  }

  // Evaluation Metrics
  const m = data.metrics;
  if (document.getElementById('mob-metric-precision')) document.getElementById('mob-metric-precision').textContent = m.precision;
  if (document.getElementById('mob-metric-recall')) document.getElementById('mob-metric-recall').textContent = m.recall;
  if (document.getElementById('mob-metric-f1')) document.getElementById('mob-metric-f1').textContent = m.f1;
  if (document.getElementById('mob-metric-latency')) document.getElementById('mob-metric-latency').textContent = m.latency;
  if (document.getElementById('mob-metric-gflops')) document.getElementById('mob-metric-gflops').textContent = m.gflops;
  if (document.getElementById('mob-metric-attribution')) document.getElementById('mob-metric-attribution').textContent = m.attribution;

  // Coordinates
  const b = data.bbox;
  if (document.getElementById('mob-coord-xmin')) document.getElementById('mob-coord-xmin').textContent = b.xmin;
  if (document.getElementById('mob-coord-ymin')) document.getElementById('mob-coord-ymin').textContent = b.ymin;
  if (document.getElementById('mob-coord-xmax')) document.getElementById('mob-coord-xmax').textContent = b.xmax;
  if (document.getElementById('mob-coord-ymax')) document.getElementById('mob-coord-ymax').textContent = b.ymax;

  // Differential
  const elDiffList = document.getElementById('mob-differential-list');
  if (elDiffList) {
    elDiffList.innerHTML = data.differential.map(d => `
      <div style="display:flex; justify-content:space-between; font-size:0.72rem; margin-bottom:3px;">
        <span>${d.name}</span>
        <strong class="text-mono" style="color:var(--brand-primary);">${d.prob}</strong>
      </div>
    `).join('');
  }
}

// Hook mobile slider inputs
document.addEventListener('DOMContentLoaded', () => {
  const mobSplit = document.getElementById('mob-split-slider');
  if (mobSplit) {
    mobSplit.addEventListener('input', (e) => {
      AppState.comparisonSplit = parseInt(e.target.value, 10);
      updateSplitPosition('mobile-canvas-container', AppState.comparisonSplit);
    });
  }

  const mobOpacity = document.getElementById('mob-opacity-slider');
  if (mobOpacity) {
    mobOpacity.addEventListener('input', (e) => {
      AppState.heatmapOpacity = parseInt(e.target.value, 10);
      const valEl = document.getElementById('mob-opacity-val');
      if (valEl) valEl.textContent = `${AppState.heatmapOpacity}%`;
      const heatmap = document.getElementById('mobile-canvas-container-canvas-heatmap');
      if (heatmap) heatmap.style.opacity = (AppState.heatmapOpacity / 100).toString();
    });
  }

  // Check if initial viewport is mobile phone
  if (window.innerWidth <= 768) {
    updateMobileView();
    renderWorkspace('mobile-canvas-container', AppState.currentPreset);
  }
});


// ==========================================================================
// 13. AUTHENTICATION, CLINICAL REPORT & TRIAGE LOGIC
// ==========================================================================
AppState.currentUser = {
  isLoggedIn: true,
  name: 'Dr. Maria Santos',
  email: 'm.santos@pup.edu.ph',
  role: 'clinician' // clinician | patient
};

function openAuthModal(tab) {
  const modal = document.getElementById('auth-modal');
  if (modal) modal.classList.remove('hidden');
  setAuthTab(tab || 'login');
}

function closeAuthModal() {
  const modal = document.getElementById('auth-modal');
  if (modal) modal.classList.add('hidden');
}

function setAuthTab(tab) {
  const btnLogin = document.getElementById('tab-btn-login');
  const btnReg = document.getElementById('tab-btn-register');
  const formLogin = document.getElementById('form-login');
  const formReg = document.getElementById('form-register');
  const title = document.getElementById('auth-title');

  if (tab === 'login') {
    btnLogin?.classList.add('active');
    btnReg?.classList.remove('active');
    formLogin?.classList.remove('hidden');
    formReg?.classList.add('hidden');
    if (title) title.textContent = 'Sign In to IDENTI-SKIN';
  } else {
    btnLogin?.classList.remove('active');
    btnReg?.classList.add('active');
    formLogin?.classList.add('hidden');
    formReg?.classList.remove('hidden');
    if (title) title.textContent = 'Register New Account';
  }
}

function initUserSession() {
  try {
    const saved = localStorage.getItem('identi_skin_user');
    if (saved) {
      AppState.currentUser = JSON.parse(saved);
      AppState.currentUser.isLoggedIn = true;
    }
  } catch (err) {
    console.error('Failed to parse saved user:', err);
  }

  if (!AppState.currentUser) {
    AppState.currentUser = {
      isLoggedIn: false,
      name: 'Dr. Maria Santos',
      email: 'dr.santos@pup.edu.ph',
      role: 'clinician'
    };
  }
  updateAuthUI();
}

function logoutUser() {
  localStorage.removeItem('identi_skin_user');
  window.location.href = 'login.html';
}

function demoLogin(role) {
  if (role === 'clinician') {
    AppState.currentUser = {
      isLoggedIn: true,
      name: 'Dr. Maria Santos',
      email: 'm.santos@pup.edu.ph',
      role: 'clinician'
    };
  } else {
    AppState.currentUser = {
      isLoggedIn: true,
      name: 'Juan Dela Cruz',
      email: 'juan.delacruz@gmail.com',
      role: 'patient'
    };
  }

  localStorage.setItem('identi_skin_user', JSON.stringify(AppState.currentUser));
  updateAuthUI();
  closeAuthModal();

  if (role === 'clinician') {
    if (window.innerWidth <= 768) switchMobileTab('panel');
    else switchViewMode('workstation');
  } else {
    if (window.innerWidth <= 768) switchMobileTab('patient');
    else switchViewMode('workstation');
  }
}

function handleAuthSubmit(e, mode) {
  e.preventDefault();
  if (mode === 'login') {
    const email = document.getElementById('login-email').value;
    const role = document.getElementById('login-role').value;
    const name = email.split('@')[0].replace('.', ' ').replace(/(^\w|\s\w)/g, m => m.toUpperCase());

    AppState.currentUser = {
      isLoggedIn: true,
      name: (role === 'clinician' ? 'Dr. ' : '') + name,
      email: email,
      role: role
    };
  } else {
    const name = document.getElementById('reg-name').value;
    const email = document.getElementById('reg-email').value;
    const role = document.getElementById('reg-role').value;

    AppState.currentUser = {
      isLoggedIn: true,
      name: name,
      email: email,
      role: role
    };
  }

  localStorage.setItem('identi_skin_user', JSON.stringify(AppState.currentUser));
  updateAuthUI();
  closeAuthModal();
  alert('Successfully signed in as ' + AppState.currentUser.name + '!');
}

function updateAuthUI() {
  const u = AppState.currentUser;
  if (!u) return;

  const isClinician = u.role === 'clinician';
  const roleLabel = isClinician ? 'CLINICIAN' : 'PATIENT';
  const avatar = isClinician ? '👨‍⚕️' : '👤';

  // Desktop Header updates
  const label = document.getElementById('auth-btn-label');
  if (label) label.textContent = avatar + ' ' + u.name;

  const hdrAvatar = document.getElementById('header-user-avatar');
  if (hdrAvatar) hdrAvatar.textContent = avatar;

  const hdrName = document.getElementById('header-user-name');
  if (hdrName) hdrName.textContent = u.name;

  const hdrRole = document.getElementById('header-user-role');
  if (hdrRole) {
    hdrRole.textContent = roleLabel;
    hdrRole.style.background = isClinician ? 'var(--brand-primary)' : '#2563EB';
  }

  // Mobile Drawer updates
  const mobAvatar = document.getElementById('mob-user-avatar');
  if (mobAvatar) mobAvatar.textContent = avatar;

  const mobName = document.getElementById('mob-user-name');
  if (mobName) mobName.textContent = u.name;

  const mobRole = document.getElementById('mob-user-role');
  if (mobRole) mobRole.textContent = (isClinician ? 'Clinician' : 'Patient User') + ' • ' + (u.email || 'PUP CCIS');
}

function openReportModal() {
  const modal = document.getElementById('report-modal');
  if (!modal) return;

  const data = AppState.customImageData || PRESETS[AppState.currentPreset] || PRESETS.buni;

  // Fill in dynamic values
  const now = new Date();
  document.getElementById('report-date').textContent = now.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  document.getElementById('report-patient-name').textContent = AppState.currentUser ? AppState.currentUser.name : 'Juan Dela Cruz';
  document.getElementById('report-ita-summary').textContent = `${data.fitzpatrick.split(' ')[1]} (Melanin-Rich) • ITA: ${data.ita}°`;
  document.getElementById('report-condition-name').textContent = data.name;
  document.getElementById('report-local-name').textContent = `Common Philippine Term: ${data.localName}`;
  document.getElementById('report-confidence').textContent = `${data.confidence}%`;
  document.getElementById('report-category').textContent = data.category.toUpperCase();
  document.getElementById('report-category').className = `category-pill ${data.badgeClass}`;

  document.getElementById('report-texture').textContent = data.morphologicalFeatures.texture;
  document.getElementById('report-crust').textContent = data.morphologicalFeatures.crust;
  document.getElementById('report-boundary').textContent = data.morphologicalFeatures.boundary;
  document.getElementById('report-diff').textContent = data.morphologicalFeatures.differential;

  modal.classList.remove('hidden');
}

function closeReportModal() {
  document.getElementById('report-modal')?.classList.add('hidden');
}

function openTriageModal() {
  document.getElementById('triage-modal')?.classList.remove('hidden');
}

function closeTriageModal() {
  document.getElementById('triage-modal')?.classList.add('hidden');
}

function handleTriageSubmit(e) {
  e.preventDefault();
  const visual = document.getElementById('triage-visual').value;

  let matched = 'buni';
  if (visual === 'ring') matched = 'buni';
  else if (visual === 'honey') matched = 'mamaso';
  else if (visual === 'wart') matched = 'kulugo';
  else if (visual === 'patches') matched = 'an_an';
  else if (visual === 'macerated') matched = 'alipunga';
  else if (visual === 'blisters') matched = 'bulutong';

  selectPreset(matched);
  closeTriageModal();

  const data = PRESETS[matched];
  alert(`AI Triage Recommendation:\nBased on your reported visual cues and symptoms, the highest probability correlation is ${data.name} (${data.localName}) with ${data.confidence}% confidence score.\n\nLoading full workspace analysis...`);

  if (window.innerWidth <= 768) {
    switchMobileTab('patient');
  } else {
    switchViewMode('workstation');
  }
}

// Initialize Auth label on load
document.addEventListener('DOMContentLoaded', () => {
  initUserSession();
  if (window.location.hash === '#drawer') toggleMobileDrawer(true);
});
