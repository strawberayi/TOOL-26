mobile_js = '''
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
'''

# Check if selectPreset already calls updateMobileView
with open('app.js', 'r', encoding='utf-8') as f:
    code = f.read()

# Make selectPreset call updateMobileView
if 'updateMobileView();' not in code:
    code = code.replace('updateTelemetryUI(data);', 'updateTelemetryUI(data);\n  if (typeof updateMobileView === "function") updateMobileView();')

code = code + '\n' + mobile_js

with open('app.js', 'w', encoding='utf-8') as f:
    f.write(code)

print("Appended mobile JS logic successfully!")
