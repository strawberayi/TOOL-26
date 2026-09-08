import re

with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Let's inspect index.html structure
# We want to insert the mobile native container inside .app-container or main-viewport
mobile_native_html = '''
    <!-- ====================================================================
         NATIVE MOBILE FULL-SCREEN APPLICATION (PHONE VIEWPORT < 768px)
         ==================================================================== -->
    <div id="mobile-native-view" class="mobile-app-root">
      <!-- Mobile Native Top Bar -->
      <header class="mobile-native-header">
        <button class="mobile-header-icon-btn" onclick="toggleMobileDrawer(true)" title="Menu">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
        </button>

        <div class="mobile-header-brand" onclick="switchMobileTab('home')">
          <div class="mobile-brand-icon-mini">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFF" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
          </div>
          <span class="mobile-brand-title">IDENTI - SKIN</span>
        </div>

        <button class="mobile-header-icon-btn" onclick="openCameraCapture()" title="Camera">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
        </button>
      </header>

      <!-- Mobile Content Container (100% full screen) -->
      <div class="mobile-native-content">
        <!-- 1. HOME TAB -->
        <div id="mob-tab-home" class="mob-tab-pane active">
          <div class="mob-greeting-box">
            <h1 class="mob-greeting-title">Welcome Back!</h1>
            <p class="mob-greeting-sub">Analyze skin images and get AI-assisted insights in seconds.</p>
          </div>

          <!-- Quick Upload / Camera Dropzone -->
          <div class="upload-dropzone mob-upload-card" onclick="document.getElementById('main-file-input').click()">
            <div class="upload-icon-circle">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
            </div>
            <div class="upload-prompt">Tap to Upload or Take a Photo</div>
            <div class="upload-sub">Auto ITA-calibrated & processed locally<br>RA 10173 compliant</div>
            <div class="upload-btn-group" style="margin-top:10px;">
              <button type="button" class="upload-btn" onclick="event.stopPropagation(); document.getElementById('main-file-input').click()">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                Upload File
              </button>
              <button type="button" class="upload-btn" onclick="event.stopPropagation(); openCameraCapture()">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
                Take Photo
              </button>
            </div>
          </div>

          <!-- Recent Analysis / Philippine Presets Section -->
          <div class="mob-section-header">
            <span class="mob-section-title">Verified Cases (Philippine Context)</span>
            <span class="mob-section-link" onclick="switchMobileTab('workspace')">Open Workspace</span>
          </div>

          <div class="presets-grid mob-presets-list">
            <div class="preset-card active" data-preset="buni" onclick="selectPreset('buni'); switchMobileTab('workspace');">
              <img src="assets/sample_buni.jpg" class="preset-thumb" alt="Buni">
              <div class="preset-info">
                <div class="preset-name">Buni (Tinea Corporis)</div>
                <div class="preset-sub">Fungal • Ringworm • 94.2%</div>
              </div>
              <span class="preset-score">94.2%</span>
            </div>

            <div class="preset-card" data-preset="mamaso" onclick="selectPreset('mamaso'); switchMobileTab('workspace');">
              <img src="assets/sample_mamaso.jpg" class="preset-thumb" alt="Mamaso">
              <div class="preset-info">
                <div class="preset-name">Mamaso (Impetigo)</div>
                <div class="preset-sub">Bacterial • Honey Crusts • 92.1%</div>
              </div>
              <span class="preset-score">92.1%</span>
            </div>

            <div class="preset-card" data-preset="kulugo" onclick="selectPreset('kulugo'); switchMobileTab('workspace');">
              <img src="assets/sample_kulugo.jpg" class="preset-thumb" alt="Kulugo">
              <div class="preset-info">
                <div class="preset-name">Kulugo (Warts)</div>
                <div class="preset-sub">Viral • Verrucous • 88.7%</div>
              </div>
              <span class="preset-score" style="background:var(--status-gold-bg); color:var(--status-gold); border-color:#FDE68A;">88.7%</span>
            </div>

            <div class="preset-card" data-preset="an_an" onclick="selectPreset('an_an'); switchMobileTab('workspace');">
              <img src="assets/sample_an_an.jpg" class="preset-thumb" alt="An-an">
              <div class="preset-info">
                <div class="preset-name">An-an (Tinea Versicolor)</div>
                <div class="preset-sub">Fungal • Hypopigmented • 91.0%</div>
              </div>
              <span class="preset-score">91.0%</span>
            </div>

            <div class="preset-card" data-preset="alipunga" onclick="selectPreset('alipunga'); switchMobileTab('workspace');">
              <img src="assets/sample_alipunga.jpg" class="preset-thumb" alt="Alipunga">
              <div class="preset-info">
                <div class="preset-name">Alipunga (Athlete's Foot)</div>
                <div class="preset-sub">Fungal • Interdigital Scale • 89.4%</div>
              </div>
              <span class="preset-score">89.4%</span>
            </div>

            <div class="preset-card" data-preset="bulutong" onclick="selectPreset('bulutong'); switchMobileTab('workspace');">
              <img src="assets/sample_bulutong.jpg" class="preset-thumb" alt="Bulutong">
              <div class="preset-info">
                <div class="preset-name">Bulutong (Chickenpox)</div>
                <div class="preset-sub">Viral • Umbilicated • 93.5%</div>
              </div>
              <span class="preset-score">93.5%</span>
            </div>
          </div>

          <!-- Compliance Note in Home -->
          <div class="compliance-box" style="margin-top:16px;">
            <div class="compliance-box-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
              <span>DATA PRIVACY ACT OF 2012 (RA 10173)</span>
            </div>
            <p class="compliance-box-desc">
              All skin photography is analyzed locally on this device. No images or biometric health records are stored in remote servers.
            </p>
          </div>
        </div>

        <!-- 2. WORKSPACE TAB -->
        <div id="mob-tab-workspace" class="mob-tab-pane">
          <div class="med-card" style="padding:14px; margin-bottom:12px;">
            <div class="med-card-header">
              <span class="med-card-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
                Image Workspace
              </span>
              <span class="badge-tag green">YOLOv26</span>
            </div>

            <!-- Full width Mobile Canvas Container -->
            <div id="mobile-canvas-container" class="workspace-canvas-container" style="aspect-ratio:1/1; max-height:360px; margin-bottom:14px;"></div>

            <!-- Controls -->
            <div class="workspace-controls" style="gap:10px;">
              <div class="control-row">
                <span class="slider-label" style="font-size:0.75rem;">Comparison Slider</span>
                <div class="slider-container">
                  <input type="range" id="mob-split-slider" min="0" max="100" value="50" class="range-slider">
                </div>
              </div>

              <div class="control-row">
                <span class="slider-label" style="font-size:0.75rem;">Heatmap Opacity</span>
                <div class="slider-container">
                  <input type="range" id="mob-opacity-slider" min="0" max="100" value="70" class="range-slider">
                  <span id="mob-opacity-val" class="slider-val" style="font-size:0.72rem;">70%</span>
                </div>
              </div>

              <div class="toggle-group" style="grid-template-columns: 1fr; gap:6px; margin-top:4px;">
                <div class="toggle-pill active" onclick="this.classList.toggle('active'); AppState.toggles.clahe = !AppState.toggles.clahe; renderWorkspace('mobile-canvas-container', AppState.currentPreset);">
                  <div class="toggle-checkbox"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg></div>
                  <span>Adaptive L* - CLAHE Enhancement</span>
                </div>

                <div class="toggle-pill active" onclick="this.classList.toggle('active'); AppState.toggles.yolo = !AppState.toggles.yolo; renderWorkspace('mobile-canvas-container', AppState.currentPreset);">
                  <div class="toggle-checkbox"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg></div>
                  <span>Decoupled YOLOv26 Bounding Box</span>
                </div>

                <div class="toggle-pill active" onclick="this.classList.toggle('active'); AppState.toggles.heatmap = !AppState.toggles.heatmap; renderWorkspace('mobile-canvas-container', AppState.currentPreset);">
                  <div class="toggle-checkbox"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg></div>
                  <span>Grad - CAM Heatmap Overlay</span>
                </div>
              </div>

              <button type="button" class="view-btn active" style="width:100%; margin-top:8px; padding:12px; justify-content:center; font-size:0.82rem;" onclick="switchMobileTab('patient')">
                View Full Patient Report →
              </button>
            </div>
          </div>
        </div>

        <!-- 3. PATIENT VIEW TAB -->
        <div id="mob-tab-patient" class="mob-tab-pane">
          <!-- Preset Selector Bar -->
          <div class="mob-presets-carousel">
            <button class="mob-preset-chip active" onclick="selectPreset('buni'); updateMobileView();">Buni</button>
            <button class="mob-preset-chip" onclick="selectPreset('mamaso'); updateMobileView();">Mamaso</button>
            <button class="mob-preset-chip" onclick="selectPreset('kulugo'); updateMobileView();">Kulugo</button>
            <button class="mob-preset-chip" onclick="selectPreset('an_an'); updateMobileView();">An-an</button>
            <button class="mob-preset-chip" onclick="selectPreset('alipunga'); updateMobileView();">Alipunga</button>
            <button class="mob-preset-chip" onclick="selectPreset('bulutong'); updateMobileView();">Bulutong</button>
          </div>

          <div class="patient-card" style="padding:14px;">
            <div class="telemetry-chip-bar" style="padding:6px 10px; margin-bottom:10px;">
              <span class="chip-title">Skin-Tone:</span>
              <span id="mob-patient-ita" class="chip-value" style="font-size:0.7rem;">Fitzpatrick Type IV | ITA: 22.4° • Clip: 3.4</span>
            </div>

            <div class="diagnosis-header" style="margin-bottom:8px;">
              <span id="mob-patient-badge" class="category-pill fungal">FUNGAL INFECTION</span>
              <h2 id="mob-patient-title" class="condition-title" style="font-size:1.3rem;">Buni (Tinea Corporis)</h2>
              <div id="mob-patient-local" class="condition-local-name">Common Local Name: Ringworm</div>
            </div>

            <div class="score-banner" style="padding:10px 14px; margin-bottom:12px;">
              <span class="score-title" style="font-size:0.75rem;">Confidence Score:</span>
              <span id="mob-patient-score" class="score-num" style="font-size:1.45rem;">94.2%</span>
            </div>

            <div style="font-size: 0.72rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px;">
              Identified Morphological Features
            </div>

            <div class="features-container" style="gap:6px; margin-bottom:12px;">
              <div class="feature-box" style="padding:8px 10px;">
                <div class="feature-box-title">Surface Texture</div>
                <div id="mob-feat-texture" class="feature-box-desc" style="font-size:0.72rem;">Active peripheral scaling with central clearing</div>
              </div>
              <div class="feature-box" style="padding:8px 10px;">
                <div class="feature-box-title">Crust & Exudate</div>
                <div id="mob-feat-crust" class="feature-box-desc" style="font-size:0.72rem;">Absent / Dry Erythema / Non-exudative</div>
              </div>
              <div class="feature-box" style="padding:8px 10px;">
                <div class="feature-box-title">Boundary / Edge</div>
                <div id="mob-feat-boundary" class="feature-box-desc" style="font-size:0.72rem;">Sharply defined annular elevated ring</div>
              </div>
              <div class="feature-box" style="padding:8px 10px;">
                <div class="feature-box-title">Differential Key</div>
                <div id="mob-feat-diff" class="feature-box-desc" style="font-size:0.72rem;">Distinguished from Impetigo by ring border & absence of honey crust</div>
              </div>
            </div>

            <div style="font-size: 0.72rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px;">
              Diagnostic & Computational Insights
            </div>
            <ul id="mob-insights-list" style="padding-left: 16px; font-size: 0.72rem; color: var(--text-body); margin-bottom: 12px; line-height: 1.4;"></ul>

            <div class="compliance-box" style="margin-top:0;">
              <div class="compliance-box-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                <span>DATA PRIVACY & MEDICAL DISCLAIMER</span>
              </div>
              <p class="compliance-box-desc">
                RA 10173 compliant on-device AI. This application is an academic screening aid and does not substitute for certified clinical dermatological diagnosis.
              </p>
            </div>
          </div>
        </div>

        <!-- 4. CLINICIAN PANEL TAB -->
        <div id="mob-tab-panel" class="mob-tab-pane">
          <div class="med-card" style="padding:14px;">
            <div class="med-card-header">
              <span class="med-card-title">Output B: Panel Dashboard</span>
              <span class="badge-tag green">Telemetry</span>
            </div>

            <div class="telemetry-grid-4" style="gap:6px; margin-bottom:12px;">
              <div class="telemetry-stat-card" style="padding:8px 4px;">
                <div class="telemetry-stat-label">ITA Value</div>
                <div id="mob-ita-val" class="telemetry-stat-num" style="font-size:1.1rem;">24.4°</div>
              </div>
              <div class="telemetry-stat-card" style="padding:8px 4px;">
                <div class="telemetry-stat-label">Clip Limit</div>
                <div id="mob-clip-val" class="telemetry-stat-num" style="font-size:1.1rem;">3.4</div>
              </div>
              <div class="telemetry-stat-card" style="padding:8px 4px;">
                <div class="telemetry-stat-label">Fitzpatrick</div>
                <div id="mob-fitz-val" class="telemetry-stat-num" style="font-size:0.95rem;">TYPE IV</div>
              </div>
              <div class="telemetry-stat-card" style="padding:8px 4px;">
                <div class="telemetry-stat-label">Contrast Gain</div>
                <div id="mob-gain-val" class="telemetry-stat-num" style="font-size:1.1rem;">+35.5%</div>
              </div>
            </div>

            <div style="font-size: 0.68rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px;">
              Decoupled Training Status
            </div>

            <div class="stage-status-card" style="padding:8px 10px; margin-bottom:6px;">
              <div class="stage-header">
                <span class="stage-title" style="font-size:0.7rem;">Stage 1: Localization Learning</span>
                <span class="stage-badge frozen">FROZEN</span>
              </div>
              <p class="stage-desc" style="font-size:0.65rem;">Spatial bounding box & edge geometry extraction. CSP-Darknet locked.</p>
            </div>

            <div class="stage-status-card active-head" style="padding:8px 10px; margin-bottom:12px;">
              <div class="stage-header">
                <span class="stage-title" style="font-size:0.7rem;">Stage 2: Discrimination</span>
                <span class="stage-badge active">ACTIVE HEAD</span>
              </div>
              <p class="stage-desc" style="font-size:0.65rem;">Focal Loss penalized long-tail classes and morphological mimics.</p>
            </div>

            <div style="font-size: 0.68rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px;">
              Morphological Overlap Profiler
            </div>
            <div id="mob-overlap-bars" class="profiler-bars" style="gap:6px; margin-bottom:14px;"></div>

            <div style="font-size: 0.68rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px;">
              YOLOv26 Model Evaluation Metrics
            </div>
            <div class="metrics-5-grid" style="gap:6px; margin-bottom:10px;">
              <div class="metric-mini-card">
                <div class="metric-mini-label">Precision</div>
                <div id="mob-metric-precision" class="metric-mini-num" style="font-size:0.92rem;">92.4%</div>
              </div>
              <div class="metric-mini-card">
                <div class="metric-mini-label">Recall</div>
                <div id="mob-metric-recall" class="metric-mini-num" style="font-size:0.92rem;">89.1%</div>
              </div>
              <div class="metric-mini-card">
                <div class="metric-mini-label">F1-Score</div>
                <div id="mob-metric-f1" class="metric-mini-num" style="font-size:0.92rem;">0.907</div>
              </div>
            </div>

            <div style="display:flex; justify-content:space-between; background:var(--bg-subtle); border:1px solid var(--brand-border); border-radius:8px; padding:8px 10px; margin-bottom:10px; font-size:0.68rem;">
              <div>Latency: <strong id="mob-metric-latency" class="text-mono" style="color:var(--brand-primary);">12.4 ms</strong></div>
              <div>GFLOPs: <strong id="mob-metric-gflops" class="text-mono" style="color:var(--brand-primary);">16.5</strong></div>
              <div>Attribution: <strong id="mob-metric-attribution" class="text-mono" style="color:#15803D;">73.1%</strong></div>
            </div>

            <div style="font-size: 0.68rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px;">
              Spatial Bounding Box Coordinates
            </div>
            <div class="bbox-coords-box" style="margin-bottom:10px;">
              <div><div class="coord-label">XMIN</div><div id="mob-coord-xmin" class="coord-val" style="font-size:0.75rem;">142</div></div>
              <div><div class="coord-label">YMIN</div><div id="mob-coord-ymin" class="coord-val" style="font-size:0.75rem;">215</div></div>
              <div><div class="coord-label">XMAX</div><div id="mob-coord-xmax" class="coord-val" style="font-size:0.75rem;">388</div></div>
              <div><div class="coord-label">YMAX</div><div id="mob-coord-ymax" class="coord-val" style="font-size:0.75rem;">420</div></div>
            </div>

            <div style="font-size: 0.68rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px;">
              Differential Diagnosis Probabilities
            </div>
            <div id="mob-differential-list" style="background:#FFF; border:1px solid var(--brand-border); border-radius:8px; padding:8px 10px;"></div>
          </div>
        </div>

        <!-- 5. XAI TAB -->
        <div id="mob-tab-xai" class="mob-tab-pane">
          <div class="gradients-section" style="padding:14px;">
            <div class="tensor-header">
              <div class="tensor-title">Raw XAI Gradients Array</div>
              <div class="tensor-formula">A_k = 1/Z Σ_i Σ_j (∂y^c / ∂A^k_ij)</div>
            </div>
            <div id="mob-tensor-grid" class="tensor-grid-container" style="max-width:280px;"></div>
            <div id="mob-tensor-inspector" class="tensor-inspector-bar" style="font-size:0.65rem; padding:8px;"></div>
            <p style="font-size:0.68rem; color:var(--text-muted); line-height:1.35; margin-top:10px;">
              Tap any tensor grid cell to inspect spatial feature activations from the final YOLOv26 convolutional layer.
            </p>
          </div>
        </div>
      </div>

      <!-- Mobile Bottom Navigation Bar -->
      <nav class="mobile-native-bottom-nav">
        <button class="mob-nav-btn active" data-mobtab="home" onclick="switchMobileTab('home')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
          <span>Home</span>
        </button>
        <button class="mob-nav-btn" data-mobtab="workspace" onclick="switchMobileTab('workspace')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
          <span>Workspace</span>
        </button>
        <button class="mob-nav-btn" data-mobtab="patient" onclick="switchMobileTab('patient')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline></svg>
          <span>Patient</span>
        </button>
        <button class="mob-nav-btn" data-mobtab="panel" onclick="switchMobileTab('panel')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
          <span>Panel</span>
        </button>
        <button class="mob-nav-btn" data-mobtab="xai" onclick="switchMobileTab('xai')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
          <span>XAI</span>
        </button>
      </nav>

      <!-- Slide-Over Drawer for Mobile -->
      <div id="mobile-drawer" class="mobile-drawer-overlay hidden" onclick="toggleMobileDrawer(false)">
        <div class="mobile-drawer-panel" onclick="event.stopPropagation()">
          <div class="mobile-drawer-header">
            <div style="display:flex; align-items:center; gap:8px;">
              <div class="brand-icon" style="width:30px; height:30px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FFF" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg></div>
              <span style="font-size:0.95rem; font-weight:800;">IDENTI-SKIN Menu</span>
            </div>
            <button class="modal-close-btn" onclick="toggleMobileDrawer(false)">×</button>
          </div>
          <div class="mobile-drawer-body">
            <div style="font-size:0.7rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:8px;">Quick Presets</div>
            <div style="display:flex; flex-direction:column; gap:6px; margin-bottom:16px;">
              <button class="upload-btn" onclick="selectPreset('buni'); toggleMobileDrawer(false); switchMobileTab('patient');">Buni (Tinea Corporis)</button>
              <button class="upload-btn" onclick="selectPreset('mamaso'); toggleMobileDrawer(false); switchMobileTab('patient');">Mamaso (Impetigo)</button>
              <button class="upload-btn" onclick="selectPreset('kulugo'); toggleMobileDrawer(false); switchMobileTab('patient');">Kulugo (Warts)</button>
              <button class="upload-btn" onclick="selectPreset('an_an'); toggleMobileDrawer(false); switchMobileTab('patient');">An-an (Tinea Versicolor)</button>
              <button class="upload-btn" onclick="selectPreset('alipunga'); toggleMobileDrawer(false); switchMobileTab('patient');">Alipunga (Athlete's Foot)</button>
              <button class="upload-btn" onclick="selectPreset('bulutong'); toggleMobileDrawer(false); switchMobileTab('patient');">Bulutong (Chickenpox)</button>
            </div>
            <div style="font-size:0.7rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:8px;">Tools</div>
            <button class="upload-btn" style="margin-bottom:6px; width:100%;" onclick="toggleMobileDrawer(false); openCameraCapture();">📷 Open Camera</button>
            <button class="upload-btn" style="margin-bottom:12px; width:100%;" onclick="toggleMobileDrawer(false); document.getElementById('main-file-input').click();">📁 Browse Photo</button>
            <div style="font-size:0.65rem; color:var(--text-muted); line-height:1.35; padding:10px; background:var(--bg-subtle); border-radius:8px; border:1px solid var(--brand-border);">
              PUP CCIS BSCS Thesis 2026<br>
              Resolving Morphological Overlap in Infectious Skin Diseases<br>
              <strong>RA 10173 Protected</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
'''

# Find the end of </main> and insert before mobile-persistent-nav
insert_marker = '<nav class="mobile-persistent-nav hidden">'
if insert_marker in html:
    html = html.replace(insert_marker, mobile_native_html + '\n    ' + insert_marker)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("Inserted #mobile-native-view into index.html successfully!")
else:
    print("Marker not found!")
