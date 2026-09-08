import sys

with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Add user profile / login button in global header
header_user_btn = '''
        <!-- Header Actions -->
        <div class="header-actions">
          <div class="compliance-badge" title="Data Privacy Act of 2012 Compliance">
            <span class="badge-dot"></span>
            <span>RA 10173 Protected</span>
          </div>
          <button class="thesis-btn" onclick="openReportModal()" title="Print / Export Clinical Report">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
            Export Report
          </button>
          <button class="thesis-btn" onclick="openTriageModal()" title="Symptoms Triage Questionnaire">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
            Triage Quiz
          </button>
          <button id="auth-header-btn" class="view-btn active" style="padding:6px 12px;" onclick="openAuthModal('login')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            <span id="auth-btn-label">Sign In</span>
          </button>
        </div>'''

# Replace header-actions
old_actions_start = '<div class="header-actions">'
old_actions_end = '</div>\n      </div>\n    </header>'
pos1 = html.find(old_actions_start)
pos2 = html.find(old_actions_end, pos1)
if pos1 != -1 and pos2 != -1:
    html = html[:pos1] + header_user_btn + html[pos2 + len(old_actions_end) - len('</div>\n    </header>'):]
    print("Updated global header actions!")

# 2. Add Modals before </body>:
# - Auth Modal (Login / Sign Up / Role selection / Demo accounts)
# - Printable Clinical Report Modal
# - Symptoms Triage Questionnaire Modal
modals_html = '''
    <!-- ====================================================================
         MODAL 1: AUTHENTICATION (LOGIN & REGISTRATION)
         ==================================================================== -->
    <div id="auth-modal" class="modal-overlay hidden" onclick="closeAuthModal()">
      <div class="modal-card auth-card" onclick="event.stopPropagation()">
        <div class="modal-header">
          <div style="display:flex; align-items:center; gap:10px;">
            <div class="brand-icon" style="width:32px; height:32px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFF" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg></div>
            <div>
              <h3 id="auth-title" style="font-size:1.1rem; font-weight:800; color:var(--text-title);">Sign In to IDENTI-SKIN</h3>
              <div style="font-size:0.7rem; color:var(--text-muted);">PUP CCIS • Skin Infection AI Analysis</div>
            </div>
          </div>
          <button class="modal-close-btn" onclick="closeAuthModal()">×</button>
        </div>

        <!-- Auth Tabs (Login vs Register) -->
        <div class="output-tab-bar" style="margin-bottom:16px;">
          <button id="tab-btn-login" class="output-tab-btn active" onclick="setAuthTab('login')">Sign In</button>
          <button id="tab-btn-register" class="output-tab-btn" onclick="setAuthTab('register')">Register New Account</button>
        </div>

        <!-- Quick 1-Tap Demo Accounts -->
        <div style="background:var(--bg-subtle); border:1px solid var(--brand-border); border-radius:10px; padding:10px 12px; margin-bottom:16px;">
          <div style="font-size:0.68rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:6px;">
            ⚡ Quick 1-Tap Demo Access:
          </div>
          <div style="display:flex; gap:8px;">
            <button type="button" class="upload-btn" style="flex:1; font-size:0.72rem; padding:6px;" onclick="demoLogin('clinician')">
              👨‍⚕️ Clinician / Researcher
            </button>
            <button type="button" class="upload-btn" style="flex:1; font-size:0.72rem; padding:6px;" onclick="demoLogin('patient')">
              👤 Patient User
            </button>
          </div>
        </div>

        <!-- Login Form -->
        <form id="form-login" onsubmit="handleAuthSubmit(event, 'login')">
          <div class="form-group" style="margin-bottom:12px;">
            <label class="form-label">Email or PUP WebMail</label>
            <input type="email" id="login-email" class="form-input" placeholder="e.g. researcher@pup.edu.ph" required>
          </div>

          <div class="form-group" style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <label class="form-label">Password</label>
              <a href="#" style="font-size:0.68rem; color:var(--brand-primary); text-decoration:none;" onclick="alert('Demo Mode: Use password \"password123\" or click the 1-Tap Demo buttons above.')">Forgot Password?</a>
            </div>
            <input type="password" id="login-password" class="form-input" placeholder="••••••••" required>
          </div>

          <div class="form-group" style="margin-bottom:16px;">
            <label class="form-label">Sign In As Role</label>
            <select id="login-role" class="form-input">
              <option value="clinician">Dermatologist / Clinical Researcher (Full Panel & Weights)</option>
              <option value="patient">Patient / Public User (Simplified View & Triage)</option>
            </select>
          </div>

          <button type="submit" class="view-btn active" style="width:100%; padding:12px; justify-content:center; font-size:0.85rem; margin-bottom:12px;">
            Sign In to Account
          </button>
        </form>

        <!-- Register Form (Hidden by default) -->
        <form id="form-register" class="hidden" onsubmit="handleAuthSubmit(event, 'register')">
          <div class="form-group" style="margin-bottom:10px;">
            <label class="form-label">Full Name</label>
            <input type="text" id="reg-name" class="form-input" placeholder="e.g. Dr. Maria Santos" required>
          </div>

          <div class="form-group" style="margin-bottom:10px;">
            <label class="form-label">Email Address</label>
            <input type="email" id="reg-email" class="form-input" placeholder="e.g. maria.santos@hospital.ph" required>
          </div>

          <div class="form-group" style="margin-bottom:10px;">
            <label class="form-label">Password</label>
            <input type="password" id="reg-pass" class="form-input" placeholder="Min. 8 characters" required>
          </div>

          <div class="form-group" style="margin-bottom:12px;">
            <label class="form-label">Account Role</label>
            <select id="reg-role" class="form-input">
              <option value="clinician">Dermatologist / Medical Researcher</option>
              <option value="patient">Patient / Individual</option>
            </select>
          </div>

          <div style="display:flex; align-items:flex-start; gap:8px; margin-bottom:14px; font-size:0.7rem; color:var(--text-body);">
            <input type="checkbox" id="reg-consent" required style="margin-top:2px;">
            <label for="reg-consent">
              I agree to the <strong>RA 10173 (Data Privacy Act of 2012)</strong> policy. My uploaded skin images will be processed locally and securely on-device.
            </label>
          </div>

          <button type="submit" class="view-btn active" style="width:100%; padding:12px; justify-content:center; font-size:0.85rem; margin-bottom:12px;">
            Create New Account
          </button>
        </form>

        <div style="font-size:0.65rem; text-align:center; color:var(--text-muted);">
          Protected under RA 10173 • Polytechnic University of the Philippines
        </div>
      </div>
    </div>

    <!-- ====================================================================
         MODAL 2: CLINICAL REPORT GENERATOR (PRINT / PDF VIEW)
         ==================================================================== -->
    <div id="report-modal" class="modal-overlay hidden" onclick="closeReportModal()">
      <div class="modal-card report-card" onclick="event.stopPropagation()">
        <div class="modal-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
            <div>
              <h3 style="font-size:1.05rem; font-weight:800; color:var(--text-title);">Clinical Diagnostic Consultation Report</h3>
              <div style="font-size:0.7rem; color:var(--text-muted);">PUP CCIS Tele-Dermatology Screening Summary</div>
            </div>
          </div>
          <div style="display:flex; gap:8px;">
            <button class="thesis-btn" onclick="window.print()">
              🖨️ Print / Save PDF
            </button>
            <button class="modal-close-btn" onclick="closeReportModal()">×</button>
          </div>
        </div>

        <div id="report-printable-area" class="report-body">
          <!-- Report Header -->
          <div class="report-doc-header">
            <div style="display:flex; align-items:center; gap:10px;">
              <div class="brand-icon" style="width:36px; height:36px;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFF" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg></div>
              <div>
                <div style="font-size:1.1rem; font-weight:900; color:#2A1A14;">IDENTI-SKIN DIAGNOSTIC SUMMARY</div>
                <div style="font-size:0.68rem; color:#846F65;">ITA-Guided Adaptive L*-CLAHE & YOLOv26 • RA 10173 Verified</div>
              </div>
            </div>
            <div style="text-align:right; font-size:0.72rem; color:#4A3830;">
              <div><strong>Date:</strong> <span id="report-date"></span></div>
              <div><strong>Report ID:</strong> <span class="text-mono" id="report-id">PUP-2026-0842</span></div>
            </div>
          </div>

          <!-- Patient & Skin Tone Telemetry -->
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:14px 0; background:#FFF7F2; padding:12px; border-radius:8px; border:1px solid #F0C4AD;">
            <div>
              <div style="font-size:0.65rem; font-weight:800; text-transform:uppercase; color:#846F65;">Patient Profile</div>
              <div style="font-size:0.85rem; font-weight:800; color:#2A1A14;" id="report-patient-name">Juan Dela Cruz</div>
              <div style="font-size:0.72rem; color:#4A3830;">Evaluated User (Age: 32 • Philippine Context)</div>
            </div>
            <div>
              <div style="font-size:0.65rem; font-weight:800; text-transform:uppercase; color:#846F65;">Skin-Tone Colorimetry (ITA-Core)</div>
              <div style="font-size:0.85rem; font-weight:800; color:#D45B28;" id="report-ita-summary">Type IV (Melanin-Rich) • ITA: 22.4°</div>
              <div style="font-size:0.72rem; color:#4A3830;">Adaptive CLAHE Clip Limit: 3.4 (+35.5% Gain)</div>
            </div>
          </div>

          <!-- Diagnosis Outcome -->
          <div style="border:1.5px solid #D45B28; border-radius:10px; padding:14px; margin-bottom:14px; background:#FFF;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
              <span id="report-category" class="category-pill fungal">FUNGAL INFECTION</span>
              <span style="font-size:1.3rem; font-weight:900; color:#D45B28; font-family:var(--font-mono);" id="report-confidence">94.2%</span>
            </div>
            <div style="font-size:1.3rem; font-weight:900; color:#2A1A14;" id="report-condition-name">Buni (Tinea Corporis)</div>
            <div style="font-size:0.78rem; color:#846F65;" id="report-local-name">Common Philippine Term: Ringworm</div>
          </div>

          <!-- Morphological Cues Table -->
          <div style="font-size:0.72rem; font-weight:800; text-transform:uppercase; color:#2A1A14; margin-bottom:6px;">
            Identified Fine-Grained Morphological Cues
          </div>
          <table class="ablation-table" style="margin-bottom:14px;">
            <tbody>
              <tr><td style="width:30%; font-weight:700;">Surface Texture</td><td id="report-texture">Active peripheral scaling with central clearing</td></tr>
              <tr><td style="font-weight:700;">Crust & Exudate</td><td id="report-crust">Absent / Dry Erythema / Non-exudative</td></tr>
              <tr><td style="font-weight:700;">Boundary / Edge</td><td id="report-boundary">Sharply defined annular elevated ring</td></tr>
              <tr><td style="font-weight:700;">Differential Key</td><td id="report-diff">Distinguished from Impetigo by ring border & absence of honey crust</td></tr>
            </tbody>
          </table>

          <!-- Model & Attribution Telemetry -->
          <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:6px; background:#FBF0E9; padding:8px; border-radius:6px; font-size:0.7rem; text-align:center; margin-bottom:14px;">
            <div><div style="font-size:0.6rem; color:#846F65;">YOLOv26 mAP50</div><strong style="color:#2A1A14;">92.2%</strong></div>
            <div><div style="font-size:0.6rem; color:#846F65;">Inference Latency</div><strong style="color:#2A1A14;">12.4 ms</strong></div>
            <div><div style="font-size:0.6rem; color:#846F65;">Grad-CAM IoU</div><strong style="color:#2A1A14;">78.2%</strong></div>
            <div><div style="font-size:0.6rem; color:#846F65;">Attribution Ratio</div><strong style="color:#15803D;">73.1%</strong></div>
          </div>

          <!-- Medical Disclaimer & Signatures -->
          <div class="compliance-box" style="margin-bottom:14px;">
            <div class="compliance-box-title">MANDATORY MEDICAL SCREENING DISCLAIMER (RA 10173 COMPLIANT)</div>
            <div class="compliance-box-desc">
              This report is generated by an assistive deep learning algorithm (PUP CCIS Thesis 2026). It does not represent a legally binding dermatological diagnosis. Final clinical diagnosis and pharmaceutical intervention must be validated by a board-certified dermatologist.
            </div>
          </div>

          <div style="display:flex; justify-content:space-between; margin-top:20px; padding-top:12px; border-top:1px dashed #F0C4AD;">
            <div style="font-size:0.72rem; color:#846F65;">
              Evaluated by: <strong>IDENTI-SKIN Clinical System v2.6</strong><br>
              Local Data Integrity: Verified On-Device
            </div>
            <div style="text-align:center; min-width:180px;">
              <div style="border-bottom:1px solid #2A1A14; width:100%; height:25px;"></div>
              <div style="font-size:0.68rem; color:#4A3830; margin-top:4px;">Attending Physician / Dermatologist</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ====================================================================
         MODAL 3: PRE-SCREENING SYMPTOMS TRIAGE QUESTIONNAIRE
         ==================================================================== -->
    <div id="triage-modal" class="modal-overlay hidden" onclick="closeTriageModal()">
      <div class="modal-card" style="max-width:560px;" onclick="event.stopPropagation()">
        <div class="modal-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <div class="brand-icon" style="width:32px; height:32px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFF" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg></div>
            <div>
              <h3 style="font-size:1.05rem; font-weight:800; color:var(--text-title);">Pre-Screening Lesion Triage</h3>
              <div style="font-size:0.7rem; color:var(--text-muted);">Philippine Clinical Symptoms Questionnaire</div>
            </div>
          </div>
          <button class="modal-close-btn" onclick="closeTriageModal()">×</button>
        </div>

        <form id="triage-form" onsubmit="handleTriageSubmit(event)">
          <div style="margin-bottom:14px;">
            <label class="form-label">1. Saan matatagpuan ang lesion / pantal? (Anatomical Location)</label>
            <select id="triage-location" class="form-input">
              <option value="body">Katawan / Braso / Binti (Trunk, Arms, Legs)</option>
              <option value="face">Mukha / Labi (Face, Perioral)</option>
              <option value="feet">Paa / Pagitan ng Daliri (Feet, Interdigital webs)</option>
              <option value="hands">Kamay / Daliri (Hands, Palms)</option>
              <option value="chest">Dibdib / Likod (Chest, Upper Back)</option>
            </select>
          </div>

          <div style="margin-bottom:14px;">
            <label class="form-label">2. Ano ang pangunahing nararamdaman? (Primary Sensation)</label>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
              <label class="toggle-pill active" style="font-size:0.72rem;">
                <input type="radio" name="sensation" value="itchy" checked style="margin-right:6px;"> Makati (Pruritic / Itchy)
              </label>
              <label class="toggle-pill" style="font-size:0.72rem;">
                <input type="radio" name="sensation" value="painful" style="margin-right:6px;"> Masakit / Mahapdi (Painful / Tender)
              </label>
              <label class="toggle-pill" style="font-size:0.72rem;">
                <input type="radio" name="sensation" value="rough" style="margin-right:6px;"> Magaspang / Buko (Rough / Asymptomatic)
              </label>
              <label class="toggle-pill" style="font-size:0.72rem;">
                <input type="radio" name="sensation" value="none" style="margin-right:6px;"> Walang maramdam (No sensation)
              </label>
            </div>
          </div>

          <div style="margin-bottom:14px;">
            <label class="form-label">3. Hitsura ng Sugat o Balat (Visual Cues)</label>
            <select id="triage-visual" class="form-input">
              <option value="ring">Pabilog na may namumulang gilid at malinis ang gitna (Annular Ring)</option>
              <option value="honey">Kulay pulut-pukyutan na langib / basa (Honey-colored crusts)</option>
              <option value="wart">Parang cauliflower na magaspang na bukol (Verrucous rough papule)</option>
              <option value="patches">Maputing patse na may pinong balakubak (Hypopigmented scale)</option>
              <option value="macerated">Basang puting balat sa pagitan ng daliri (Macerated skin)</option>
              <option value="blisters">Maliit na paltos na may tubig (Fluid-filled vesicles)</option>
            </select>
          </div>

          <div style="margin-bottom:16px;">
            <label class="form-label">4. Gaano na katagal ang kondisyon? (Duration)</label>
            <select id="triage-duration" class="form-input">
              <option value="acute">Wala pang 1 linggo (Less than 1 week)</option>
              <option value="subacute">1 hanggang 4 na linggo (1-4 weeks)</option>
              <option value="chronic">Higit sa 1 buwan (Chronic > 1 month)</option>
            </select>
          </div>

          <button type="submit" class="view-btn active" style="width:100%; padding:12px; justify-content:center; font-size:0.85rem;">
            Match With AI Clinical Presets →
          </button>
        </form>
      </div>
    </div>
'''

# Insert before </body>
pos_body = html.rfind('</body>')
if pos_body != -1:
    html = html[:pos_body] + modals_html + '\n' + html[pos_body:]
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("Modals added to index.html successfully!")
