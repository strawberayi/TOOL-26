login_html = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>Sign In | IDENTI-SKIN (PUP CCIS)</title>
  <link rel="stylesheet" href="styles.css">
  <style>
    /* Dedicated full-screen Login Page Styles */
    .login-body {
      min-height: 100vh;
      min-height: 100dvh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: radial-gradient(circle at 50% 20%, #FFEFE6 0%, #FFF8F3 60%, #FBF0E9 100%);
      padding: 16px;
      box-sizing: border-box;
      position: relative;
      overflow-x: hidden;
    }

    /* Subtle background concentric circles matching 1st Page */
    .login-body::before {
      content: '';
      position: absolute;
      width: 650px;
      height: 650px;
      border-radius: 50%;
      border: 1px dashed rgba(212, 91, 40, 0.15);
      top: -120px;
      left: 50%;
      transform: translateX(-50%);
      pointer-events: none;
    }
    .login-body::after {
      content: '';
      position: absolute;
      width: 900px;
      height: 900px;
      border-radius: 50%;
      border: 1px solid rgba(212, 91, 40, 0.08);
      top: -240px;
      left: 50%;
      transform: translateX(-50%);
      pointer-events: none;
    }

    .login-container {
      width: 100%;
      max-width: 440px;
      background: #FFFFFF;
      border: 1px solid var(--brand-border);
      border-radius: var(--radius-xl);
      box-shadow: 0 12px 35px rgba(42, 26, 20, 0.08), 0 1px 3px rgba(42, 26, 20, 0.04);
      padding: 32px 24px;
      box-sizing: border-box;
      position: relative;
      z-index: 10;
    }

    @media (max-width: 480px) {
      .login-body {
        padding: 0;
        background: #FFF8F3;
      }
      .login-container {
        max-width: 100vw;
        min-height: 100dvh;
        border-radius: 0;
        border: none;
        box-shadow: none;
        padding: 40px 20px 30px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }
    }

    .login-brand-hero {
      text-align: center;
      margin-bottom: 24px;
    }

    .login-shield-logo {
      width: 72px;
      height: 72px;
      background: linear-gradient(135deg, #FF6F3C 0%, #D45B28 100%);
      border-radius: 22px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 16px;
      box-shadow: 0 8px 20px rgba(212, 91, 40, 0.32);
      color: #FFF;
      transition: transform 0.2s;
    }
    .login-shield-logo:hover {
      transform: scale(1.05);
    }

    .login-main-title {
      font-size: 1.6rem;
      font-weight: 900;
      letter-spacing: 1px;
      color: var(--text-title);
      margin-bottom: 6px;
    }

    .login-sub-tagline {
      font-size: 0.78rem;
      color: var(--text-muted);
      line-height: 1.4;
      max-width: 300px;
      margin: 0 auto;
    }

    .demo-strip {
      background: var(--bg-subtle);
      border: 1px solid var(--brand-border);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 18px;
    }

    .demo-strip-title {
      font-size: 0.68rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .demo-btns-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .demo-quick-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--brand-border);
      background: #FFF;
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-title);
      cursor: pointer;
      transition: all 0.18s;
    }
    .demo-quick-btn:hover {
      border-color: var(--brand-primary);
      color: var(--brand-primary);
      background: var(--brand-primary-light);
    }

    .login-compliance-footer {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font-size: 0.7rem;
      color: var(--text-muted);
      margin-top: 24px;
      text-align: center;
    }
  </style>
</head>
<body class="login-body">

  <div class="login-container">
    <div>
      <!-- Brand & Shield Logo matching 1st Page -->
      <div class="login-brand-hero">
        <div class="login-shield-logo">
          <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            <line x1="12" y1="8" x2="12" y2="16"></line>
            <line x1="8" y1="12" x2="16" y2="12"></line>
          </svg>
        </div>
        <h1 class="login-main-title">IDENTI - SKIN</h1>
        <p class="login-sub-tagline">
          AI - Powered Skin Infection Analysis for Smarter, Faster, and Safer Decision
        </p>
      </div>

      <!-- Quick 1-Tap Demo Logins -->
      <div class="demo-strip">
        <div class="demo-strip-title">
          <span>⚡ 1-Tap Instant Demo Access</span>
        </div>
        <div class="demo-btns-row">
          <button type="button" class="demo-quick-btn" onclick="quickLogin('clinician')">
            👨‍⚕️ Clinician
          </button>
          <button type="button" class="demo-quick-btn" onclick="quickLogin('patient')">
            👤 Patient
          </button>
        </div>
      </div>

      <!-- Tab Switcher: Sign In vs Register -->
      <div class="output-tab-bar" style="margin-bottom: 16px;">
        <button id="tab-btn-signin" class="output-tab-btn active" onclick="switchAuthMode('signin')">Sign In</button>
        <button id="tab-btn-signup" class="output-tab-btn" onclick="switchAuthMode('signup')">Create Account</button>
      </div>

      <!-- Sign In Form -->
      <form id="login-form" onsubmit="submitAuth(event, 'signin')">
        <div class="form-group" style="margin-bottom: 12px;">
          <label class="form-label">Email or PUP WebMail</label>
          <input type="email" id="input-email" class="form-input" placeholder="e.g. researcher@pup.edu.ph" value="dr.santos@pup.edu.ph" required>
        </div>

        <div class="form-group" style="margin-bottom: 14px;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <label class="form-label">Password</label>
            <a href="#" style="font-size: 0.68rem; color: var(--brand-primary); text-decoration: none;" onclick="alert('Demo password is: password123')">Forgot?</a>
          </div>
          <input type="password" id="input-password" class="form-input" placeholder="••••••••" value="password123" required>
        </div>

        <div class="form-group" style="margin-bottom: 16px;">
          <label class="form-label">Role Access</label>
          <select id="select-role" class="form-input">
            <option value="clinician">Dermatologist / Clinical Researcher (Full Dashboard & Telemetry)</option>
            <option value="patient">Patient User (Simplified View & Localized Cues)</option>
          </select>
        </div>

        <button type="submit" class="view-btn active" style="width: 100%; padding: 12px; justify-content: center; font-size: 0.88rem; font-weight: 700; margin-bottom: 10px;">
          Sign In to IDENTI-SKIN →
        </button>

        <button type="button" class="upload-btn" style="width: 100%; padding: 11px; justify-content: center; font-size: 0.82rem;" onclick="continueAsGuest()">
          📷 Continue as Guest (No Login Required)
        </button>
      </form>

      <!-- Create Account Form (Hidden by default) -->
      <form id="signup-form" class="hidden" onsubmit="submitAuth(event, 'signup')">
        <div class="form-group" style="margin-bottom: 10px;">
          <label class="form-label">Full Name</label>
          <input type="text" id="reg-fullname" class="form-input" placeholder="e.g. Maria Santos" required>
        </div>

        <div class="form-group" style="margin-bottom: 10px;">
          <label class="form-label">Email Address</label>
          <input type="email" id="reg-email-input" class="form-input" placeholder="e.g. maria@gmail.com" required>
        </div>

        <div class="form-group" style="margin-bottom: 10px;">
          <label class="form-label">Create Password</label>
          <input type="password" id="reg-password-input" class="form-input" placeholder="At least 6 characters" required>
        </div>

        <div class="form-group" style="margin-bottom: 12px;">
          <label class="form-label">Account Role</label>
          <select id="reg-role-select" class="form-input">
            <option value="clinician">Dermatologist / Medical Researcher</option>
            <option value="patient" selected>Patient / Public User</option>
          </select>
        </div>

        <div style="display: flex; align-items: flex-start; gap: 8px; margin-bottom: 16px; font-size: 0.72rem; color: var(--text-body);">
          <input type="checkbox" id="reg-checkbox" checked required style="margin-top: 2px;">
          <label for="reg-checkbox">
            I agree to the <strong>Data Privacy Act of 2012 (RA 10173)</strong> terms. All skin images remain protected on-device.
          </label>
        </div>

        <button type="submit" class="view-btn active" style="width: 100%; padding: 12px; justify-content: center; font-size: 0.88rem; font-weight: 700;">
          Create New Account →
        </button>
      </form>
    </div>

    <!-- Compliance Footer matching 1st Page -->
    <div class="login-compliance-footer">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#15803D" stroke-width="2.5">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
        <polyline points="9 12 11 14 15 10"></polyline>
      </svg>
      <span>Your data is protected & safe • <strong>RA 10173 Compliant</strong></span>
    </div>
  </div>

  <script>
    function switchAuthMode(mode) {
      const btnIn = document.getElementById('tab-btn-signin');
      const btnUp = document.getElementById('tab-btn-signup');
      const formIn = document.getElementById('login-form');
      const formUp = document.getElementById('signup-form');

      if (mode === 'signin') {
        btnIn.classList.add('active');
        btnUp.classList.remove('active');
        formIn.classList.remove('hidden');
        formUp.classList.add('hidden');
      } else {
        btnIn.classList.remove('active');
        btnUp.classList.add('active');
        formIn.classList.add('hidden');
        formUp.classList.remove('hidden');
      }
    }

    function quickLogin(role) {
      const user = role === 'clinician' ? {
        name: 'Dr. Maria Santos',
        email: 'm.santos@pup.edu.ph',
        role: 'clinician'
      } : {
        name: 'Juan Dela Cruz',
        email: 'juan.delacruz@gmail.com',
        role: 'patient'
      };

      localStorage.setItem('identi_skin_user', JSON.stringify(user));
      window.location.href = 'index.html' + (role === 'clinician' ? '#panel' : '#patient');
    }

    function submitAuth(e, mode) {
      e.preventDefault();
      let user;
      if (mode === 'signin') {
        const email = document.getElementById('input-email').value;
        const role = document.getElementById('select-role').value;
        const name = (role === 'clinician' ? 'Dr. ' : '') + email.split('@')[0].replace('.', ' ').replace(/(^\\w|\\s\\w)/g, m => m.toUpperCase());
        user = { name, email, role };
      } else {
        const name = document.getElementById('reg-fullname').value;
        const email = document.getElementById('reg-email-input').value;
        const role = document.getElementById('reg-role-select').value;
        user = { name, email, role };
      }

      localStorage.setItem('identi_skin_user', JSON.stringify(user));
      window.location.href = 'index.html' + (user.role === 'clinician' ? '#panel' : '#patient');
    }

    function continueAsGuest() {
      const guest = { name: 'Guest User', email: 'guest@identiskin.ph', role: 'patient' };
      localStorage.setItem('identi_skin_user', JSON.stringify(guest));
      window.location.href = 'index.html#home';
    }
  </script>
</body>
</html>
'''

with open('login.html', 'w', encoding='utf-8') as f:
    f.write(login_html)

print("login.html created successfully!")
