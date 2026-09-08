# 1. WELCOME.HTML (Exact 1st Page from reference)
welcome_html = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>Welcome | IDENTI-SKIN (PUP CCIS)</title>
  <link rel="stylesheet" href="styles.css">
  <style>
    body, html {
      margin: 0;
      padding: 0;
      width: 100vw;
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      background: radial-gradient(circle at 50% 25%, #FFF0E6 0%, #FFF8F3 60%, #FDF1EA 100%);
      font-family: var(--font-sans);
      display: flex;
      align-items: center;
      justify-content: center;
    }

    /* Subtle decorative curved rings from reference 1st Page */
    .bg-wave-1 {
      position: absolute;
      width: 580px;
      height: 580px;
      border-radius: 50%;
      border: 1px dashed rgba(212, 91, 40, 0.18);
      top: -80px;
      left: 50%;
      transform: translateX(-50%);
      pointer-events: none;
    }
    .bg-wave-2 {
      position: absolute;
      width: 820px;
      height: 820px;
      border-radius: 50%;
      border: 1px solid rgba(212, 91, 40, 0.08);
      top: -200px;
      left: 50%;
      transform: translateX(-50%);
      pointer-events: none;
    }

    .welcome-card {
      width: 100%;
      max-width: 400px;
      height: 100%;
      max-height: 720px;
      background: #FFFFFF;
      border: 1px solid var(--brand-border);
      border-radius: 36px;
      box-shadow: 0 16px 40px rgba(42, 26, 20, 0.1);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      align-items: center;
      padding: 50px 24px 34px;
      box-sizing: border-box;
      position: relative;
      z-index: 10;
      text-align: center;
    }

    @media (max-width: 480px) {
      body, html {
        background: #FFF8F3;
      }
      .welcome-card {
        max-width: 100vw;
        max-height: 100dvh;
        border-radius: 0;
        border: none;
        box-shadow: none;
        padding: 60px 24px 36px;
      }
    }

    .welcome-logo-box {
      width: 84px;
      height: 84px;
      background: linear-gradient(135deg, #FF6F3C 0%, #D45B28 100%);
      border-radius: 26px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 20px;
      box-shadow: 0 10px 25px rgba(212, 91, 40, 0.35);
      color: #FFF;
      animation: logoFloat 3s ease-in-out infinite alternate;
    }

    @keyframes logoFloat {
      from { transform: translateY(0); }
      to { transform: translateY(-6px); }
    }

    .welcome-title {
      font-size: 1.75rem;
      font-weight: 900;
      letter-spacing: 1.2px;
      color: var(--text-title);
      margin-bottom: 8px;
    }

    .welcome-tagline {
      font-size: 0.8rem;
      color: var(--text-muted);
      line-height: 1.45;
      max-width: 270px;
      margin: 0 auto;
    }

    .welcome-actions-group {
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin-top: 40px;
      margin-bottom: 20px;
    }

    .btn-get-started {
      width: 100%;
      box-sizing: border-box;
      padding: 15px 20px;
      background: var(--brand-primary);
      color: #FFFFFF;
      border: none;
      border-radius: 16px;
      font-size: 0.92rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      text-decoration: none;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(212, 91, 40, 0.35);
      transition: all 0.2s ease;
    }
    .btn-get-started:hover {
      background: var(--brand-primary-hover);
      transform: translateY(-2px);
    }

    .btn-have-account {
      width: 100%;
      box-sizing: border-box;
      padding: 14px 20px;
      background: #FFFFFF;
      color: var(--text-title);
      border: 1.5px solid var(--brand-border);
      border-radius: 16px;
      font-size: 0.92rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      text-decoration: none;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-have-account:hover {
      border-color: var(--brand-primary);
      color: var(--brand-primary);
      background: var(--brand-primary-light);
    }

    .welcome-footer-compliance {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      font-size: 0.72rem;
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <div class="bg-wave-1"></div>
  <div class="bg-wave-2"></div>

  <div class="welcome-card">
    <div style="width: 100%;">
      <!-- Shield Logo -->
      <div class="welcome-logo-box">
        <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
          <line x1="12" y1="8" x2="12" y2="16"></line>
          <line x1="8" y1="12" x2="16" y2="12"></line>
        </svg>
      </div>

      <!-- App Title & Tagline -->
      <h1 class="welcome-title">IDENTI - SKIN</h1>
      <p class="welcome-tagline">
        AI - Powered Skin Infection Analysis for Smarter, Faster, and Safer Decision
      </p>

      <!-- Action Buttons matching 1st Page -->
      <div class="welcome-actions-group">
        <a href="index.html#home" class="btn-get-started">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
            <circle cx="12" cy="13" r="4"></circle>
          </svg>
          Get Started
        </a>

        <a href="login.html" class="btn-have-account">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
            <circle cx="12" cy="7" r="4"></circle>
          </svg>
          I Have an Account
        </a>
      </div>
    </div>

    <!-- Security Compliance Footer -->
    <div class="welcome-footer-compliance">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#15803D" stroke-width="2.5">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
        <polyline points="9 12 11 14 15 10"></polyline>
      </svg>
      <span>Your data is protected & safe - <strong>RA 10173 Compliant</strong></span>
    </div>
  </div>
</body>
</html>
'''

# 2. LOGIN.HTML (Pure Dedicated Sign In Page)
pure_login_html = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>Sign In | IDENTI-SKIN</title>
  <link rel="stylesheet" href="styles.css">
  <style>
    body, html {
      margin: 0;
      padding: 0;
      width: 100vw;
      min-height: 100vh;
      min-height: 100dvh;
      background: radial-gradient(circle at 50% 15%, #FFF0E6 0%, #FFF8F3 60%, #FDF1EA 100%);
      font-family: var(--font-sans);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
      box-sizing: border-box;
    }

    .auth-page-box {
      width: 100%;
      max-width: 420px;
      background: #FFFFFF;
      border: 1px solid var(--brand-border);
      border-radius: var(--radius-xl);
      box-shadow: 0 12px 35px rgba(42, 26, 20, 0.08);
      padding: 30px 24px;
      box-sizing: border-box;
    }

    @media (max-width: 480px) {
      body, html {
        padding: 0;
        background: #FFF8F3;
      }
      .auth-page-box {
        max-width: 100vw;
        min-height: 100dvh;
        border-radius: 0;
        border: none;
        box-shadow: none;
        padding: 30px 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }
    }

    .auth-back-link {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-muted);
      text-decoration: none;
      margin-bottom: 18px;
      transition: color 0.15s;
    }
    .auth-back-link:hover {
      color: var(--brand-primary);
    }

    .auth-header-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
    }

    .auth-brand-badge {
      width: 44px;
      height: 44px;
      background: linear-gradient(135deg, #FF6F3C 0%, #D45B28 100%);
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #FFF;
      box-shadow: 0 4px 12px rgba(212, 91, 40, 0.3);
      flex-shrink: 0;
    }

    .auth-heading {
      font-size: 1.35rem;
      font-weight: 900;
      color: var(--text-title);
      line-height: 1.2;
    }

    .auth-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .demo-bar {
      background: var(--bg-subtle);
      border: 1px solid var(--brand-border);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 20px;
    }

    .demo-title {
      font-size: 0.68rem;
      font-weight: 800;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 8px;
    }

    .demo-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .demo-btn {
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--brand-border);
      background: #FFF;
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-title);
      cursor: pointer;
      transition: all 0.18s;
      text-align: center;
    }
    .demo-btn:hover {
      border-color: var(--brand-primary);
      color: var(--brand-primary);
      background: var(--brand-primary-light);
    }
  </style>
</head>
<body>
  <div class="auth-page-box">
    <div>
      <!-- Back button to Welcome / 1st Page -->
      <a href="welcome.html" class="auth-back-link">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
        <span>Back to Welcome</span>
      </a>

      <!-- Header with mini brand -->
      <div class="auth-header-row">
        <div class="auth-brand-badge">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            <line x1="12" y1="8" x2="12" y2="16"></line>
            <line x1="8" y1="12" x2="16" y2="12"></line>
          </svg>
        </div>
        <div>
          <h2 class="auth-heading">Sign In</h2>
          <div class="auth-sub">Access your clinical dashboard or patient screening</div>
        </div>
      </div>

      <!-- 1-Tap Quick Demo Access -->
      <div class="demo-bar">
        <div class="demo-title">⚡ 1-Tap Quick Demo Access</div>
        <div class="demo-grid">
          <button type="button" class="demo-btn" onclick="directLogin('clinician')">
            👨‍⚕️ Clinician
          </button>
          <button type="button" class="demo-btn" onclick="directLogin('patient')">
            👤 Patient
          </button>
        </div>
      </div>

      <!-- Sign In Form -->
      <form id="pure-signin-form" onsubmit="handlePureLogin(event)">
        <div class="form-group" style="margin-bottom: 14px;">
          <label class="form-label">Email or PUP WebMail</label>
          <input type="email" id="f-email" class="form-input" placeholder="e.g. researcher@pup.edu.ph" value="dr.santos@pup.edu.ph" required>
        </div>

        <div class="form-group" style="margin-bottom: 14px;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <label class="form-label">Password</label>
            <a href="#" style="font-size:0.68rem; color:var(--brand-primary); text-decoration:none;" onclick="alert('Demo password: password123')">Forgot?</a>
          </div>
          <input type="password" id="f-pass" class="form-input" placeholder="••••••••" value="password123" required>
        </div>

        <div class="form-group" style="margin-bottom: 18px;">
          <label class="form-label">Account Role</label>
          <select id="f-role" class="form-input">
            <option value="clinician">Dermatologist / Clinical Researcher (Full Dashboard & Telemetry)</option>
            <option value="patient">Patient User (Simplified View & Localized Cues)</option>
          </select>
        </div>

        <button type="submit" class="view-btn active" style="width:100%; padding:14px; justify-content:center; font-size:0.9rem; font-weight:800; margin-bottom:12px;">
          Sign In to Account →
        </button>
      </form>

      <!-- Switch to Register -->
      <div style="text-align:center; margin-top:14px; font-size:0.75rem; color:var(--text-muted);">
        Don't have an account yet? <a href="register.html" style="color:var(--brand-primary); font-weight:700; text-decoration:none;">Create an Account</a>
      </div>
    </div>

    <!-- Security Compliance Footer -->
    <div style="display:flex; align-items:center; justify-content:center; gap:6px; font-size:0.68rem; color:var(--text-muted); margin-top:24px;">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#15803D" stroke-width="2.5">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
        <polyline points="9 12 11 14 15 10"></polyline>
      </svg>
      <span>Protected under <strong>RA 10173 (Data Privacy Act of 2012)</strong></span>
    </div>
  </div>

  <script>
    function directLogin(role) {
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

    function handlePureLogin(e) {
      e.preventDefault();
      const email = document.getElementById('f-email').value;
      const role = document.getElementById('f-role').value;
      const name = (role === 'clinician' ? 'Dr. ' : '') + email.split('@')[0].replace('.', ' ').replace(/(^\\w|\\s\\w)/g, m => m.toUpperCase());

      const user = { name, email, role };
      localStorage.setItem('identi_skin_user', JSON.stringify(user));
      window.location.href = 'index.html' + (role === 'clinician' ? '#panel' : '#patient');
    }
  </script>
</body>
</html>
'''

# 3. REGISTER.HTML (Pure Dedicated Create Account Page)
pure_register_html = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>Create Account | IDENTI-SKIN</title>
  <link rel="stylesheet" href="styles.css">
  <style>
    body, html {
      margin: 0;
      padding: 0;
      width: 100vw;
      min-height: 100vh;
      min-height: 100dvh;
      background: radial-gradient(circle at 50% 15%, #FFF0E6 0%, #FFF8F3 60%, #FDF1EA 100%);
      font-family: var(--font-sans);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
      box-sizing: border-box;
    }

    .auth-page-box {
      width: 100%;
      max-width: 420px;
      background: #FFFFFF;
      border: 1px solid var(--brand-border);
      border-radius: var(--radius-xl);
      box-shadow: 0 12px 35px rgba(42, 26, 20, 0.08);
      padding: 30px 24px;
      box-sizing: border-box;
    }

    @media (max-width: 480px) {
      body, html {
        padding: 0;
        background: #FFF8F3;
      }
      .auth-page-box {
        max-width: 100vw;
        min-height: 100dvh;
        border-radius: 0;
        border: none;
        box-shadow: none;
        padding: 30px 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }
    }

    .auth-back-link {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-muted);
      text-decoration: none;
      margin-bottom: 18px;
      transition: color 0.15s;
    }
    .auth-back-link:hover {
      color: var(--brand-primary);
    }

    .auth-header-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
    }

    .auth-brand-badge {
      width: 44px;
      height: 44px;
      background: linear-gradient(135deg, #FF6F3C 0%, #D45B28 100%);
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #FFF;
      box-shadow: 0 4px 12px rgba(212, 91, 40, 0.3);
      flex-shrink: 0;
    }

    .auth-heading {
      font-size: 1.35rem;
      font-weight: 900;
      color: var(--text-title);
      line-height: 1.2;
    }

    .auth-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 2px;
    }
  </style>
</head>
<body>
  <div class="auth-page-box">
    <div>
      <!-- Back button to Sign In -->
      <a href="login.html" class="auth-back-link">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
        <span>Back to Sign In</span>
      </a>

      <!-- Header with mini brand -->
      <div class="auth-header-row">
        <div class="auth-brand-badge">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
            <circle cx="8.5" cy="7" r="4"></circle>
            <line x1="20" y1="8" x2="20" y2="14"></line>
            <line x1="23" y1="11" x2="17" y2="11"></line>
          </svg>
        </div>
        <div>
          <h2 class="auth-heading">Create Account</h2>
          <div class="auth-sub">Join the Philippine clinical skin AI research network</div>
        </div>
      </div>

      <!-- Registration Form -->
      <form id="pure-register-form" onsubmit="handlePureRegister(event)">
        <div class="form-group" style="margin-bottom: 12px;">
          <label class="form-label">Full Name</label>
          <input type="text" id="r-name" class="form-input" placeholder="e.g. Maria Santos" required>
        </div>

        <div class="form-group" style="margin-bottom: 12px;">
          <label class="form-label">Email Address</label>
          <input type="email" id="r-email" class="form-input" placeholder="e.g. maria@gmail.com" required>
        </div>

        <div class="form-group" style="margin-bottom: 12px;">
          <label class="form-label">Password</label>
          <input type="password" id="r-pass" class="form-input" placeholder="At least 6 characters" required>
        </div>

        <div class="form-group" style="margin-bottom: 14px;">
          <label class="form-label">Account Role</label>
          <select id="r-role" class="form-input">
            <option value="clinician">Dermatologist / Medical Researcher</option>
            <option value="patient" selected>Patient / Public User</option>
          </select>
        </div>

        <div style="display:flex; align-items:flex-start; gap:8px; margin-bottom:18px; font-size:0.72rem; color:var(--text-body);">
          <input type="checkbox" id="r-consent" checked required style="margin-top:2px;">
          <label for="r-consent">
            I consent to the <strong>Data Privacy Act of 2012 (RA 10173)</strong> policies. All skin images are processed locally on device.
          </label>
        </div>

        <button type="submit" class="view-btn active" style="width:100%; padding:14px; justify-content:center; font-size:0.9rem; font-weight:800; margin-bottom:12px;">
          Register Account →
        </button>
      </form>

      <!-- Switch to Sign In -->
      <div style="text-align:center; margin-top:14px; font-size:0.75rem; color:var(--text-muted);">
        Already have an account? <a href="login.html" style="color:var(--brand-primary); font-weight:700; text-decoration:none;">Sign In here</a>
      </div>
    </div>

    <!-- Security Compliance Footer -->
    <div style="display:flex; align-items:center; justify-content:center; gap:6px; font-size:0.68rem; color:var(--text-muted); margin-top:24px;">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#15803D" stroke-width="2.5">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
        <polyline points="9 12 11 14 15 10"></polyline>
      </svg>
      <span>Protected under <strong>RA 10173 (Data Privacy Act of 2012)</strong></span>
    </div>
  </div>

  <script>
    function handlePureRegister(e) {
      e.preventDefault();
      const name = document.getElementById('r-name').value;
      const email = document.getElementById('r-email').value;
      const role = document.getElementById('r-role').value;

      const user = { name, email, role };
      localStorage.setItem('identi_skin_user', JSON.stringify(user));
      window.location.href = 'index.html' + (role === 'clinician' ? '#panel' : '#patient');
    }
  </script>
</body>
</html>
'''

with open('welcome.html', 'w', encoding='utf-8') as f:
    f.write(welcome_html)
print("welcome.html created!")

with open('login.html', 'w', encoding='utf-8') as f:
    f.write(pure_login_html)
print("login.html updated to pure Sign In page!")

with open('register.html', 'w', encoding='utf-8') as f:
    f.write(pure_register_html)
print("register.html created!")
