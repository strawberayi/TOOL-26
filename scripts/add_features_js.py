features_js = '''
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

  updateAuthUI();
  closeAuthModal();
  alert(`Welcome, ${AppState.currentUser.name}!\nLogged in as: ${role === 'clinician' ? 'Dermatologist / CCIS Researcher' : 'Patient User'}`);

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

  updateAuthUI();
  closeAuthModal();
  alert(`Successfully signed in as ${AppState.currentUser.name}!`);
}

function updateAuthUI() {
  const label = document.getElementById('auth-btn-label');
  if (label && AppState.currentUser) {
    label.textContent = AppState.currentUser.role === 'clinician' ? '👨‍⚕️ ' + AppState.currentUser.name : '👤 ' + AppState.currentUser.name;
  }
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
  alert(`AI Triage Recommendation:\\nBased on your reported visual cues and symptoms, the highest probability correlation is ${data.name} (${data.localName}) with ${data.confidence}% confidence score.\\n\\nLoading full workspace analysis...`);

  if (window.innerWidth <= 768) {
    switchMobileTab('patient');
  } else {
    switchViewMode('workstation');
  }
}

// Initialize Auth label on load
document.addEventListener('DOMContentLoaded', () => {
  updateAuthUI();
});
'''

with open('app.js', 'a', encoding='utf-8') as f:
    f.write('\n' + features_js)

print("Appended features JS logic successfully!")
