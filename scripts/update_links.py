with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Update header actions with direct Login link
old_auth_btn = '''<button id="auth-header-btn" class="view-btn active" style="padding:6px 12px;" onclick="openAuthModal('login')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            <span id="auth-btn-label">Sign In</span>
          </button>'''

new_auth_btn = '''<a href="login.html" class="view-btn active" style="padding:6px 14px; text-decoration:none;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"></path><polyline points="10 17 15 12 10 7"></polyline><line x1="15" y1="12" x2="3" y2="12"></line></svg>
            <span>Login Page</span>
          </a>
          <button id="auth-header-btn" class="view-btn" style="padding:6px 10px;" onclick="openAuthModal('login')">
            <span id="auth-btn-label">👤 Account</span>
          </button>'''

if old_auth_btn in html:
    html = html.replace(old_auth_btn, new_auth_btn)
    print("Updated header with direct Login Page link!")

# 2. In mobile drawer, make Login Page prominent
old_drawer_tools = '<div style="font-size:0.7rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:8px;">Features & Tools</div>'
new_drawer_tools = '''<div style="font-size:0.7rem; font-weight:800; text-transform:uppercase; color:var(--text-muted); margin-bottom:8px;">Authentication & Tools</div>
            <a href="login.html" class="upload-btn" style="margin-bottom:6px; width:100%; text-decoration:none; background:var(--brand-primary); color:#FFF; border-color:var(--brand-primary); font-weight:800;">🔐 Open Dedicated Login Page</a>
            <button class="upload-btn" style="margin-bottom:6px; width:100%;" onclick="toggleMobileDrawer(false); openAuthModal('login');">👤 Switch User / Account</button>'''

if old_drawer_tools in html:
    html = html.replace(old_drawer_tools, new_drawer_tools)
    print("Updated mobile drawer with Login Page link!")

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)
