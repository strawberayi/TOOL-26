mobile_css = '''
/* ==========================================================================
   DEDICATED FULL-SCREEN NATIVE MOBILE EXPERIENCE (< 768px)
   ========================================================================== */
#mobile-native-view {
  display: none;
}

@media (max-width: 768px) {
  html, body {
    width: 100vw !important;
    max-width: 100vw !important;
    overflow-x: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    background: var(--bg-app) !important;
  }

  .global-header {
    display: none !important;
  }

  .main-viewport {
    display: none !important;
  }

  .mobile-persistent-nav {
    display: none !important;
  }

  #mobile-native-view {
    display: flex !important;
    flex-direction: column;
    width: 100vw;
    min-height: 100dvh;
    background: var(--bg-app);
    box-sizing: border-box;
  }

  .mobile-native-header {
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    padding-top: calc(10px + env(safe-area-inset-top, 0px));
    background: rgba(255, 247, 242, 0.98);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--brand-border);
    box-shadow: 0 1px 4px rgba(42, 26, 20, 0.04);
  }

  .mobile-header-icon-btn {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    border: 1px solid var(--brand-border);
    background: #FFF;
    color: var(--text-title);
    cursor: pointer;
    box-shadow: var(--shadow-sm);
  }

  .mobile-header-icon-btn:active {
    background: var(--brand-primary-light);
    transform: scale(0.96);
  }

  .mobile-header-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
  }

  .mobile-brand-icon-mini {
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #FF6F3C, #D45B28);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 6px rgba(212, 91, 40, 0.3);
  }

  .mobile-brand-title {
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: var(--text-title);
  }

  .mobile-native-content {
    flex: 1;
    width: 100vw;
    box-sizing: border-box;
    padding: 14px 14px calc(76px + env(safe-area-inset-bottom, 0px));
    overflow-y: auto;
    overflow-x: hidden;
    -webkit-overflow-scrolling: touch;
  }

  .mob-tab-pane {
    display: none;
    width: 100%;
  }

  .mob-tab-pane.active {
    display: block;
    animation: mobFadeIn 0.22s ease-out;
  }

  @keyframes mobFadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .mob-greeting-box {
    margin-bottom: 14px;
  }

  .mob-greeting-title {
    font-size: 1.4rem;
    font-weight: 900;
    color: var(--text-title);
    line-height: 1.2;
    margin-bottom: 2px;
  }

  .mob-greeting-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .mob-upload-card {
    padding: 18px 12px;
    margin-bottom: 14px;
    border-radius: var(--radius-md);
  }

  .mob-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 16px 0 10px;
  }

  .mob-section-title {
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-title);
  }

  .mob-section-link {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--brand-primary);
    cursor: pointer;
  }

  .mob-presets-list {
    gap: 8px;
  }

  .mob-presets-carousel {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 8px;
    margin-bottom: 12px;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
  }

  .mob-presets-carousel::-webkit-scrollbar {
    display: none;
  }

  .mob-preset-chip {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    border: 1px solid var(--brand-border);
    background: #FFF;
    color: var(--text-body);
    white-space: nowrap;
    cursor: pointer;
    flex-shrink: 0;
    transition: all 0.18s;
  }

  .mob-preset-chip.active {
    background: var(--brand-primary);
    color: #FFF;
    border-color: var(--brand-primary);
    box-shadow: 0 2px 6px rgba(212, 91, 40, 0.3);
  }

  .mobile-native-bottom-nav {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 58px;
    padding-bottom: env(safe-area-inset-bottom, 0px);
    background: rgba(255, 255, 255, 0.98);
    backdrop-filter: blur(14px);
    border-top: 1px solid var(--brand-border);
    display: flex;
    align-items: center;
    justify-content: space-around;
    z-index: 1000;
    box-shadow: 0 -2px 12px rgba(42, 26, 20, 0.06);
  }

  .mob-nav-btn {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text-muted);
    background: transparent;
    border: none;
    padding: 6px 0;
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
  }

  .mob-nav-btn svg {
    width: 20px;
    height: 20px;
  }

  .mob-nav-btn.active {
    color: var(--brand-primary);
  }

  .mob-nav-btn.active svg {
    stroke: var(--brand-primary);
    transform: scale(1.1);
  }

  .mobile-drawer-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(17, 11, 8, 0.65);
    backdrop-filter: blur(4px);
    z-index: 2000;
    display: flex;
  }

  .mobile-drawer-panel {
    width: 290px;
    height: 100%;
    background: #FFF;
    padding: 20px 16px;
    box-sizing: border-box;
    box-shadow: 4px 0 25px rgba(0,0,0,0.25);
    overflow-y: auto;
    animation: drawerSlide 0.22s ease-out;
  }

  @keyframes drawerSlide {
    from { transform: translateX(-100%); }
    to { transform: translateX(0); }
  }

  .mobile-drawer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--brand-border);
    padding-bottom: 12px;
  }
}
'''

with open('styles.css', 'a', encoding='utf-8') as f:
    f.write('\n' + mobile_css)

print("Appended dedicated native mobile CSS successfully!")
