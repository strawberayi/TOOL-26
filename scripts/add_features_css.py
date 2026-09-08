features_css = '''
/* ==========================================================================
   AUTHENTICATION, FORMS, REPORT & TRIAGE STYLING
   ========================================================================== */
.form-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.form-label {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--text-title);
}

.form-input {
  width: 100%;
  box-sizing: border-box;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid var(--brand-border);
  background: #FFF;
  font-size: 0.8rem;
  color: var(--text-title);
  outline: none;
  font-family: inherit;
  transition: all 0.18s;
}

.form-input:focus {
  border-color: var(--brand-primary);
  box-shadow: 0 0 0 3px rgba(212, 91, 40, 0.15);
}

.auth-card {
  max-width: 440px;
}

.report-card {
  max-width: 780px;
  max-height: 92vh;
  overflow-y: auto;
}

.report-doc-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 2px solid var(--brand-primary);
  padding-bottom: 12px;
}

@media print {
  body * {
    visibility: hidden;
  }
  #report-modal,
  #report-modal * {
    visibility: visible;
  }
  #report-modal {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    background: #FFF;
    padding: 0;
  }
  .modal-card {
    box-shadow: none;
    border: none;
    width: 100%;
    max-width: 100%;
    padding: 0;
  }
  .modal-header {
    display: none !important;
  }
}
'''

with open('styles.css', 'a', encoding='utf-8') as f:
    f.write('\n' + features_css)

print("Added features CSS successfully!")
