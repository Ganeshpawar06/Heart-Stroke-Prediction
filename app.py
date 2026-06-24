import streamlit as st
import pandas as pd
import joblib
import time

# ──────────────────────────────────────────────────────────────────────────
# BACKEND — UNCHANGED. Same model, scaler, columns, inputs, and prediction
# logic as the original app. Do not modify this section.
# ──────────────────────────────────────────────────────────────────────────

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heart Stroke Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --deep-blue: #0F4C81;
    --deep-blue-dark: #0A3A63;
    --teal: #00B8A9;
    --teal-light: #E5F8F6;
    --surface: #FFFFFF;
    --surface-soft: #F6F8FA;
    --ink: #142433;
    --ink-soft: #5C7184;
    --line: #E3E9EE;
    --danger: #D64550;
    --danger-soft: #FDECEC;
    --success: #1FA98A;
    --success-soft: #E6F8F2;
    --shadow-sm: 0 2px 8px rgba(15, 76, 129, 0.06);
    --shadow-md: 0 8px 24px rgba(15, 76, 129, 0.10);
    --shadow-lg: 0 20px 48px rgba(15, 76, 129, 0.14);
}

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }
.mono { font-family: 'IBM Plex Mono', monospace; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1180px; }
.stApp { background: var(--surface); }

/* ---------- Sticky Nav ---------- */
.navbar {
    position: sticky; top: 0; z-index: 999;
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 28px;
    background: rgba(255,255,255,0.78);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--line);
    margin: 0 -100px 0 -100px;
}
.nav-logo { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 18px; color: var(--deep-blue); }
.nav-logo .pulse-dot {
    width: 10px; height: 10px; border-radius: 50%; background: var(--teal);
    box-shadow: 0 0 0 0 rgba(0,184,169,0.6);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0,184,169,0.5); }
    70% { box-shadow: 0 0 0 8px rgba(0,184,169,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,184,169,0); }
}
.nav-links { display: flex; gap: 28px; font-weight: 600; font-size: 14.5px; color: var(--ink-soft); }
.nav-links a { color: var(--ink-soft); text-decoration: none; transition: color .2s; }
.nav-links a:hover { color: var(--teal); }

/* ---------- Hero ---------- */
.hero-wrap {
    position: relative;
    margin: 0 -100px;
    padding: 64px 100px 56px 100px;
    background: linear-gradient(135deg, #0A3A63 0%, #0F4C81 55%, #0E6E72 130%);
    overflow: hidden;
}
.hero-grid { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 48px; align-items: center; position: relative; z-index: 2; }
.eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,184,169,0.16); color: #6FE3D6;
    border: 1px solid rgba(0,184,169,0.35);
    padding: 6px 14px; border-radius: 100px;
    font-size: 12.5px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    margin-bottom: 20px;
}
.hero-title { color: #FFFFFF; font-size: 46px; font-weight: 800; line-height: 1.12; letter-spacing: -0.01em; margin-bottom: 14px; }
.hero-title span { color: #6FE3D6; }
.hero-subtitle { color: #BFD8EC; font-size: 18px; font-weight: 600; margin-bottom: 14px; }
.hero-desc { color: #9FB9CE; font-size: 15.5px; line-height: 1.7; max-width: 560px; margin-bottom: 30px; }

.feature-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 8px; }
.feature-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 14px; padding: 16px 14px;
    backdrop-filter: blur(6px);
    transition: transform .25s, background .25s;
}
.feature-card:hover { transform: translateY(-4px); background: rgba(255,255,255,0.1); }
.feature-card .ic { font-size: 22px; margin-bottom: 8px; }
.feature-card .ft { color: #fff; font-weight: 700; font-size: 13.5px; margin-bottom: 2px; }
.feature-card .fd { color: #A9C2D6; font-size: 11.5px; line-height: 1.4; }

/* ECG illustration */
.ecg-wrap {
    position: relative; height: 280px;
    border-radius: 22px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.14);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
}
.ecg-wrap svg { width: 100%; }
.ecg-line {
    fill: none; stroke: #6FE3D6; stroke-width: 2.5;
    stroke-dasharray: 1200; stroke-dashoffset: 1200;
    animation: draw 3.2s ease-in-out infinite;
    filter: drop-shadow(0 0 6px rgba(111,227,214,0.7));
}
@keyframes draw { to { stroke-dashoffset: 0; } }
.heart-badge {
    position: absolute; top: 18px; right: 18px;
    background: rgba(0,184,169,0.18); border: 1px solid rgba(111,227,214,0.4);
    color: #fff; font-size: 11.5px; font-weight: 700;
    padding: 6px 12px; border-radius: 100px;
}

/* ---------- Section shells ---------- */
.section { padding: 64px 0 8px 0; }
.section-tight { padding: 28px 0 8px 0; }
.section-label {
    color: var(--teal); font-weight: 800; font-size: 12.5px; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 8px;
}
.section-title { color: var(--deep-blue); font-size: 30px; font-weight: 800; margin-bottom: 10px; letter-spacing: -0.01em; }
.section-desc { color: var(--ink-soft); font-size: 15px; line-height: 1.7; max-width: 680px; margin-bottom: 30px; }
.divider { height: 1px; background: var(--line); margin: 48px 0; border: none; }

/* ---------- Progress steps ---------- */
.steps-row { display: flex; gap: 8px; margin-bottom: 36px; }
.step-pill {
    flex: 1; height: 6px; border-radius: 6px; background: var(--line);
    position: relative; overflow: hidden;
}
.step-pill.done { background: linear-gradient(90deg, var(--teal), var(--deep-blue)); }
.step-caption { display: flex; justify-content: space-between; font-size: 12px; color: var(--ink-soft); font-weight: 600; margin-top: 8px; margin-bottom: 28px; }

/* ---------- Form group cards ---------- */
.group-card {
    background: rgba(255,255,255,0.7);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 26px 26px 8px 26px;
    margin-bottom: 22px;
    box-shadow: var(--shadow-sm);
    backdrop-filter: blur(8px);
    transition: box-shadow .25s, transform .25s;
}
.group-card:hover { box-shadow: var(--shadow-md); }
.group-head { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.group-ic {
    width: 36px; height: 36px; border-radius: 10px;
    background: var(--teal-light); color: var(--teal);
    display: flex; align-items: center; justify-content: center; font-size: 17px;
    flex-shrink: 0;
}
.group-title { font-weight: 800; font-size: 16.5px; color: var(--deep-blue); }
.group-sub { font-size: 12.5px; color: var(--ink-soft); margin-bottom: 16px; margin-left: 46px; }

/* Streamlit widget restyle */
div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    font-weight: 700 !important; font-size: 13.5px !important; color: var(--ink) !important;
}
div[data-baseweb="select"] > div {
    border-radius: 10px !important; border-color: var(--line) !important;
}
div[data-testid="stNumberInput"] input {
    border-radius: 10px !important;
}
.stSlider [data-baseweb="slider"] { margin-top: 6px; }

/* Predict button */
div.stButton > button {
    background: linear-gradient(135deg, var(--deep-blue), var(--teal));
    color: #fff; font-weight: 800; font-size: 15.5px;
    border: none; border-radius: 12px; padding: 14px 0;
    width: 100%;
    box-shadow: var(--shadow-md);
    transition: transform .2s, box-shadow .2s;
    letter-spacing: 0.01em;
}
div.stButton > button:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); color: #fff; }
div.stButton > button:active { transform: translateY(0); }

/* ---------- Result cards ---------- */
.result-card {
    border-radius: 20px; padding: 32px; margin-top: 18px;
    display: flex; gap: 20px; align-items: flex-start;
    box-shadow: var(--shadow-md);
    animation: fadeUp .5s ease;
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
.result-card.high { background: var(--danger-soft); border: 1px solid #F4C9CC; }
.result-card.low { background: var(--success-soft); border: 1px solid #BFE9DC; }
.result-ic { font-size: 38px; line-height: 1; }
.result-title { font-size: 21px; font-weight: 800; margin-bottom: 4px; }
.result-card.high .result-title { color: var(--danger); }
.result-card.low .result-title { color: var(--success); }
.result-text { color: var(--ink-soft); font-size: 14.5px; line-height: 1.7; margin-bottom: 14px; }
.result-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
.result-list li { font-size: 14px; color: var(--ink); padding-left: 22px; position: relative; }
.result-list li:before { content: "→"; position: absolute; left: 0; color: var(--teal); font-weight: 800; }

.kpi-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 18px; }
.kpi-card { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; box-shadow: var(--shadow-sm); }
.kpi-label { font-size: 11.5px; color: var(--ink-soft); font-weight: 700; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }
.kpi-value { font-size: 20px; font-weight: 800; color: var(--deep-blue); }

/* ---------- Education cards ---------- */
.edu-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.edu-card {
    background: var(--surface-soft); border: 1px solid var(--line); border-radius: 16px;
    padding: 22px; transition: transform .25s, box-shadow .25s;
}
.edu-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-md); }
.edu-ic { font-size: 24px; margin-bottom: 10px; }
.edu-title { font-weight: 800; font-size: 15px; color: var(--deep-blue); margin-bottom: 6px; }
.edu-text { font-size: 13.5px; color: var(--ink-soft); line-height: 1.65; }

/* How it works timeline */
.flow-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.flow-card { position: relative; background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 20px 18px; }
.flow-num { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--teal); font-weight: 700; margin-bottom: 10px; }
.flow-title { font-weight: 800; font-size: 14.5px; color: var(--deep-blue); margin-bottom: 6px; }
.flow-text { font-size: 13px; color: var(--ink-soft); line-height: 1.6; }
.flow-arrow { display:none; }

/* Prevention */
.prevent-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
.prevent-card { text-align: center; background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 18px 10px; transition: transform .25s; }
.prevent-card:hover { transform: translateY(-4px); border-color: var(--teal); }
.prevent-ic { font-size: 26px; margin-bottom: 8px; }
.prevent-title { font-weight: 700; font-size: 12.5px; color: var(--ink); }

/* Footer */
.site-footer {
    margin: 64px -100px 0 -100px; padding: 40px 100px 26px 100px;
    background: var(--deep-blue-dark); color: #BFD8EC;
}
.footer-grid { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 32px; margin-bottom: 24px; }
.footer-title { color: #fff; font-weight: 800; font-size: 16px; margin-bottom: 10px; }
.footer-text { font-size: 13px; line-height: 1.7; color: #9FB9CE; max-width: 380px; }
.footer-head { color: #fff; font-weight: 700; font-size: 13px; margin-bottom: 10px; }
.footer-link { display: block; font-size: 13px; color: #9FB9CE; margin-bottom: 7px; }
.footer-bottom { border-top: 1px solid rgba(255,255,255,0.12); padding-top: 18px; font-size: 12px; color: #7E9BB3; display: flex; justify-content: space-between; }
.disclaimer-box {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px; padding: 12px 16px; font-size: 12px; color: #9FB9CE; line-height: 1.6; margin-bottom: 18px;
}

@media (max-width: 900px) {
    .hero-grid, .edu-grid, .flow-row, .prevent-grid, .kpi-row, .feature-row, .footer-grid { grid-template-columns: 1fr 1fr; }
    .nav-links { display: none; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# NAVBAR
# ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="navbar">
    <div class="nav-logo"><span class="pulse-dot"></span> CardioAI</div>
    <div class="nav-links">
        <a href="#top">Home</a>
        <a href="#assessment">Assessment</a>
        <a href="#about">About</a>
        <a href="#faq">FAQ</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────

st.markdown('<div id="top"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero-wrap">
  <div class="hero-grid">
    <div>
      <div class="eyebrow">🫀 AI Clinical Decision Support</div>
      <div class="hero-title">Heart Stroke<br><span>Prediction</span></div>
      <div class="hero-subtitle">AI-Powered Heart Disease Risk Assessment System</div>
      <div class="hero-desc">
        This platform uses a trained machine learning model to estimate cardiovascular risk
        from everyday clinical metrics — age, blood pressure, cholesterol, ECG readings, and more.
        In under a minute, it turns eleven simple inputs into a clear, explainable risk signal that
        can support — never replace — a real conversation with your doctor.
      </div>
      <div class="feature-row">
        <div class="feature-card"><div class="ic">🤖</div><div class="ft">AI Prediction</div><div class="fd">KNN clinical model</div></div>
        <div class="feature-card"><div class="ic">⚡</div><div class="ft">Instant Results</div><div class="fd">Under 3 seconds</div></div>
        <div class="feature-card"><div class="ic">🔒</div><div class="ft">Secure Assessment</div><div class="fd">Nothing is stored</div></div>
        <div class="feature-card"><div class="ic">📊</div><div class="ft">Healthcare Insights</div><div class="fd">Actionable guidance</div></div>
      </div>
    </div>
    <div class="ecg-wrap">
      <div class="heart-badge">● LIVE MONITOR</div>
      <svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
        <polyline class="ecg-line" points="0,100 40,100 55,100 65,60 75,150 85,20 95,100 110,100 160,100 175,90 190,110 205,100 260,100 275,100 290,60 300,150 310,20 320,100 335,100 400,100" />
      </svg>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# ASSESSMENT FORM
# ──────────────────────────────────────────────────────────────────────────

st.markdown('<div id="assessment"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section">
    <div class="section-label">Patient Assessment</div>
    <div class="section-title">Tell us about your health</div>
    <div class="section-desc">
        Fill in the fields below as accurately as you can. All eleven values feed directly into the
        prediction model — there's no extra step, no account, and nothing is saved after you close this tab.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="steps-row">
    <div class="step-pill done"></div>
    <div class="step-pill done"></div>
    <div class="step-pill done"></div>
    <div class="step-pill done"></div>
</div>
<div class="step-caption">
    <span>① Welcome</span><span>② Personal Info</span><span>③ Cardiovascular Metrics</span><span>④ Diagnostic Info</span>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# WELCOME — collects the patient's name and greets them by name
# ──────────────────────────────────────────────────────────────────────────

def render_welcome() -> str:
    """Render the welcome card and return the patient's name (str)."""
    st.markdown("""
    <div class="group-card">
        <div class="group-head"><div class="group-ic">👋</div><div class="group-title">Welcome</div></div>
        <div class="group-sub">Let's start with your name so we can personalize your assessment</div>
    """, unsafe_allow_html=True)
    name = st.text_input(
        "Your Name",
        value="",
        placeholder="e.g. John Doe",
        help="Used only to personalize your results on this screen — it is not stored or sent anywhere.",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return name


def get_patient_name(raw_name: str, fallback: str = "Patient") -> str:
    """Return a clean display name, trimmed and title-cased, with a safe fallback."""
    cleaned = (raw_name or "").strip()
    if not cleaned:
        return fallback
    return cleaned.title()


patient_name_input = render_welcome()
patient_name = get_patient_name(patient_name_input)

if patient_name_input.strip():
    st.markdown(f"""
    <div style="background:var(--teal-light); border:1px solid #BFEDE8; border-radius:12px;
                padding:12px 18px; margin-bottom:22px; font-size:14px; color:var(--deep-blue); font-weight:700;">
        👋 Welcome, {patient_name}! Let's assess your heart health below.
    </div>
    """, unsafe_allow_html=True)

# ---- Group 1: Personal Information ----
st.markdown("""
<div class="group-card">
    <div class="group-head"><div class="group-ic">👤</div><div class="group-title">Personal Information</div></div>
    <div class="group-sub">Basic demographic details used as baseline risk factors</div>
""", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 18, 100, 40, help="Patient's age in years. Risk generally increases with age.")
with c2:
    sex = st.selectbox("Sex", ["M", "F"], help="Biological sex as recorded in clinical data.")
st.markdown("</div>", unsafe_allow_html=True)

# ---- Group 2: Cardiovascular Metrics ----
st.markdown("""
<div class="group-card">
    <div class="group-head"><div class="group-ic">❤️</div><div class="group-title">Cardiovascular Metrics</div></div>
    <div class="group-sub">Core measurements of heart and circulatory function</div>
""", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Blood pressure measured at rest, in mm Hg. Normal range is roughly 90–120.")
with c2:
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200, help="Serum cholesterol level. Above 200 mg/dL is considered borderline-high.")
with c3:
    max_hr = st.slider("Max Heart Rate", 60, 220, 150, help="Highest heart rate achieved, typically during exercise testing.")
st.markdown("</div>", unsafe_allow_html=True)

# ---- Group 3: Diagnostic Information ----
st.markdown("""
<div class="group-card">
    <div class="group-head"><div class="group-ic">🩺</div><div class="group-title">Diagnostic Information</div></div>
    <div class="group-sub">Clinical test results that refine the risk signal</div>
""", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], help="ATA: Atypical Angina · NAP: Non-Anginal Pain · TA: Typical Angina · ASY: Asymptomatic")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], help="Resting electrocardiogram result category.")
with c2:
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"], help="Whether chest pain occurs during physical exertion.")
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], help="Slope of the ST segment during peak exercise.")
with c3:
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, help="ST depression induced by exercise relative to rest.")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], help="1 if fasting blood sugar exceeds 120 mg/dL, otherwise 0.")
st.markdown("</div>", unsafe_allow_html=True)

predict_clicked = st.button("Run AI Risk Assessment →")

# ──────────────────────────────────────────────────────────────────────────
# PREDICTION — BACKEND LOGIC UNCHANGED
# ──────────────────────────────────────────────────────────────────────────

if predict_clicked:
    progress_box = st.empty()
    stages = [
        ("🔎", "Validating patient inputs..."),
        ("🧬", "Running KNN clinical model..."),
        ("📈", "Scoring cardiovascular risk..."),
        ("✅", "Finalizing assessment..."),
    ]
    for i, (icon, label) in enumerate(stages):
        pct = int(((i + 1) / len(stages)) * 100)
        progress_box.markdown(f"""
        <div class="group-card" style="text-align:center; padding:34px;">
            <div style="font-size:30px; margin-bottom:10px;">{icon}</div>
            <div style="font-weight:700; color:var(--deep-blue); margin-bottom:14px;">{label}</div>
            <div style="background:var(--line); border-radius:8px; height:8px; overflow:hidden;">
                <div style="background:linear-gradient(90deg, var(--teal), var(--deep-blue)); width:{pct}%; height:100%; border-radius:8px; transition: width .3s;"></div>
            </div>
            <div style="font-size:12px; color:var(--ink-soft); margin-top:8px;">{pct}%</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.35)
    progress_box.empty()

    # ---- UNCHANGED PREDICTION LOGIC ----
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    # ---- END UNCHANGED PREDICTION LOGIC ----

    if prediction == 1:
        st.markdown(f"""
        <div class="result-card high">
            <div class="result-ic">⚠️</div>
            <div>
                <div class="result-title">{patient_name}, your risk is: High Risk of Heart Disease</div>
                <div class="result-text">
                    The model flags this profile as higher-risk based on the values provided.
                    This is a screening signal, not a diagnosis — please share these results with a doctor.
                </div>
                <ul class="result-list">
                    <li>Book a consultation with a cardiologist soon</li>
                    <li>Track blood pressure and cholesterol regularly</li>
                    <li>Avoid strenuous activity until cleared by a doctor</li>
                    <li>Watch for chest pain, shortness of breath, or dizziness</li>
                </ul>
            </div>
        </div>
        <div class="kpi-row">
            <div class="kpi-card"><div class="kpi-label">Resting BP</div><div class="kpi-value">{resting_bp} mmHg</div></div>
            <div class="kpi-card"><div class="kpi-label">Cholesterol</div><div class="kpi-value">{cholesterol} mg/dL</div></div>
            <div class="kpi-card"><div class="kpi-label">Max Heart Rate</div><div class="kpi-value">{max_hr} bpm</div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card low">
            <div class="result-ic">✅</div>
            <div>
                <div class="result-title">{patient_name}, your risk is: Low Risk of Heart Disease</div>
                <div class="result-text">
                    The model does not flag elevated risk for this profile. Keep up the healthy habits —
                    prevention is always easier than treatment.
                </div>
                <ul class="result-list">
                    <li>Maintain a balanced, low-sodium diet</li>
                    <li>Keep up regular cardio exercise</li>
                    <li>Recheck blood pressure and cholesterol yearly</li>
                    <li>Get a routine checkup even when feeling well</li>
                </ul>
            </div>
        </div>
        <div class="kpi-row">
            <div class="kpi-card"><div class="kpi-label">Resting BP</div><div class="kpi-value">{resting_bp} mmHg</div></div>
            <div class="kpi-card"><div class="kpi-label">Cholesterol</div><div class="kpi-value">{cholesterol} mg/dL</div></div>
            <div class="kpi-card"><div class="kpi-label">Max Heart Rate</div><div class="kpi-value">{max_hr} bpm</div></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# ABOUT HEART DISEASE
# ──────────────────────────────────────────────────────────────────────────

st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-tight">
    <div class="section-label">Education</div>
    <div class="section-title">About Heart Disease</div>
    <div class="section-desc">
        Heart disease covers a range of conditions that affect how the heart pumps and how blood
        flows through the body. It remains one of the leading causes of death worldwide — but it's
        also one of the most preventable, when risk factors are caught early.
    </div>
</div>
<div class="edu-grid">
    <div class="edu-card">
        <div class="edu-ic">🫀</div>
        <div class="edu-title">What happens</div>
        <div class="edu-text">Narrowed or blocked arteries reduce blood flow to the heart muscle, which can lead to chest pain, heart attack, or stroke over time.</div>
    </div>
    <div class="edu-card">
        <div class="edu-ic">📉</div>
        <div class="edu-title">Common risk factors</div>
        <div class="edu-text">High blood pressure, high cholesterol, smoking, diabetes, obesity, and family history all raise the likelihood of cardiovascular disease.</div>
    </div>
    <div class="edu-card">
        <div class="edu-ic">🩻</div>
        <div class="edu-title">How it's measured</div>
        <div class="edu-text">Doctors combine vitals like resting blood pressure and max heart rate with ECG readings and exercise response to assess heart health.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# WHY EARLY DETECTION MATTERS
# ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-tight">
    <div class="section-label">Prevention</div>
    <div class="section-title">Why Early Detection Matters</div>
    <div class="section-desc">
        Cardiovascular disease often develops silently for years before symptoms appear. Catching
        warning signs early gives you and your doctor far more options.
    </div>
</div>
<div class="edu-grid">
    <div class="edu-card">
        <div class="edu-ic">⏱️</div>
        <div class="edu-title">More time to act</div>
        <div class="edu-text">Lifestyle changes and medication are far more effective before damage to the heart and arteries becomes severe.</div>
    </div>
    <div class="edu-card">
        <div class="edu-ic">💸</div>
        <div class="edu-title">Lower long-term cost</div>
        <div class="edu-text">Preventive care and monitoring are consistently cheaper than emergency treatment or long-term cardiac care.</div>
    </div>
    <div class="edu-card">
        <div class="edu-ic">🧘</div>
        <div class="edu-title">Peace of mind</div>
        <div class="edu-text">Knowing where you stand removes uncertainty and helps you make confident decisions about your health.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# HOW OUR AI WORKS
# ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-tight">
    <div class="section-label">Under the hood</div>
    <div class="section-title">How Our AI Works</div>
    <div class="section-desc">
        The assessment runs on a K-Nearest Neighbors model trained on clinical heart disease data.
        Here's what happens between clicking "Run AI Risk Assessment" and seeing your result.
    </div>
</div>
<div class="flow-row">
    <div class="flow-card"><div class="flow-num">STEP 1</div><div class="flow-title">Collect inputs</div><div class="flow-text">Your eleven form values are gathered into a single patient record.</div></div>
    <div class="flow-card"><div class="flow-num">STEP 2</div><div class="flow-title">Encode &amp; align</div><div class="flow-text">Categorical fields are one-hot encoded and matched to the model's expected columns.</div></div>
    <div class="flow-card"><div class="flow-num">STEP 3</div><div class="flow-title">Scale features</div><div class="flow-text">Values are normalized using the same scaler used during model training.</div></div>
    <div class="flow-card"><div class="flow-num">STEP 4</div><div class="flow-title">Predict</div><div class="flow-text">The KNN model compares your profile to similar patients to estimate risk.</div></div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# PREVENTION TIPS
# ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-tight">
    <div class="section-label">Stay ahead</div>
    <div class="section-title">Prevention Tips</div>
</div>
<div class="prevent-grid">
    <div class="prevent-card"><div class="prevent-ic">🏃</div><div class="prevent-title">Exercise</div></div>
    <div class="prevent-card"><div class="prevent-ic">🥗</div><div class="prevent-title">Healthy Diet</div></div>
    <div class="prevent-card"><div class="prevent-ic">🩸</div><div class="prevent-title">BP Monitoring</div></div>
    <div class="prevent-card"><div class="prevent-ic">🧪</div><div class="prevent-title">Cholesterol Mgmt</div></div>
    <div class="prevent-card"><div class="prevent-ic">📅</div><div class="prevent-title">Regular Checkups</div></div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# FAQ
# ──────────────────────────────────────────────────────────────────────────

st.markdown('<div id="faq"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-tight">
    <div class="section-label">Questions</div>
    <div class="section-title">Frequently Asked Questions</div>
</div>
""", unsafe_allow_html=True)

faqs = [
    ("Is this a medical diagnosis?", "No. This tool gives a screening-level risk estimate based on a machine learning model. It is not a substitute for a doctor's evaluation, diagnosis, or treatment plan."),
    ("What data does the model use?", "Eleven inputs: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, oldpeak, and ST slope."),
    ("Is my data stored anywhere?", "No. Your inputs are used only to generate the prediction in your current session and are not saved or sent anywhere afterward."),
    ("How accurate is the prediction?", "Accuracy depends on the quality and representativeness of the training data. Treat the result as a helpful signal, not a certainty — always confirm with a healthcare professional."),
    ("What should I do if I get a high-risk result?", "Don't panic. Share the result with a doctor, who can run further tests and give you a proper clinical assessment."),
]

for q, a in faqs:
    with st.expander(q):
        st.markdown(f'<div style="color:var(--ink-soft); font-size:14px; line-height:1.7;">{a}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="site-footer">
    <div class="footer-grid">
        <div>
            <div class="footer-title">🫀 Heart Stroke Prediction</div>
            <div class="footer-text">An AI-powered cardiovascular risk screening tool built to make early
            heart health awareness more accessible. Built with a KNN model trained on clinical heart
            disease data.</div>
        </div>
        <div>
            <div class="footer-head">Navigate</div>
            <a class="footer-link" href="#top">Home</a>
            <a class="footer-link" href="#assessment">Assessment</a>
            <a class="footer-link" href="#about">About</a>
            <a class="footer-link" href="#faq">FAQ</a>
        </div>
        <div>
            <div class="footer-head">Contact</div>
            <span class="footer-link">AI-powered risk screening</span>
            <span class="footer-link">For educational & demo use</span>
        </div>
    </div>
    <div class="disclaimer-box">
        ⚠️ <strong>Disclaimer:</strong> This application is an educational demo and does not provide
        medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with
        any questions about a medical condition.
    </div>
    <div class="footer-bottom">
        <span>© 2026 Heart Stroke Prediction</span>
        <span>Built with Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)