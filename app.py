"""
RavenStack — Customer Churn Prediction
Streamlit App v2 — Business Edition
Tema: Glassmorphism + Bento Grid + Business Icons + Dual Input
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="RavenStack Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=Montserrat:ital,wght@1,600&display=swap');

:root {
    --bg          : #07080f;
    --bg-card     : rgba(255,255,255,0.035);
    --bg-hover    : rgba(255,255,255,0.07);
    --border      : rgba(255,255,255,0.08);
    --border-glow : rgba(99,102,241,0.45);
    --indigo      : #6366f1;
    --indigo-lt   : #a5b4fc;
    --violet      : #8b5cf6;
    --cyan        : #22d3ee;
    --rose        : #f43f5e;
    --emerald     : #10b981;
    --amber       : #f59e0b;
    --text        : #f1f5f9;
    --muted       : #64748b;
    --muted-lt    : #94a3b8;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

/* Ambient glow background */
[data-testid="stAppViewContainer"]::before {
    content:'';
    position:fixed; top:-30%; left:-15%;
    width:700px; height:700px;
    background: radial-gradient(ellipse, rgba(99,102,241,0.09) 0%, transparent 65%);
    pointer-events:none; z-index:0;
}
[data-testid="stAppViewContainer"]::after {
    content:'';
    position:fixed; bottom:-20%; right:-10%;
    width:500px; height:500px;
    background: radial-gradient(ellipse, rgba(34,211,238,0.07) 0%, transparent 65%);
    pointer-events:none; z-index:0;
}

[data-testid="stHeader"]    { background: transparent !important; }
[data-testid="stSidebar"]   { display:none; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 1.75rem 2.5rem !important; max-width:1380px; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif; }
hr { border:none!important; border-top:1px solid var(--border)!important; margin:1.75rem 0!important; }

/* ── GLASS CARD ── */
.gcard {
    background: var(--bg-card);
    backdrop-filter: blur(18px);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem;
    transition: all .28s cubic-bezier(.4,0,.2,1);
    position: relative; overflow: hidden;
}
.gcard::after {
    content:'';
    position:absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg,transparent,rgba(99,102,241,.4),transparent);
    opacity:0; transition: opacity .3s;
}
.gcard:hover { background:var(--bg-hover); border-color:var(--border-glow);
               transform:translateY(-2px); box-shadow:0 10px 30px rgba(99,102,241,.12); }
.gcard:hover::after { opacity:1; }
.gcard:active { transform:scale(.98); box-shadow:0 2px 8px rgba(99,102,241,.15); }

/* ── HERO ── */
.hero { text-align:center; padding:2.5rem 1rem 1.5rem; }
.hero-eyebrow {
    ...
    font-size: .85rem;        /* ← ubah dari .72rem */
    padding: .4rem 1.1rem;   /* ← ubah dari .3rem .9rem */
}
.hero-eyebrow {
    ...
    background: rgba(99,102,241,.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,.25);
    box-shadow: 0 4px 15px rgba(99,102,241,.1), inset 0 1px 0 rgba(255,255,255,.08);
}
.hero-title {
    font-family:'Space Grotesk',sans-serif;
    font-size:clamp(2rem,4.5vw,3.2rem); font-weight:700; line-height:1.1;
    background: linear-gradient(130deg, #f1f5f9 30%, #a5b4fc 70%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: #f1f5f9;
    margin:0 0 .9rem;

}
.hero-sub {
    font-family: 'Montserrat', sans-serif;
    font-style: italic;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--muted-lt);
    font-size: .78rem;
    white-space: nowrap;
    margin: 0 auto 1.5rem;
    line-height: 1.65;
    letter-spacing: .03em;
}

/* ── BENTO METRIC ── */
.bento { display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin:1.25rem 0; }
.bcard {
    background:var(--bg-card); backdrop-filter:blur(18px);
    border:1px solid var(--border); border-radius:16px;
    padding:1.2rem 1.4rem; transition:all .25s ease;
    position:relative; overflow:hidden;
}
.bcard::before {
    content:''; position:absolute; inset:0;
    background:linear-gradient(135deg,rgba(99,102,241,.06),transparent);
    opacity:0; transition:opacity .3s;
}
.bcard:hover { border-color:rgba(99,102,241,.35); transform:translateY(-3px);
               box-shadow:0 14px 28px rgba(0,0,0,.35); }
.bcard:hover::before { opacity:1; }
.bcard:active { transform:scale(.97); }
.bcard-icon { font-size:1.4rem; margin-bottom:.6rem; }
.bcard-label { font-size:.68rem; text-transform:uppercase;
               letter-spacing:.09em; color:var(--muted); margin-bottom:.4rem; }
.bcard-value { font-family:'Space Grotesk',sans-serif;
               font-size:1.75rem; font-weight:700; line-height:1; }
.bcard-sub { font-size:.7rem; color:var(--muted); margin-top:.3rem; }

/* ── SECTION LABEL ── */
.slabel {
    font-size:.68rem; text-transform:uppercase; letter-spacing:.12em;
    color:var(--indigo-lt); margin-bottom:.9rem;
    display:flex; align-items:center; gap:.6rem;
}
.slabel::after { content:''; flex:1; height:1px;
                 background:linear-gradient(90deg,rgba(99,102,241,.3),transparent); }

/* ── RESULT CARDS ── */
.res-critical {
    background:linear-gradient(135deg,rgba(244,63,94,.1),rgba(244,63,94,.03));
    border:1px solid rgba(244,63,94,.28); border-radius:20px;
    padding:2rem; text-align:center;
    animation: pulseRed 2.5s ease-in-out infinite;
}
.res-high {
    background:linear-gradient(135deg,rgba(245,158,11,.1),rgba(245,158,11,.03));
    border:1px solid rgba(245,158,11,.28); border-radius:20px; padding:2rem; text-align:center;
}
.res-medium {
    background:linear-gradient(135deg,rgba(34,211,238,.09),rgba(34,211,238,.02));
    border:1px solid rgba(34,211,238,.22); border-radius:20px; padding:2rem; text-align:center;
}
.res-low {
    background:linear-gradient(135deg,rgba(16,185,129,.1),rgba(16,185,129,.03));
    border:1px solid rgba(16,185,129,.28); border-radius:20px; padding:2rem; text-align:center;
}
@keyframes pulseRed {
    0%,100% { box-shadow:0 0 0 0 rgba(244,63,94,0); }
    50%      { box-shadow:0 0 0 10px rgba(244,63,94,.08); }
}
.prob-num {
    font-family:'Space Grotesk',sans-serif;
    font-size:3.8rem; font-weight:700; line-height:1; margin:.6rem 0 .4rem;
}
.pbar-wrap { background:rgba(255,255,255,.06); border-radius:100px; height:7px;
             margin:.65rem 0; overflow:hidden; }
.pbar-fill { height:100%; border-radius:100px;
             transition:width .9s cubic-bezier(.4,0,.2,1); }

/* ── ACTION BOX ── */
.abox {
    background:rgba(99,102,241,.07); border:1px solid rgba(99,102,241,.18);
    border-radius:13px; padding:1rem 1.2rem; margin-top:.9rem;
    font-size:.88rem; line-height:1.55;
}

/* ── SIGNAL ROW ── */
.sig-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:.45rem 0; border-bottom:1px solid rgba(255,255,255,.05);
    font-size:.83rem;
}
.sig-row:last-child { border-bottom:none; }
.sig-icon { font-size:.9rem; margin-right:.4rem; }
.sig-name { color:var(--muted-lt); }
.sig-val  { font-weight:600; }

/* ── BUTTON ── */
.stButton>button {
    width:100%; background:linear-gradient(135deg,var(--indigo),#4f46e5)!important;
    color:#fff!important; border:none!important; border-radius:13px!important;
    padding:.78rem 2rem!important; font-family:'Space Grotesk',sans-serif!important;
    font-size:1rem!important; font-weight:600!important; letter-spacing:.02em!important;
    transition:all .2s ease!important; cursor:pointer!important;
    box-shadow:0 4px 15px rgba(99,102,241,.25)!important;
}
.stButton>button:hover {
    transform:translateY(-2px)!important;
    box-shadow:0 8px 25px rgba(99,102,241,.4)!important;
}
.stButton>button:active {
    transform:scale(.97) translateY(0)!important;
    box-shadow:0 2px 8px rgba(99,102,241,.25)!important;
}

/* ── INPUT OVERRIDES ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background:rgba(255,255,255,.05)!important;
    border:1px solid var(--border)!important;
    border-radius:10px!important; color:var(--text)!important;
    font-size:.9rem!important;
}
[data-testid="stNumberInput"] input:focus {
    border-color:var(--indigo)!important;
    box-shadow:0 0 0 3px rgba(99,102,241,.15)!important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background:var(--indigo)!important; border-color:var(--indigo-lt)!important;
}
label, [data-testid="stWidgetLabel"] p {
    color:var(--muted-lt)!important; font-size:.82rem!important;
}
.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,.04)!important;
                                     border-radius:10px!important; padding:.2rem!important; }
.stTabs [data-baseweb="tab"] { border-radius:8px!important; color:var(--muted-lt)!important; }
.stTabs [aria-selected="true"] { background:rgba(99,102,241,.2)!important;
                                  color:var(--indigo-lt)!important; }

/* ── TOOLTIP / INFO ── */
.info-pill {
    display:inline-block; background:rgba(99,102,241,.1);
    border:1px solid rgba(99,102,241,.2); border-radius:6px;
    padding:.1rem .5rem; font-size:.7rem; color:var(--indigo-lt);
    margin-left:.4rem; vertical-align:middle;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_assets():
    from sklearn.model_selection import train_test_split

    model  = joblib.load("models/random_forest_final.pkl")
    meta   = json.load(open("models/random_forest_metadata.json"))
    df_ref = pd.read_csv("data/processed/ravenstack_features_for_modeling.csv")
    bc     = df_ref.select_dtypes(include='bool').columns
    df_ref[bc] = df_ref[bc].astype(int)
    feats  = [c for c in df_ref.columns if c != 'target']

    X_full = df_ref[feats]
    y_full = df_ref['target']

    # Reproduksi SPLIT YANG SAMA PERSIS dengan training (lihat notebook 02, SEED=42)
    # supaya kita tahu persis baris mana yang TIDAK PERNAH dilihat model.
    _, X_test, _, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    threshold  = float(meta.get("threshold", 0.32))
    proba_test = model.predict_proba(X_test)[:, 1]

    test_df = X_test.copy()
    test_df['_proba']  = proba_test
    test_df['_target'] = y_test.values

    real_examples = {}
    churned_test = test_df[test_df['_target'] == 1]
    safe_test    = test_df[test_df['_target'] == 0]

    if len(churned_test) > 0:
        real_examples['high']   = test_df.loc[churned_test['_proba'].idxmax()]
    if len(safe_test) > 0:
        real_examples['low']    = test_df.loc[safe_test['_proba'].idxmin()]
    if len(test_df) > 0:
        real_examples['border'] = test_df.loc[(test_df['_proba'] - threshold).abs().idxmin()]

    return model, meta, df_ref, feats, real_examples

model, meta, df_ref, FEATURES, REAL_EXAMPLES = load_assets()
THRESHOLD = round(float(meta.get("threshold", 0.32)), 2)
KURS = 17500  # Kurs referensi USD → IDR (per Juli 2026)
def fmt_idr(n):
    return f"Rp {n:,.0f}".replace(",", ".")
X_ref     = df_ref[FEATURES]
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = {}
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None
if 'selected_real_row' not in st.session_state:
    st.session_state.selected_real_row = None


# ============================================================
# HELPERS
# ============================================================
def get_risk(p):
    if p >= .70: return "Critical Risk", "#f43f5e", "🔴", "res-critical", "📛"
    if p >= .45: return "High Risk",     "#f59e0b", "🟠", "res-high",     "⚠️"
    if p >= .32: return "Medium Risk",   "#22d3ee", "🟡", "res-medium",   "📋"
    return             "Low Risk",       "#10b981", "🟢", "res-low",      "✅"

def get_action(risk):
    return {
        "Critical Risk": "🚨 Eskalasi ke Account Manager dalam 24 jam. Jadwalkan retention call segera dan siapkan paket penawaran khusus.",
        "High Risk"    : "📞 CS outreach personal dalam 48 jam. Tawarkan diskon retensi 20–30% atau upgrade fitur gratis.",
        "Medium Risk"  : "📧 Kirim email engagement + highlight fitur yang belum dipakai. Follow up dalam 1 minggu.",
        "Low Risk"     : "📊 Monitor bulanan. Tidak perlu intervensi aktif saat ini.",
    }.get(risk, "-")

def num_input_with_slider(label, min_v, max_v, default, step=1, fmt="%d", key=None, help=None, is_float=False):
    fmt_str = "%.2f" if is_float else "%.0f"
    val = st.number_input(label, min_value=float(min_v), max_value=float(max_v),
                          value=float(default), step=float(step),
                          key=f"n_{key}", help=help,
                          label_visibility="visible", format=fmt_str)
    if is_float:
        st.slider("", min_value=float(min_v), max_value=float(max_v),
                  value=float(val), step=float(step),
                  key=f"s_{key}", label_visibility="collapsed", format="%.2f")
    else:
        st.slider("", min_value=int(min_v), max_value=int(max_v),
                  value=int(val), step=int(step),
                  key=f"s_{key}", label_visibility="collapsed", format="%d")
    return float(val) if is_float else int(val)

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div style="position:relative">
    <div style="position:absolute;top:0;right:0;
         background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.2);
         border-radius:100px;padding:.3rem .9rem;
         font-size:.72rem;color:#67e8f9;letter-spacing:.08em">
        ⚡ Developed by Pandu Bashir Alamin
    </div>
    <div class="hero">
        <div class="hero-eyebrow">
            <span>📊</span> Business Intelligence · RavenStack AI
        </div>
        <svg width="100%" viewBox="0 0 800 120" style="overflow:visible;margin:0 0 .9rem">
        <defs>
            <linearGradient id="tg" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#f1f5f9"/>
                <stop offset="50%" style="stop-color:#a5b4fc"/>
                <stop offset="100%" style="stop-color:#22d3ee"/>
            </linearGradient>
        </defs>
        <text x="400" y="48" text-anchor="middle"
              font-family="Space Grotesk, sans-serif"
              font-size="52" font-weight="700" fill="url(#tg)">
            Customer Churn
        </text>
        <text x="400" y="108" text-anchor="middle"
              font-family="Space Grotesk, sans-serif"
              font-size="52" font-weight="700" fill="url(#tg)">
            Intelligence Platform
        </text>
    </svg>
        <p class="hero-sub">
    Identifikasi pelanggan berisiko sebelum mereka pergi. Prediksi real-time berbasis 83 sinyal bisnis &amp; perilaku.
</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# BENTO STATS
# ============================================================
churn_rate   = df_ref['target'].mean()
total_cust   = len(df_ref)
at_risk      = int((model.predict_proba(X_ref)[:,1] >= THRESHOLD).sum())
model_recall = meta.get("metrics", {}).get("recall", 0)
mrr_at_risk  = df_ref[model.predict_proba(X_ref)[:,1] >= THRESHOLD]['total_mrr'].sum() \
               if 'total_mrr' in df_ref.columns else 0
mrr_at_risk_idr = mrr_at_risk * KURS
st.markdown(f"""
<div class="bento">
    <div class="bcard">
        <div class="bcard-icon">🏢</div>
        <div class="bcard-label">Total Pelanggan</div>
        <div class="bcard-value" style="color:#a5b4fc">{total_cust:,}</div>
        <div class="bcard-sub">aktif dalam dataset</div>
    </div>
    <div class="bcard">
        <div class="bcard-icon">📉</div>
        <div class="bcard-label">Churn Rate</div>
        <div class="bcard-value" style="color:#f43f5e">{churn_rate:.0%}</div>
        <div class="bcard-sub">dari total pelanggan</div>
    </div>
    <div class="bcard">
        <div class="bcard-icon">⚠️</div>
        <div class="bcard-label">Pelanggan At Risk</div>
        <div class="bcard-value" style="color:#f59e0b">{at_risk}</div>
        <div class="bcard-sub">perlu intervensi CS</div>
    </div>
    <div class="bcard">
        <div class="bcard-icon">💰</div>
        <div class="bcard-label">MRR at Risk</div>
        <div class="bcard-value" style="color:#10b981">{fmt_idr(mrr_at_risk_idr)}</div>
        <div class="bcard-sub">potensi revenue hilang</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Row 2 bento — model info
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.9rem;margin-bottom:1.5rem">
    <div class="bcard">
        <div class="bcard-icon">🎯</div>
        <div class="bcard-label">Model</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:600;color:#a5b4fc">XGBoost Optimized</div>
        <div class="bcard-sub">RandomizedSearchCV · 50 iterasi</div>
    </div>
    <div class="bcard">
        <div class="bcard-icon">📡</div>
        <div class="bcard-label">Recall · F1 · AUC</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:600;color:#22d3ee">
            {meta.get('metrics',{}).get('recall',0):.0%} · {meta.get('metrics',{}).get('f1',0):.2f} · {meta.get('metrics',{}).get('auc_roc',0):.2f}
        </div>
        <div class="bcard-sub">pada test set (threshold {THRESHOLD})</div>
    </div>
    <div class="bcard">
        <div class="bcard-icon">🔬</div>
        <div class="bcard-label">Fitur yang dianalisis</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:600;color:#f59e0b">83 sinyal</div>
        <div class="bcard-sub">dari 5 tabel data bisnis</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# MAIN — Input + Result
# ============================================================
col_left, col_right = st.columns([1.15, 0.85], gap="large")

with col_left:
    st.markdown('<div class="slabel">📋 Input Data Pelanggan</div>', unsafe_allow_html=True)

    tab_manual, tab_real = st.tabs(
        ["✏️  Input Manual", "📁  Contoh Profil"]
    )

    with tab_manual:
        st.markdown("""
        <div style="font-size:.8rem;color:#64748b;margin-bottom:1rem">
        💡 Ketik langsung di kotak angka atau geser slider — keduanya bisa dipakai.
        </div>""", unsafe_allow_html=True)

        # ── Group 1: Akun ──
        st.markdown('<div style="font-size:.75rem;color:#6366f1;text-transform:uppercase;letter-spacing:.08em;margin:.75rem 0 .4rem">🏢 Profil Akun</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            tenure_days = num_input_with_slider("Tenure (hari)", 1, 1000, 180, 1, key="tenure",
                                                help="Lama pelanggan bergabung sejak signup")
        with c2:
            seats = num_input_with_slider("Jumlah seats", 1, 200, 10, 1, key="seats",
                                          help="Jumlah lisensi pengguna")

        # ── Group 2: Revenue ──
        st.markdown('<div style="font-size:.75rem;color:#6366f1;text-transform:uppercase;letter-spacing:.08em;margin:.75rem 0 .4rem">💰 Revenue</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            mrr_input_str = st.text_input(
                "Total MRR (Rp)",
                value="5.000.000",
                help="Ketik nominal MRR dalam Rupiah, contoh: 5.000.000",
                key="mrr_text"
            )
            try:
                total_mrr_idr = int(mrr_input_str.replace(".", "").replace(",", "").replace("Rp", "").strip())
            except:
                total_mrr_idr = 5_000_000
            total_mrr_idr = max(0, min(total_mrr_idr, 35_000_000))
            total_mrr = total_mrr_idr / KURS
            st.caption(f"≈ USD ${total_mrr:,.0f} | Range: Rp 0 – Rp 35.000.000")
        with c4:
            sub_churn_ratio = num_input_with_slider("Subscription churn ratio", 0.0, 1.0, 0.2, 0.05, key="sub_churn",is_float=True)

        # ── Group 3: Engagement ──
        st.markdown('<div style="font-size:.75rem;color:#6366f1;text-transform:uppercase;letter-spacing:.08em;margin:.75rem 0 .4rem">📱 Engagement & Penggunaan</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            days_since_last_usage = num_input_with_slider("Hari sejak terakhir login", 0, 180, 14, 1, key="recency",
                                                           help="Semakin lama = semakin berisiko")
        with c6:
            unique_features_used = num_input_with_slider("Fitur yang pernah dipakai (dari 40)", 1, 40, 8, 1, key="features",
                                                          help="Feature breadth — makin banyak makin loyal")

        c7, c8 = st.columns(2)
        with c7:
                    net_plan_movement = num_input_with_slider(
            "Net plan movement", -5, 5, 0, 1, key="plan",
            help="Positif = pernah upgrade, negatif = downgrade, 0 = tidak ada perubahan")
        with c8:
           usage_density = num_input_with_slider("Usage density (events/hari)", 0.0, 20.0, 3.0, 0.5, key="density", is_float=True)

        # ── Group 4: Support ──
        st.markdown('<div style="font-size:.75rem;color:#6366f1;text-transform:uppercase;letter-spacing:.08em;margin:.75rem 0 .4rem">🎧 Customer Support</div>', unsafe_allow_html=True)
        c9, c10 = st.columns(2)
        with c9:
            avg_satisfaction_score = num_input_with_slider("Satisfaction score (1–5)", 1.0, 5.0, 3.5, 0.1, key="sat",
                                                            help="Rata-rata skor kepuasan dari tiket support", is_float=True)
        with c10:
            total_tickets = num_input_with_slider("Total tiket support", 0, 50, 3, 1, key="tickets")

        c11, c12 = st.columns(2)
        with c11:
            escalation_rate = num_input_with_slider("Escalation rate (0–1)", 0.0, 1.0, 0.1, 0.05, key="esc", is_float=True)
                                                     
        with c12:
            pct_urgent = num_input_with_slider("% tiket urgent (0–1)", 0.0, 1.0, 0.1, 0.05, is_float=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Analisis Risiko Churn", use_container_width=True)
        if predict_btn:
            st.session_state.input_mode = "manual"

    with tab_real:
        st.markdown("""
        <div style="font-size:.8rem;color:#64748b;margin-bottom:1rem">
        Pilih profil contoh untuk melihat cara kerja model
        <span style="color:#475569">(diambil dari akun pelanggan sungguhan, bukan skenario buatan)</span>.
        </div>
        """, unsafe_allow_html=True)

        real_labels = {
            'high':   "🔴 Risiko Tertinggi",
            'border': "🟡 Perlu Dipantau",
            'low':    "🟢 Risiko Terendah",
        }
        available = [k for k in ['high', 'border', 'low'] if k in REAL_EXAMPLES]

        if not available:
            st.warning("Data contoh real tidak tersedia.")
        else:
            real_choice = st.selectbox(
                "", [real_labels[k] for k in available],
                label_visibility="collapsed", key="real_choice"
            )
            real_key = available[[real_labels[k] for k in available].index(real_choice)]
            row = REAL_EXAMPLES[real_key]

            actual = "Pernah churn" if row['_target'] == 1 else "Aktif bertahan"
            actual_color = "#94a3b8"

            st.markdown(f"""
            <div class="gcard" style="margin-top:.5rem">
                <div style="font-size:.68rem;color:#64748b;margin-bottom:.6rem;
                            display:flex;justify-content:space-between">
                    <span>Status historis: <b style="color:{actual_color}">{actual}</b></span>
                    <span>Skor model: <b style="color:#f1f5f9">{row['_proba']:.0%}</b></span>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem">
                    <div class="sig-row"><span class="sig-name">🏢 Tenure</span><span class="sig-val">{int(row['tenure_days'])} hari</span></div>
                    <div class="sig-row"><span class="sig-name">💰 MRR</span><span class="sig-val">{fmt_idr(row['total_mrr'] * KURS)}</span></div>
                    <div class="sig-row"><span class="sig-name">📱 Last login</span><span class="sig-val">{int(row['days_since_last_usage'])} hari lalu</span></div>
                    <div class="sig-row"><span class="sig-name">🔧 Fitur dipakai</span><span class="sig-val">{int(row['unique_features_used'])}/40</span></div>
                    <div class="sig-row"><span class="sig-name">⭐ Satisfaction</span><span class="sig-val">{row['avg_satisfaction_score']:.1f}/5</span></div>
                    <div class="sig-row"><span class="sig-name">🎧 Tiket</span><span class="sig-val">{int(row['total_tickets'])}</span></div>
                    <div class="sig-row"><span class="sig-name">🚨 Escalation</span><span class="sig-val">{row['escalation_rate']:.0%}</span></div>
                    <div class="sig-row"><span class="sig-name">📊 Plan movement</span><span class="sig-val">{int(row['net_plan_movement']):+d}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            real_btn = st.button("🔍 Prediksi Profil Ini", use_container_width=True, key="real_predict_btn")

            if real_btn:
                st.session_state.selected_real_row = row
                st.session_state.input_mode = "real"
                predict_btn = True


# ============================================================
# RESULT
# ============================================================
with col_right:
    st.markdown('<div class="slabel">📈 Hasil Analisis Risiko</div>', unsafe_allow_html=True)

    if predict_btn:
        if st.session_state.input_mode == "real":
            # ── Mode Data Real: pakai vektor 83 fitur ASLI, tidak direkonstruksi ──
            row  = st.session_state.selected_real_row
            X_in = pd.DataFrame([row[FEATURES]])

            # Variabel display diambil langsung dari baris asli
            tenure_days             = int(row['tenure_days'])
            total_mrr                = float(row['total_mrr'])
            days_since_last_usage    = int(row['days_since_last_usage'])
            unique_features_used     = int(row['unique_features_used'])
            avg_satisfaction_score   = float(row['avg_satisfaction_score'])
            total_tickets            = int(row['total_tickets'])
            escalation_rate          = float(row['escalation_rate'])
            sub_churn_ratio          = float(row['sub_churn_ratio'])
            net_plan_movement        = int(row['net_plan_movement'])

            ground_truth_note = (
                f'<div style="font-size:.68rem;color:#64748b;margin-bottom:.7rem;text-align:center">'
                f'📌 Status historis pelanggan ini: '
                f'<b style="color:#94a3b8">{"pernah churn" if row["_target"]==1 else "aktif bertahan"}</b>'
                f'</div>'
            )
        else:
            _ss = st.session_state.selected_sample
            ground_truth_note = ""

            # Gunakan 0 sebagai default, bukan mean dataset
            base = {col: 0 for col in FEATURES}

            base.update({
                "tenure_days"            : tenure_days,
                "total_mrr"              : total_mrr,
                "avg_mrr"                : total_mrr,
                "total_arr"              : total_mrr * 12,
                "days_since_last_usage"  : days_since_last_usage,
                "unique_features_used"   : unique_features_used,
                "feature_breadth_ratio"  : unique_features_used / 40,
                "avg_satisfaction_score" : avg_satisfaction_score,
                "min_satisfaction_score" : avg_satisfaction_score,
                "total_tickets"          : total_tickets,
                "escalation_rate"        : escalation_rate,
                "num_escalated"          : int(escalation_rate * max(total_tickets,1)),
                "sub_churn_ratio"        : sub_churn_ratio,
                "net_plan_movement"      : net_plan_movement,
                "seats"                  : seats,
                "pct_urgent"             : pct_urgent,
                "usage_density"          : usage_density,
                "error_rate"             : _ss.get("error_rate", escalation_rate * 0.5),
                "avg_error_count"        : _ss.get("avg_error_count", escalation_rate * 0.5),
                "avg_first_response_mins": _ss.get("avg_first_response_mins", 180.0 if escalation_rate > 0.5 else 30.0),
                "pct_annual_billing"     : _ss.get("pct_annual_billing", 0.9 if net_plan_movement > 0 else 0.3),
                "ever_downgraded"        : _ss.get("ever_downgraded", 1 if net_plan_movement < 0 else 0),
                "has_low_satisfaction"   : _ss.get("has_low_satisfaction", 1 if avg_satisfaction_score < 2.5 else 0),
                "churn_event_count"      : _ss.get("churn_event_count", 0),
                "total_refund_usd"       : 0,
                "days_since_last_churn"  : 0,
                "max_mrr"                  : total_mrr,
                "subscription_duration_avg": tenure_days * 0.8,
                "avg_usage_duration_secs"  : usage_density * 300,
                "total_usage_events"       : usage_density * 30,
                "total_usage_count"        : usage_density * 30,
                "usage_days"               : min(30, int(tenure_days * 0.7)),
                "total_usage_duration_secs": usage_density * 300 * 30,
                "avg_usage_count"          : usage_density,
                "total_subscriptions"      : 1,
                "active_subscriptions"     : 1,
                "churned_subscriptions"    : int(sub_churn_ratio),
                "num_upgrades"             : max(0, net_plan_movement),
                "num_downgrades"           : max(0, -net_plan_movement),
                "num_trials"               : 0,
                "has_auto_renew"           : 1,
                "net_plan_movement"        : net_plan_movement,
                "beta_feature_usage_count" : 0,
                "total_tickets"            : total_tickets,
                "num_urgent_tickets"       : int(pct_urgent * total_tickets),
                "num_high_tickets"         : int(pct_urgent * total_tickets * 0.5),
                "critical_ticket_count"    : int(pct_urgent * total_tickets),
                "min_satisfaction_score"   : avg_satisfaction_score - 0.5,
                "max_resolution_hours"     : 48.0 if escalation_rate > 0.3 else 8.0,
                "avg_resolution_hours"     : 24.0 if escalation_rate > 0.3 else 4.0,
                # One-hot defaults — set industry_DevTools sebagai default
                "industry_DevTools"        : 1,
                "referral_source_organic"  : 1,
                "recency_bucket_encoded"   : 1 if days_since_last_usage <= 7 else 2 if days_since_last_usage <= 30 else 3,
                "latest_plan_tier_encoded" : 1 if net_plan_movement >= 0 else 0,
                "seats_bucket_encoded"     : 0 if seats <= 5 else 1 if seats <= 20 else 2 if seats <= 50 else 3,
                "tenure_bucket_encoded"    : 0 if tenure_days <= 90 else 1 if tenure_days <= 180 else 2 if tenure_days <= 365 else 3,
            })

            X_in = pd.DataFrame([base]).reindex(columns=FEATURES, fill_value=0)

        bc   = X_in.select_dtypes(include='bool').columns
        X_in[bc] = X_in[bc].astype(int)

        prob  = float(model.predict_proba(X_in)[0, 1])
        risk_label, risk_color, risk_emoji, css_class, biz_icon = get_risk(prob)
        action    = get_action(risk_label)
        predicted = prob >= THRESHOLD

        st.markdown(f"""
        {ground_truth_note}
        <div class="{css_class}">
            <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;
                        color:{risk_color};margin-bottom:.4rem">{biz_icon} {risk_label}</div>
            <div class="prob-num" style="color:{risk_color}">{prob:.0%}</div>
            <div style="font-size:.82rem;color:#94a3b8;margin-bottom:.9rem">probabilitas churn</div>
            <div class="pbar-wrap">
                <div class="pbar-fill" style="width:{int(prob*100)}%;
                     background:linear-gradient(90deg,{risk_color}77,{risk_color})"></div>
            </div>
            <div style="font-size:.78rem;color:#64748b;margin-top:.5rem">
                threshold {THRESHOLD} &nbsp;·&nbsp;
                {'<span style="color:#f43f5e;font-weight:600">⚠️ DIPREDIKSI CHURN</span>'
                 if predicted else
                 '<span style="color:#10b981;font-weight:600">✅ AMAN</span>'}
            </div>
        </div>
        <div class="abox">
            <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;
                        color:#6366f1;margin-bottom:.4rem">💼 Rekomendasi Tim CS</div>
            <span style="color:#cbd5e1">{action}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Sinyal Risiko ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="slabel">🔬 Detail Sinyal Risiko</div>', unsafe_allow_html=True)

        def sig_color(val, bad_high=True, threshold=None):
            if threshold is None: return "#f1f5f9"
            if bad_high: return "#f43f5e" if val > threshold else "#10b981"
            return "#f43f5e" if val < threshold else "#10b981"

        signals = [
            ("📅", "Tenure",           f"{tenure_days} hari",                sig_color(tenure_days, bad_high=False, threshold=90)),
            ("📱", "Last login",        f"{days_since_last_usage} hari lalu", sig_color(days_since_last_usage, True, 14)),
            ("⭐", "Satisfaction",      f"{avg_satisfaction_score:.1f} / 5",  sig_color(avg_satisfaction_score, False, 3.5)),
            ("🔧", "Feature breadth",   f"{unique_features_used} / 40",       sig_color(unique_features_used, False, 10)),
            ("💰", "MRR", fmt_idr(total_mrr * KURS), "#f1f5f9"),
            ("🚨", "Escalation rate",   f"{escalation_rate:.0%}",             sig_color(escalation_rate, True, 0.2)),
            ("📊", "Plan movement",     f"{net_plan_movement:+d}",            sig_color(net_plan_movement, False, 0)),
            ("🎧", "Sub churn ratio",   f"{sub_churn_ratio:.0%}",             sig_color(sub_churn_ratio, True, 0.3)),
        ]

        rows = "".join([
            f'<div class="sig-row">'
            f'<span class="sig-name"><span class="sig-icon">{ico}</span>{name}</span>'
            f'<span class="sig-val" style="color:{color}">{val}</span>'
            f'</div>'
            for ico, name, val, color in signals
        ])

        st.markdown(f'<div class="gcard">{rows}</div>', unsafe_allow_html=True)

        # ── Business Impact ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="slabel">📊 Business Impact</div>', unsafe_allow_html=True)

        monthly_loss = total_mrr
        annual_loss  = monthly_loss * 12
        intervention_cost = total_mrr * 0.25

        st.markdown(f"""
        <div class="gcard">
            <div class="sig-row">
                <span class="sig-name">💸 Potensi MRR hilang</span>
                <span class="sig-val" style="color:#f43f5e">${monthly_loss:,.0f}/bln</span>
            </div>
            <div class="sig-row">
                <span class="sig-name">📅 ARR at risk</span>
                <span class="sig-val" style="color:#f59e0b">${annual_loss:,.0f}/thn</span>
            </div>
            <div class="sig-row">
                <span class="sig-name">🎁 Est. biaya retensi</span>
                <span class="sig-val" style="color:#10b981">${intervention_cost:,.0f}</span>
            </div>
            <div class="sig-row">
                <span class="sig-name">📈 ROI intervensi</span>
                <span class="sig-val" style="color:#a5b4fc">
                    {'∞ (perlu intervensi)' if monthly_loss > intervention_cost else 'Monitor saja'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="gcard" style="text-align:center;padding:3.5rem 2rem;min-height:320px;
             display:flex;flex-direction:column;justify-content:center;align-items:center;gap:.75rem">
            <div style="font-size:3rem">📊</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;
                        font-weight:600;color:#e2e8f0">Siap Menganalisis</div>
            <div style="color:#64748b;font-size:.85rem;line-height:1.7;max-width:260px">
                Masukkan data pelanggan atau pilih contoh profil,
                lalu klik tombol analisis untuk melihat risk score.
            </div>
            <div style="margin-top:.5rem;display:flex;gap:.5rem;flex-wrap:wrap;justify-content:center">
                <span style="background:rgba(244,63,94,.12);border:1px solid rgba(244,63,94,.2);
                      border-radius:20px;padding:.25rem .7rem;font-size:.7rem;color:#f43f5e">🔴 Critical</span>
                <span style="background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.2);
                      border-radius:20px;padding:.25rem .7rem;font-size:.7rem;color:#f59e0b">🟠 High</span>
                <span style="background:rgba(34,211,238,.10);border:1px solid rgba(34,211,238,.2);
                      border-radius:20px;padding:.25rem .7rem;font-size:.7rem;color:#22d3ee">🟡 Medium</span>
                <span style="background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.2);
                      border-radius:20px;padding:.25rem .7rem;font-size:.7rem;color:#10b981">🟢 Low</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("""
    <div style="font-size:.75rem;color:#475569">
        <strong style="color:#6366f1">🔬 Model</strong><br>
        Random Forest · Anti-Overfit (max_depth=10)
        SMOTE · Threshold 0.33
    </div>""", unsafe_allow_html=True)
with col_f2:
    st.markdown("""
    <div style="font-size:.75rem;color:#475569;text-align:center">
        <strong style="color:#6366f1">📊 Dataset</strong><br>
        RavenStack Synthetic SaaS<br>
        by <a href="https://pandubashir.my.id" target="_blank"
        style="color:#6366f1">Pandu Bashir Alamin</a>
    </div>""", unsafe_allow_html=True)
with col_f3:
    st.markdown("""
    <div style="font-size:.75rem;color:#475569;text-align:right">
        <strong style="color:#6366f1">⚡ Stack</strong><br>
        Python · Streamlit · SHAP<br>
        Portfolio Project — Data Sintetis
    </div>""", unsafe_allow_html=True)