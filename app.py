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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

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
    display:inline-flex; align-items:center; gap:.5rem;
    background:rgba(99,102,241,.12); border:1px solid rgba(99,102,241,.25);
    border-radius:100px; padding:.3rem .9rem;
    font-size:.72rem; color:var(--indigo-lt);
    letter-spacing:.1em; text-transform:uppercase; margin-bottom:1.2rem;
}
.hero-title {
    font-family:'Space Grotesk',sans-serif;
    font-size:clamp(2rem,4.5vw,3.2rem); font-weight:700; line-height:1.1;
    background:linear-gradient(130deg,#f1f5f9 25%,var(--indigo-lt) 60%,var(--cyan) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0 0 .9rem;
}
.hero-sub { color:var(--muted-lt); font-size:.95rem; max-width:500px;
            margin:0 auto 1.5rem; line-height:1.65; }

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
    model  = joblib.load("models/xgboost_optimized.pkl")
    meta   = json.load(open("models/xgboost_metadata.json"))
    df_ref = pd.read_csv("data/processed/ravenstack_features_for_modeling.csv")
    bc     = df_ref.select_dtypes(include='bool').columns
    df_ref[bc] = df_ref[bc].astype(int)
    feats  = [c for c in df_ref.columns if c != 'target']
    return model, meta, df_ref, feats

model, meta, df_ref, FEATURES = load_assets()
THRESHOLD = meta.get("threshold", 0.32)
X_ref     = df_ref[FEATURES]


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

def num_input_with_slider(label, min_v, max_v, default, step=1, fmt="%d", key=None, help=None):
    """Dual input: number box + slider"""
    col_n, col_s = st.columns([1, 2])
    with col_n:
        val_n = st.number_input(label, min_value=float(min_v), max_value=float(max_v),
                                value=float(default), step=float(step),
                                key=f"n_{key}", help=help, label_visibility="visible")
    with col_s:
        st.markdown("<div style='margin-top:1.65rem'></div>", unsafe_allow_html=True)
        val_s = st.slider("", min_value=float(min_v), max_value=float(max_v),
                          value=float(val_n), step=float(step),
                          key=f"s_{key}", label_visibility="collapsed")
    # Sync: slider mengikuti number input
    return val_n


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">
        <span>📊</span> Business Intelligence · RavenStack AI
    </div>
    <h1 class="hero-title">Customer Churn<br>Intelligence Platform</h1>
    <p class="hero-sub">
        Identifikasi pelanggan berisiko sebelum mereka pergi.
        Prediksi real-time berbasis 83 sinyal bisnis &amp; perilaku.
    </p>
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
        <div class="bcard-value" style="color:#10b981">${mrr_at_risk:,.0f}</div>
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

    tab_manual, tab_sample = st.tabs(["✏️  Input Manual", "📁  Contoh Profil"])

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
            total_mrr = num_input_with_slider("Total MRR ($)", 0, 3000, 300, 10, key="mrr",
                                              help="Monthly Recurring Revenue pelanggan ini")
        with c4:
            sub_churn_ratio = num_input_with_slider("Subscription churn ratio", 0.0, 1.0, 0.2, 0.05, key="sub_churn",
                                                     help="Proporsi subscription yang pernah churn")

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
            net_plan_movement = num_input_with_slider("Net plan movement", -5, 5, 0, 1, key="plan",
                                                       help="Positif = pernah upgrade, negatif = downgrade")
        with c8:
            usage_density = num_input_with_slider("Usage density (events/hari)", 0.0, 20.0, 3.0, 0.5, key="density",
                                                   help="Rata-rata aktivitas per hari aktif")

        # ── Group 4: Support ──
        st.markdown('<div style="font-size:.75rem;color:#6366f1;text-transform:uppercase;letter-spacing:.08em;margin:.75rem 0 .4rem">🎧 Customer Support</div>', unsafe_allow_html=True)
        c9, c10 = st.columns(2)
        with c9:
            avg_satisfaction_score = num_input_with_slider("Satisfaction score (1–5)", 1.0, 5.0, 3.5, 0.1, key="sat",
                                                            help="Rata-rata skor kepuasan dari tiket support")
        with c10:
            total_tickets = num_input_with_slider("Total tiket support", 0, 50, 3, 1, key="tickets")

        c11, c12 = st.columns(2)
        with c11:
            escalation_rate = num_input_with_slider("Escalation rate (0–1)", 0.0, 1.0, 0.1, 0.05, key="esc",
                                                     help="Proporsi tiket yang dieskalasi")
        with c12:
            pct_urgent = num_input_with_slider("% tiket urgent (0–1)", 0.0, 1.0, 0.1, 0.05, key="urgent")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Analisis Risiko Churn", use_container_width=True)

    with tab_sample:
        st.markdown("""
        <div style="font-size:.8rem;color:#64748b;margin-bottom:1rem">
        Pilih profil contoh untuk melihat cara kerja model.
        </div>""", unsafe_allow_html=True)

        sample_choice = st.selectbox("", [
            "🔴 Pelanggan Berisiko Tinggi — startup baru, banyak komplain",
            "🟢 Pelanggan Sehat — enterprise, loyal & aktif",
            "🟡 Pelanggan Borderline — perlu dipantau",
        ], label_visibility="collapsed")

        SAMPLES = {
            "🔴 Pelanggan Berisiko Tinggi — startup baru, banyak komplain": {
                "tenure_days":45,"total_mrr":50,"days_since_last_usage":35,
                "unique_features_used":2,"avg_satisfaction_score":1.5,
                "total_tickets":12,"escalation_rate":0.75,"sub_churn_ratio":0.6,
                "net_plan_movement":-2,"seats":3,"pct_urgent":0.5,"usage_density":0.5,
            },
            "🟢 Pelanggan Sehat — enterprise, loyal & aktif": {
                "tenure_days":600,"total_mrr":1200,"days_since_last_usage":1,
                "unique_features_used":32,"avg_satisfaction_score":4.9,
                "total_tickets":1,"escalation_rate":0.0,"sub_churn_ratio":0.0,
                "net_plan_movement":4,"seats":50,"pct_urgent":0.0,"usage_density":12.0,
            },
            "🟡 Pelanggan Borderline — perlu dipantau": {
                "tenure_days":210,"total_mrr":280,"days_since_last_usage":13,
                "unique_features_used":10,"avg_satisfaction_score":3.0,
                "total_tickets":5,"escalation_rate":0.2,"sub_churn_ratio":0.25,
                "net_plan_movement":0,"seats":9,"pct_urgent":0.15,"usage_density":2.5,
            },
        }

        s = SAMPLES[sample_choice]
        st.markdown(f"""
        <div class="gcard" style="margin-top:.75rem">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem">
                <div class="sig-row"><span class="sig-name">🏢 Tenure</span><span class="sig-val">{s['tenure_days']} hari</span></div>
                <div class="sig-row"><span class="sig-name">💰 MRR</span><span class="sig-val">${s['total_mrr']}</span></div>
                <div class="sig-row"><span class="sig-name">📱 Last login</span><span class="sig-val">{s['days_since_last_usage']} hari lalu</span></div>
                <div class="sig-row"><span class="sig-name">🔧 Fitur dipakai</span><span class="sig-val">{s['unique_features_used']}/40</span></div>
                <div class="sig-row"><span class="sig-name">⭐ Satisfaction</span><span class="sig-val">{s['avg_satisfaction_score']}/5</span></div>
                <div class="sig-row"><span class="sig-name">🎧 Tiket</span><span class="sig-val">{s['total_tickets']}</span></div>
                <div class="sig-row"><span class="sig-name">🚨 Escalation</span><span class="sig-val">{s['escalation_rate']:.0%}</span></div>
                <div class="sig-row"><span class="sig-name">📊 Plan movement</span><span class="sig-val">{s['net_plan_movement']:+d}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sample_btn = st.button("🔍 Prediksi Profil Ini", use_container_width=True)

        if sample_btn:
            tenure_days=s["tenure_days"]; total_mrr=s["total_mrr"]
            days_since_last_usage=s["days_since_last_usage"]; unique_features_used=s["unique_features_used"]
            avg_satisfaction_score=s["avg_satisfaction_score"]; total_tickets=s["total_tickets"]
            escalation_rate=s["escalation_rate"]; sub_churn_ratio=s["sub_churn_ratio"]
            net_plan_movement=s["net_plan_movement"]; seats=s["seats"]
            pct_urgent=s["pct_urgent"]; usage_density=s["usage_density"]
            predict_btn=True


# ============================================================
# RESULT
# ============================================================
with col_right:
    st.markdown('<div class="slabel">📈 Hasil Analisis Risiko</div>', unsafe_allow_html=True)

    if predict_btn:
        base = X_ref.mean().to_dict()
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
        })

        X_in = pd.DataFrame([base]).reindex(columns=FEATURES, fill_value=0)
        bc   = X_in.select_dtypes(include='bool').columns
        X_in[bc] = X_in[bc].astype(int)

        prob  = float(model.predict_proba(X_in)[0, 1])
        risk_label, risk_color, risk_emoji, css_class, biz_icon = get_risk(prob)
        action    = get_action(risk_label)
        predicted = prob >= THRESHOLD

        st.markdown(f"""
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
            ("💰", "MRR",               f"${total_mrr:,}",                    "#f1f5f9"),
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
        XGBoost · RandomizedSearchCV<br>
        SMOTE · Threshold 0.32
    </div>""", unsafe_allow_html=True)
with col_f2:
    st.markdown("""
    <div style="font-size:.75rem;color:#475569;text-align:center">
        <strong style="color:#6366f1">📊 Dataset</strong><br>
        RavenStack Synthetic SaaS<br>
        by <a href="https://rivalytics.medium.com" target="_blank"
        style="color:#6366f1">River @ Rivalytics</a>
    </div>""", unsafe_allow_html=True)
with col_f3:
    st.markdown("""
    <div style="font-size:.75rem;color:#475569;text-align:right">
        <strong style="color:#6366f1">⚡ Stack</strong><br>
        Python · Streamlit · SHAP<br>
        Portfolio Project — Data Sintetis
    </div>""", unsafe_allow_html=True)