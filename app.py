"""
Dashboard Analisis & Prediksi Kerawanan Sosio-Kesehatan Indonesia 2024
=======================================================================
Studi Kasus : Hubungan Kemiskinan (X) ↔ AHH (Y) di 38 Provinsi
Data        : BPS 2024  ·  data/bps_2024_clean.xlsx
Metode ML   : Linear Regression  &  K-Means Clustering (K=3)
"""

import streamlit as st
import pandas as pd
import numpy as np

from modules.data_loader   import load_and_preprocess_excel
from modules.model_trainer import train_linear_regression, train_kmeans_clustering
from modules.plots         import (
    plot_top_5_bar,
    plot_raw_scatter,
    plot_regression_scatter,
    plot_kmeans_clusters,
    plot_silhouette_gauge,
)

# ══════════════════════════════════════════════════════════════════════════════
# 0. PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dashboard Kerawanan Sosio-Kesehatan 2024",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. GLOBAL CSS — Green & White Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Google Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Reset & Base ────────────────────────────────────────────────────────── */
*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

/* ─── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"]     { display: none !important; }
[data-testid="stDecoration"]  { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
/* Header: buat transparan, sembunyikan isi tapi JANGAN hapus dari DOM */
[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}
[data-testid="stHeader"] > div:first-child { visibility: hidden !important; }

/* ─── Sidebar: tombol collapse dihapus, sidebar tidak bisa ditutup ────────── */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
/* Tombol expand (saat sidebar tertutup) — tetap bisa diklik */
[data-testid="collapsedControl"] {
    background: #16a34a !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 2px 0 10px rgba(22,163,74,0.35) !important;
    visibility: visible !important;
    display: flex !important;
}
[data-testid="collapsedControl"] button { color: #ffffff !important; }
[data-testid="collapsedControl"] svg    { fill: #ffffff !important; stroke: #ffffff !important; }

/* ─── App Background (Light) ──────────────────────────────────────────────── */
.stApp {
    background: #f0faf4;
}
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px;
}

/* ─── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #16a34a 0%, #15803d 60%, #14532d 100%) !important;
    border-right: 2px solid #166534;
    width: 270px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* Force ALL text in sidebar to white */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {
    color: rgba(255,255,255,0.9) !important;
}

/* ─── Radio nav items ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px !important; }
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    margin: 2px 0;
    transition: background 0.18s;
    cursor: pointer;
    border: none;
    width: 100%;
}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"][aria-checked="true"] {
    background: rgba(255,255,255,0.22) !important;
    border-left: 3px solid #bbf7d0 !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"][aria-checked="true"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ─── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: rgba(22,163,74,0.2) !important; margin: 1rem 0 !important; }

/* ─── Streamlit Slider ────────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div { background: #16a34a !important; }

/* ─── Buttons ─────────────────────────────────────────────────────────────── */
[data-testid="baseButton-secondary"] {
    background: rgba(22,163,74,0.08) !important;
    border: 1px solid rgba(22,163,74,0.3) !important;
    color: #15803d !important;
}

/* ─── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(22,163,74,0.06);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(22,163,74,0.12);
}
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 8px !important;
    padding: 0.45rem 1.1rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border: none !important;
    transition: all 0.18s !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    background: rgba(22,163,74,0.1) !important;
    color: #15803d !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: rgba(22,163,74,0.18) !important;
    color: #14532d !important;
    font-weight: 700 !important;
}
[data-testid="stTabs"] [role="tabpanel"] {
    padding-top: 1rem !important;
}

/* ─── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(22,163,74,0.15) !important;
}

/* ─── Alert boxes ─────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}

/* ─── Metric widget ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid rgba(22,163,74,0.15);
    border-radius: 10px;
    padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; }
[data-testid="stMetricValue"] { color: #14532d !important; font-weight: 700 !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   COMPONENT STYLES
   ═══════════════════════════════════════════════════════════════════════════ */

/* ─── KPI Card ────────────────────────────────────────────────────────────── */
.kpi-card {
    background: #ffffff;
    border: 1px solid rgba(22,163,74,0.15);
    border-top: 3px solid;
    border-radius: 12px;
    padding: 1.2rem 1.2rem 1rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(22,163,74,0.12);
}
.kpi-label {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #14532d;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.kpi-sub {
    font-size: 0.73rem;
    color: #94a3b8;
    line-height: 1.4;
}

/* ─── Section Title ───────────────────────────────────────────────────────── */
.sec-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #15803d;
    margin: 2rem 0 0.85rem;
    padding-bottom: 0.55rem;
    border-bottom: 2px solid rgba(22,163,74,0.2);
}

/* ─── Hero Banner ─────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(120deg, #14532d 0%, #166534 45%, #15803d 100%);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(22,163,74,0.2);
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(187,247,208,0.15) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -40px; left: 20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #bbf7d0;
    margin-bottom: 0.9rem;
}
.hero-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.5rem;
    line-height: 1.3;
}
.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1rem;
    margin-top: 0.5rem;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.65);
}
.hero-tag b { color: rgba(255,255,255,0.9); font-weight: 500; }

/* ─── Info Box ────────────────────────────────────────────────────────────── */
.info-box {
    background: rgba(22,163,74,0.05);
    border: 1px solid rgba(22,163,74,0.2);
    border-left: 3px solid #16a34a;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0 1rem;
    font-size: 0.87rem;
    color: #374151;
    line-height: 1.65;
}
.info-box code {
    background: rgba(22,163,74,0.1);
    padding: 0.1em 0.4em;
    border-radius: 4px;
    color: #15803d;
    font-size: 0.9em;
}

/* ─── Equation Box ────────────────────────────────────────────────────────── */
.eq-box {
    background: #ffffff;
    border: 1px solid rgba(22,163,74,0.2);
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    margin: 0.6rem 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.eq-label {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.5rem;
}
.eq-formula {
    font-size: 1.2rem;
    font-weight: 700;
    color: #15803d;
}

/* ─── Correlation Box ─────────────────────────────────────────────────────── */
.corr-box {
    background: #ffffff;
    border: 1px solid rgba(22,163,74,0.2);
    border-top: 3px solid #16a34a;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(22,163,74,0.08);
}
.corr-r {
    font-size: 2.8rem;
    font-weight: 800;
    color: #16a34a;
    line-height: 1;
}
.corr-label {
    font-size: 0.7rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.5rem;
}
.corr-desc {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.6rem;
    line-height: 1.55;
}
.corr-desc b { color: #374151; }

/* ─── Risk Badges ─────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-low  { background: rgba(22,163,74,0.1);   border: 1px solid rgba(22,163,74,0.3);   color: #15803d; }
.badge-med  { background: rgba(245,158,11,0.1);  border: 1px solid rgba(245,158,11,0.3);  color: #b45309; }
.badge-high { background: rgba(239,68,68,0.1);   border: 1px solid rgba(239,68,68,0.3);   color: #b91c1c; }

/* ─── Simulator Output Card ───────────────────────────────────────────────── */
.sim-card {
    background: #ffffff;
    border: 1px solid rgba(22,163,74,0.15);
    border-top: 3px solid;
    border-radius: 12px;
    padding: 1.8rem 1.4rem 1.4rem;
    text-align: center;
    min-height: 190px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.sim-card-label {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.sim-val { font-size: 3.2rem; font-weight: 800; line-height: 1; }
.sim-unit { font-size: 0.85rem; color: #94a3b8; margin-top: 0.1rem; }
.sim-formula { font-size: 0.73rem; color: #94a3b8; margin-top: 0.6rem; font-family: 'Courier New', monospace; }

/* ─── Sidebar Header (Wordmark) ───────────────────────────────────────────── */
.sb-wordmark {
    padding: 1.8rem 1.4rem 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.15);
}
.sb-wordmark-title {
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    line-height: 1.35;
    letter-spacing: -0.01em;
}
.sb-wordmark-sub {
    font-size: 0.72rem !important;
    color: rgba(255,255,255,0.55) !important;
    margin-top: 0.3rem;
}

/* ─── Sidebar Nav Label ───────────────────────────────────────────────────── */
.sb-section-label {
    font-size: 0.62rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45) !important;
    font-weight: 700;
    padding: 1.2rem 1.4rem 0.4rem;
}

/* ─── Sidebar Team Section ────────────────────────────────────────────────── */
.sb-team {
    margin-top: 1.5rem;
    padding: 1rem 1.4rem 0.6rem;
    border-top: 1px solid rgba(255,255,255,0.15);
}
.sb-team-label {
    font-size: 0.62rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45) !important;
    font-weight: 700;
    margin-bottom: 0.75rem;
}
.sb-team-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.38rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.sb-team-row:last-of-type { border-bottom: none; }
.sb-team-idx {
    font-size: 0.62rem !important;
    color: rgba(255,255,255,0.35) !important;
    font-weight: 600;
    width: 1rem;
    text-align: center;
    flex-shrink: 0;
}
.sb-team-name {
    font-size: 0.78rem !important;
    color: rgba(255,255,255,0.85) !important;
    font-weight: 500;
    flex: 1;
}
.sb-team-nim {
    font-size: 0.65rem !important;
    color: rgba(255,255,255,0.4) !important;
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
}
.sb-footer {
    font-size: 0.65rem !important;
    color: rgba(255,255,255,0.3) !important;
    padding: 0.6rem 1.4rem 1.2rem;
    text-align: center;
}

/* ─── Slider input display ────────────────────────────────────────────────── */
.input-display {
    background: #ffffff;
    border: 1px solid rgba(22,163,74,0.2);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin-top: 1.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.input-display-label {
    font-size: 0.65rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.input-display-value {
    font-size: 2rem;
    font-weight: 800;
    color: #16a34a;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD & TRAIN (cached)
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH = "data/bps_2024_clean.xlsx"

@st.cache_data(show_spinner="⏳ Memuat dataset BPS 2024…")
def load_data():
    return load_and_preprocess_excel(DATA_PATH)

@st.cache_data(show_spinner="🤖 Melatih model Machine Learning…")
def train_models(_df):
    lr_model, y_pred, lr_metrics = train_linear_regression(_df)
    kmeans_model, df_c, sil = train_kmeans_clustering(_df)
    return lr_model, y_pred, lr_metrics, kmeans_model, df_c, sil

try:
    df_raw, scaler = load_data()
except Exception as e:
    st.error(f"❌ Gagal membaca `{DATA_PATH}`.\n\n`{e}`")
    st.stop()

lr_model, y_pred, lr_metrics, kmeans_model, df, sil_score = train_models(df_raw)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Wordmark ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sb-wordmark">
        <div class="sb-wordmark-title">🌿 Dashboard Analisis<br>Sosio-Kesehatan Indonesia</div>
        <div class="sb-wordmark-sub">BPS 2024 &nbsp;·&nbsp; Proyek Data Sains</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section-label">Menu Navigasi</div>', unsafe_allow_html=True)

    menu = st.radio(
        "nav",
        options=[
            "Profil & Preprocessing",
            "Eksplorasi Data (EDA)",
            "Regresi Linear",
            "K-Means Clustering",
            "Simulator Prediksi",
        ],
        label_visibility="collapsed",
    )

    # ── Team (bottom) ─────────────────────────────────────────────────────────
    members = [
        ("Fito Anadiansyah",    "2101020013"),
        ("Ahmad Zidane",        "2301020004"),
        ("Andi Kevin R. Panca", "2301020015"),
        ("Rizki Wardani",       "2301020090"),
    ]
    rows = "".join([
        f'<div class="sb-team-row">'
        f'  <span class="sb-team-idx">{i+1}</span>'
        f'  <span class="sb-team-name">{name}</span>'
        f'  <span class="sb-team-nim">{nim}</span>'
        f'</div>'
        for i, (name, nim) in enumerate(members)
    ])
    st.markdown(f"""
    <div class="sb-team">
        <div class="sb-team-label">Kelompok Penyusun</div>
        {rows}
    </div>
    <div class="sb-footer">Teknik Informatika &nbsp;·&nbsp; 2024</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
module_names = {
    "Profil & Preprocessing": "Modul 1",
    "Eksplorasi Data (EDA)":  "Modul 2",
    "Regresi Linear":         "Modul 3",
    "K-Means Clustering":     "Modul 4",
    "Simulator Prediksi":     "Modul 5",
}
active_mod = module_names.get(menu, "")

st.markdown(f"""
<div class="hero">
    <div class="hero-badge">🌿 {active_mod} &nbsp;&middot;&nbsp; BPS 2024</div>
    <div class="hero-title">Dashboard Analisis &amp; Prediksi<br>Kerawanan Sosio-Kesehatan Indonesia</div>
    <div class="hero-meta">
        <span class="hero-tag">📍 <b>38 Provinsi Indonesia</b></span>
        <span class="hero-tag">🤖 <b>Linear Regression &amp; K-Means</b></span>
        <span class="hero-tag">📊 <b>Data Resmi BPS 2024</b></span>
        <span class="hero-tag">👥 <b>Fito &middot; Ahmad &middot; Kevin &middot; Rizki</b></span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def sec(title: str):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)

def kpi(col, label: str, value: str, sub: str, accent: str = "#16a34a"):
    col.markdown(
        f'<div class="kpi-card" style="border-top-color:{accent};">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  <div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODUL 1 — PROFIL & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
if menu == "Profil & Preprocessing":

    sec("Ringkasan Statistik Dataset")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Wilayah",        f"{len(df)}",                                     "Provinsi Indonesia",                                           "#16a34a")
    kpi(c2, "Rata-rata Kemiskinan", f"{df['Kemiskinan_Pct'].mean():.2f}%",
        f"Min {df['Kemiskinan_Pct'].min():.2f}%  ·  Maks {df['Kemiskinan_Pct'].max():.2f}%", "#ef4444")
    kpi(c3, "Rata-rata AHH",        f"{df['AHH_Tahun'].mean():.2f}",
        f"Min {df['AHH_Tahun'].min():.2f}  ·  Maks {df['AHH_Tahun'].max():.2f} Thn",        "#059669")
    kpi(c4, "Sumber Data",          "BPS 2024",                                        "bps_2024_clean.xlsx",                                          "#f59e0b")

    sec("Algoritma Min-Max Normalization")
    st.markdown("""
    <div class="info-box">
        <b>Formula:</b> &nbsp;<code>X_norm = (X − X_min) / (X_max − X_min)</code><br><br>
        Normalisasi mengubah seluruh nilai numerik ke rentang <code>[0.0 – 1.0]</code> agar tidak terjadi dominasi
        skala saat proses K-Means Clustering. Setiap kolom dinormalisasi secara independen.
    </div>
    """, unsafe_allow_html=True)

    sec("Tabel Master Data — Aktual vs Normalisasi")

    display_df = df[["Provinsi", "Kemiskinan_Pct", "AHH_Tahun", "Kemiskinan_Norm", "AHH_Norm"]].copy()
    display_df = display_df.rename(columns={
        "Kemiskinan_Pct":  "Kemiskinan (%)",
        "AHH_Tahun":       "AHH (Tahun)",
        "Kemiskinan_Norm": "Kemiskinan_Norm",
        "AHH_Norm":        "AHH_Norm",
    })
    st.dataframe(
        display_df.style
            .format({
                "Kemiskinan (%)":  "{:.2f}",
                "AHH (Tahun)":     "{:.2f}",
                "Kemiskinan_Norm": "{:.4f}",
                "AHH_Norm":        "{:.4f}",
            })
            .background_gradient(subset=["Kemiskinan (%)"],  cmap="YlOrRd", vmin=0)
            .background_gradient(subset=["AHH (Tahun)"],     cmap="YlGn",   vmin=60)
            .background_gradient(subset=["Kemiskinan_Norm"], cmap="YlOrRd", vmin=0, vmax=1)
            .background_gradient(subset=["AHH_Norm"],        cmap="YlGn",   vmin=0, vmax=1),
        width="stretch",
        height=480,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODUL 2 — EKSPLORASI DATA (EDA)
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Eksplorasi Data (EDA)":

    sec("Top 5 Provinsi")
    fig_pov, fig_ahh = plot_top_5_bar(df)
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_pov, use_container_width=True)
    c2.plotly_chart(fig_ahh, use_container_width=True)

    sec("Korelasi Pearson")
    corr_val = df["Kemiskinan_Pct"].corr(df["AHH_Tahun"])

    ca, cb, cc = st.columns([1, 1.8, 1])
    with cb:
        st.markdown(
            f'<div class="corr-box">'
            f'  <div class="corr-label">Koefisien Korelasi Pearson (r)</div>'
            f'  <div class="corr-r">{corr_val:.4f}</div>'
            f'  <div class="corr-desc">'
            f'    Hubungan <b>Negatif Kuat</b> antara Kemiskinan &amp; AHH.<br>'
            f'    Semakin tinggi kemiskinan → semakin rendah angka harapan hidup.'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    sec("Scatter Plot Sebaran Data Asli")
    st.plotly_chart(plot_raw_scatter(df), use_container_width=True)

    sec("Statistik Deskriptif")
    desc = df[["Kemiskinan_Pct", "AHH_Tahun"]].describe().T
    desc.index = ["Kemiskinan (%)", "AHH (Tahun)"]
    st.dataframe(desc.style.format("{:.4f}"), width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# MODUL 3 — REGRESI LINEAR
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Regresi Linear":

    sec("Persamaan Regresi Linear (OLS)")

    sign = "−" if lr_metrics["coef"] < 0 else "+"
    st.markdown(
        f'<div class="eq-box">'
        f'  <div class="eq-label">Model Regresi OLS — Scikit-Learn</div>'
        f'  <div class="eq-formula">'
        f'    Ŷ&thinsp;(AHH) = {lr_metrics["intercept"]:.4f} {sign} {abs(lr_metrics["coef"]):.4f} × X&thinsp;(Kemiskinan)'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    latex_sign = "-" if lr_metrics["coef"] < 0 else "+"
    st.latex(
        rf"\hat{{Y}} = {lr_metrics['intercept']:.4f}"
        rf"\; {latex_sign} \;"
        rf"{abs(lr_metrics['coef']):.4f} \cdot X"
    )

    sec("Metrik Evaluasi Model")
    m1, m2, m3 = st.columns(3)
    kpi(m1, "R² Score",  f"{lr_metrics['r2']:.4f}",   "Koefisien Determinasi",    "#16a34a")
    kpi(m2, "MAE",       f"{lr_metrics['mae']:.4f}",  "Mean Absolute Error (Thn)", "#059669")
    kpi(m3, "RMSE",      f"{lr_metrics['rmse']:.4f}", "Root Mean Sq. Error (Thn)", "#f59e0b")

    sec("Plot Regresi Interaktif")
    st.plotly_chart(plot_regression_scatter(df, lr_metrics, y_pred), use_container_width=True)

    sec("Perbandingan Prediksi vs Aktual")
    df_pred = df[["Provinsi", "Kemiskinan_Pct", "AHH_Tahun"]].copy()
    df_pred["AHH_Prediksi"]  = y_pred
    df_pred["Residual"]      = df_pred["AHH_Tahun"] - df_pred["AHH_Prediksi"]
    df_pred["Akurasi (%)"]   = (1 - abs(df_pred["Residual"]) / df_pred["AHH_Tahun"]) * 100

    st.dataframe(
        df_pred.rename(columns={
            "Kemiskinan_Pct": "Kemiskinan (%)",
            "AHH_Tahun":      "AHH Aktual (Thn)",
            "AHH_Prediksi":   "AHH Prediksi (Thn)",
        }).style.format({
            "Kemiskinan (%)":      "{:.2f}",
            "AHH Aktual (Thn)":    "{:.2f}",
            "AHH Prediksi (Thn)":  "{:.2f}",
            "Residual":            "{:+.4f}",
            "Akurasi (%)":         "{:.2f}%",
        }).background_gradient(subset=["Akurasi (%)"], cmap="RdYlGn", vmin=95, vmax=100),
        width="stretch",
        height=440,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODUL 4 — K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "K-Means Clustering":

    sec("Parameter & Evaluasi Model")
    p1, p2, p3 = st.columns(3)
    kpi(p1, "Jumlah Kluster (K)",  "3",               "Kelompok Risiko Sosio-Kesehatan", "#16a34a")
    kpi(p2, "Silhouette Score",    f"{sil_score:.4f}", "0 = Buruk  ·  1 = Sempurna",      "#059669")

    rc = df["Kategori_Risiko"].value_counts()
    dist_str = " · ".join([
        f"{v} {'Tinggi' if 'Tinggi' in k else 'Sedang' if 'Sedang' in k else 'Rendah'}"
        for k, v in rc.items()
    ])
    kpi(p3, "Distribusi Provinsi", dist_str, "Per kelompok risiko", "#f59e0b")

    sec("Evaluasi Silhouette Score")
    gc1, gc2, gc3 = st.columns([1, 2, 1])
    with gc2:
        st.plotly_chart(plot_silhouette_gauge(sil_score), use_container_width=True)

    sec("Peta Segmentasi Wilayah K-Means (K=3)")
    st.plotly_chart(plot_kmeans_clusters(df), use_container_width=True)

    sec("Rincian Anggota Tiap Kluster")
    risk_order = [
        "Kluster 3: Risiko Sangat Tinggi (Prioritas)",
        "Kluster 2: Risiko Sedang",
        "Kluster 1: Risiko Rendah (Kondisi Baik)",
    ]
    tabs = st.tabs(["🔴 Risiko Sangat Tinggi", "🟡 Risiko Sedang", "🟢 Risiko Rendah"])
    for tab, rkey in zip(tabs, risk_order):
        sub = (
            df[df["Kategori_Risiko"] == rkey][
                ["Provinsi", "Kemiskinan_Pct", "AHH_Tahun", "Kemiskinan_Norm", "AHH_Norm"]
            ]
            .sort_values("Kemiskinan_Pct", ascending=False)
            .reset_index(drop=True)
        )
        sub.index += 1
        with tab:
            a, b, c_ = st.columns(3)
            a.metric("Jumlah Provinsi",     f"{len(sub)}")
            b.metric("Rata-rata Kemiskinan", f"{sub['Kemiskinan_Pct'].mean():.2f}%")
            c_.metric("Rata-rata AHH",       f"{sub['AHH_Tahun'].mean():.2f} Thn")
            st.dataframe(
                sub.rename(columns={
                    "Kemiskinan_Pct":  "Kemiskinan (%)",
                    "AHH_Tahun":       "AHH (Tahun)",
                    "Kemiskinan_Norm": "Kemiskinan_Norm",
                    "AHH_Norm":        "AHH_Norm",
                }).style.format({
                    "Kemiskinan (%)":  "{:.2f}",
                    "AHH (Tahun)":     "{:.2f}",
                    "Kemiskinan_Norm": "{:.4f}",
                    "AHH_Norm":        "{:.4f}",
                }),
                width="stretch",
            )


# ══════════════════════════════════════════════════════════════════════════════
# MODUL 5 — SIMULATOR PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Simulator Prediksi":

    sec("Kalkulator Estimasi AHH & Kategori Risiko")

    st.markdown("""
    <div class="info-box">
        Masukkan persentase penduduk miskin pada bilah geser di bawah. Sistem akan mengestimasi
        <b>Angka Harapan Hidup (AHH)</b> menggunakan model Regresi Linear, serta menentukan
        <b>Kategori Risiko Sosio-Kesehatan</b> via K-Means Clustering secara real-time.
    </div>
    """, unsafe_allow_html=True)

    # ── Slider ──────────────────────────────────────────────────────────────
    sa, sb = st.columns([3, 1])
    with sa:
        val_poverty = st.slider(
            "Persentase Penduduk Miskin (%)",
            min_value=0.0, max_value=40.0, value=10.0, step=0.1, format="%.1f%%",
        )
    with sb:
        st.markdown(f"""
        <div class="input-display">
            <div class="input-display-label">Input</div>
            <div class="input-display-value">{val_poverty:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Compute ──────────────────────────────────────────────────────────────
    pred_ahh  = float(lr_model.predict([[val_poverty]])[0])
    k_min_v,   k_max_v   = df["Kemiskinan_Pct"].min(), df["Kemiskinan_Pct"].max()
    ahh_min_v, ahh_max_v = df["AHH_Tahun"].min(),      df["AHH_Tahun"].max()
    pov_norm  = float(np.clip((val_poverty - k_min_v)   / (k_max_v   - k_min_v),  0, 1))
    ahh_norm_ = float(np.clip((pred_ahh    - ahh_min_v) / (ahh_max_v - ahh_min_v), 0, 1))
    cluster_raw  = int(kmeans_model.predict([[pov_norm, ahh_norm_]])[0])
    centroids    = kmeans_model.cluster_centers_
    sorted_by_c  = sorted(range(len(centroids)), key=lambda i: centroids[i][0])
    raw_to_label = {
        sorted_by_c[0]: ("Kluster 1: Risiko Rendah (Kondisi Baik)",           "badge-low",  "🟢", "#16a34a"),
        sorted_by_c[1]: ("Kluster 2: Risiko Sedang",                           "badge-med",  "🟡", "#f59e0b"),
        sorted_by_c[2]: ("Kluster 3: Risiko Sangat Tinggi (Prioritas)",        "badge-high", "🔴", "#ef4444"),
    }
    kat_label, badge_cls, risk_emoji, accent_color = raw_to_label[cluster_raw]

    # ── Output Cards ─────────────────────────────────────────────────────────
    sec("Hasil Estimasi")
    r1, r2 = st.columns(2)

    sign_str = "−" if lr_metrics["coef"] < 0 else "+"
    with r1:
        st.markdown(
            f'<div class="sim-card" style="border-top-color:#16a34a;">'
            f'  <div class="sim-card-label">Estimasi Angka Harapan Hidup (AHH)</div>'
            f'  <div class="sim-val" style="color:#16a34a;">{pred_ahh:.2f}</div>'
            f'  <div class="sim-unit">Tahun</div>'
            f'  <div class="sim-formula">'
            f'    AHH = {lr_metrics["intercept"]:.4f} {sign_str} {abs(lr_metrics["coef"]):.4f} × {val_poverty:.1f}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f'<div class="sim-card" style="border-top-color:{accent_color};">'
            f'  <div class="sim-card-label">Kategori Risiko Sosio-Kesehatan</div>'
            f'  <div style="font-size:2.4rem;line-height:1;margin:.4rem 0;">{risk_emoji}</div>'
            f'  <span class="badge {badge_cls}">{kat_label}</span>'
            f'  <div class="sim-formula" style="margin-top:.8rem;">'
            f'    Titik Normalisasi: [{pov_norm:.3f}, {ahh_norm_:.3f}]'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    sec("Kalkulasi Detail")
    latex_sign = "-" if lr_metrics["coef"] < 0 else "+"
    st.latex(
        rf"\hat{{Y}} = {lr_metrics['intercept']:.4f}"
        rf"\; {latex_sign} \;"
        rf"{abs(lr_metrics['coef']):.4f} \times {val_poverty:.1f}"
        rf"= {pred_ahh:.4f} \text{{ Tahun}}"
    )

    # ── Context ───────────────────────────────────────────────────────────────
    sec("Konteks vs Data 38 Provinsi")
    n_below = int((df["Kemiskinan_Pct"] <= val_poverty).sum())
    st.info(
        f"Input kemiskinan **{val_poverty:.1f}%** berada di atas/setara "
        f"**{n_below} dari {len(df)} provinsi** ({round(n_below/len(df)*100)}%) dalam dataset BPS 2024."
    )
    df_ctx = df.copy()
    df_ctx["Δ Kemiskinan"] = (df_ctx["Kemiskinan_Pct"] - val_poverty).abs()
    nearest = (
        df_ctx.nsmallest(8, "Δ Kemiskinan")[
            ["Provinsi", "Kemiskinan_Pct", "AHH_Tahun", "Kategori_Risiko", "Δ Kemiskinan"]
        ]
        .reset_index(drop=True)
    )
    nearest.index += 1
    st.caption("8 Provinsi terdekat dengan nilai input kemiskinan:")
    st.dataframe(
        nearest.rename(columns={
            "Kemiskinan_Pct": "Kemiskinan (%)",
            "AHH_Tahun":      "AHH Aktual (Thn)",
        }).style.format({
            "Kemiskinan (%)":   "{:.2f}",
            "AHH Aktual (Thn)": "{:.2f}",
            "Δ Kemiskinan":     "{:.2f}",
        }),
        width="stretch",
    )