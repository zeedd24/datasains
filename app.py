
import streamlit as st
import pandas as pd
import numpy as np

from modules.data_loader import load_and_preprocess_excel
from modules.model_trainer import (
    train_linear_regression,
    train_kmeans_clustering,
)
from modules.plots import (
    plot_top_5_bar,
    plot_raw_scatter,
    plot_regression_scatter,
    plot_kmeans_clusters,
    plot_silhouette_gauge,
)

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Dashboard Kerawanan Sosio-Kesehatan 2024",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. CSS - GREEN & WHITE THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], [data-testid="stApp"] {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background: #f0faf4;
}

.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    transition: all 0.3s ease;
}

/* Header tetap tersedia agar tombol sidebar bawaan berfungsi */
[data-testid="stHeader"] {
    background: rgba(240, 250, 244, 0.95) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        160deg,
        #166534 0%,
        #15803d 55%,
        #14532d 100%
    ) !important;
    border-right: 1px solid #166534;
}

/* Padding sidebar */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem;
}

/* Teks sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small {
    color: #ffffff;
}

/* Tombol collapse sidebar bawaan Streamlit */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    display: flex !important;
}

[data-testid="stSidebarCollapseButton"] button {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 8px !important;
    color: white !important;
}

[data-testid="stSidebarCollapseButton"] svg {
    fill: white !important;
    stroke: white !important;
}

/* Tombol membuka sidebar saat sidebar tertutup */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    z-index: 999999 !important;
}

[data-testid="collapsedControl"] button {
    background: #16a34a !important;
    color: white !important;
    border: 1px solid #15803d !important;
    border-radius: 0 10px 10px 0 !important;
    box-shadow: 2px 2px 12px rgba(22,163,74,0.25);
}

[data-testid="collapsedControl"] svg {
    fill: white !important;
    stroke: white !important;
}

/* ─── Sidebar: Wordmark ────────────────────────────────── */
.sb-wordmark {
    padding: 1.8rem 1.2rem 1.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.sb-wordmark-title {
    color: #ffffff;
    font-size: 0.92rem;
    font-weight: 700;
    line-height: 1.5;
    letter-spacing: -0.01em;
}

.sb-wordmark-sub {
    color: rgba(255,255,255,0.45);
    font-size: 0.68rem;
    margin-top: 0.3rem;
    letter-spacing: 0.04em;
}

/* ─── Sidebar: Nav label ───────────────────────────────── */
.sb-section-label {
    padding: 1.3rem 1.2rem 0.45rem;
    color: rgba(255,255,255,0.38);
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}

/* ─── Sidebar: Radio — sembunyikan lingkaran bawaan ─────── */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 2px;
    padding: 0 0.6rem;
}

/* Sembunyikan circle radio indicator */
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] label[data-baseweb="radio"] {
    background: transparent;
    padding: 0.65rem 1rem;
    border-radius: 8px;
    margin: 1px 0;
    width: 100%;
    border-left: 3px solid transparent;
    transition: background 0.18s, border-left-color 0.18s;
    cursor: pointer;
}

[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.08);
    border-left-color: rgba(255,255,255,0.25);
}

[data-testid="stSidebar"] label[data-baseweb="radio"][aria-checked="true"] {
    background: rgba(255,255,255,0.13);
    border-left: 3px solid #4ade80;
}

[data-testid="stSidebar"] label[data-baseweb="radio"] span {
    color: rgba(255,255,255,0.8) !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] label[data-baseweb="radio"][aria-checked="true"] span {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ─── Sidebar: Footer ──────────────────────────────────── */
.sb-footer {
    margin-top: 2rem;
    padding: 1rem 1.2rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.3);
    font-size: 0.62rem;
    letter-spacing: 0.04em;
    line-height: 1.6;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #14532d, #166534, #16a34a);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 5px 20px rgba(22,163,74,0.15);
}

.hero-badge {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 30px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.2);
    color: #dcfce7;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.35;
    color: white;
    margin-bottom: 0.7rem;
}

.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem 1.2rem;
    margin-top: 1rem;
}

.hero-tag {
    font-size: 0.78rem;
    color: #dcfce7;
}

/* Judul bagian */
.sec-title {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.07em;
    color: #15803d;
    text-transform: uppercase;
    margin: 1.8rem 0 0.8rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #d1fae5;
}

/* KPI */
.kpi-card {
    background: white;
    border: 1px solid #d1fae5;
    border-top: 4px solid #16a34a;
    border-radius: 12px;
    padding: 1.1rem;
    height: 100%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.035);
}

.kpi-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.kpi-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #14532d;
    overflow-wrap: anywhere;
}

.kpi-sub {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* Kotak informasi */
.info-box {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.6rem 0 1rem;
    color: #374151;
    font-size: 0.87rem;
    line-height: 1.7;
}

/* Persamaan regresi */
.eq-box, .corr-box, .input-display {
    background: white;
    border: 1px solid #d1fae5;
    border-radius: 12px;
    padding: 1.3rem;
    text-align: center;
}

.eq-label, .corr-label, .input-display-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.eq-formula {
    font-size: 1.15rem;
    font-weight: 800;
    color: #15803d;
    overflow-wrap: anywhere;
}

.corr-r {
    font-size: 2.7rem;
    font-weight: 800;
    color: #16a34a;
}

.corr-desc {
    color: #64748b;
    font-size: 0.85rem;
    line-height: 1.6;
    margin-top: 0.7rem;
}

/* Badge risiko */
.badge {
    display: inline-block;
    border-radius: 30px;
    padding: 0.35rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 700;
}

.badge-low {
    background: #dcfce7;
    color: #166534;
}

.badge-med {
    background: #fef3c7;
    color: #92400e;
}

.badge-high {
    background: #fee2e2;
    color: #991b1b;
}

/* Simulator */
.sim-card {
    background: white;
    border: 1px solid #d1fae5;
    border-top: 4px solid #16a34a;
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    min-height: 190px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.sim-card-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
}

.sim-val {
    font-size: 2.7rem;
    font-weight: 800;
}

.sim-unit, .sim-formula {
    font-size: 0.78rem;
    color: #64748b;
    overflow-wrap: anywhere;
}

.input-display {
    margin-top: 1.5rem;
}

.input-display-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #16a34a;
}

/* Tabs, tombol, tabel, dan notifikasi */
[data-testid="stTabs"] [role="tablist"] {
    gap: 5px;
    background: #ecfdf5;
    padding: 5px;
    border-radius: 10px;
}

[data-testid="stTabs"] button[role="tab"] {
    border-radius: 8px;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #166534;
    font-weight: 700;
}

[data-testid="stDataFrame"] {
    border: 1px solid #d1fae5;
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d1fae5;
    border-radius: 10px;
    padding: 0.8rem;
}

[data-testid="stMetricValue"] {
    color: #14532d;
}

hr {
    border-color: #d1fae5;
}

/* Responsif untuk layar kecil */
@media (max-width: 768px) {
    .hero {
        padding: 1.2rem;
    }

    .hero-title {
        font-size: 1.35rem;
    }

    .kpi-value {
        font-size: 1.35rem;
    }

    .eq-formula {
        font-size: 0.95rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD DATA & TRAIN MODEL
# ============================================================
DATA_PATH = "data/bps_2024_clean.xlsx"


@st.cache_data(show_spinner="Memuat dataset BPS 2024...")
def load_data():
    return load_and_preprocess_excel(DATA_PATH)


@st.cache_data(show_spinner="Melatih model Machine Learning...")
def train_models(_df):
    lr_model, y_pred, lr_metrics = train_linear_regression(_df)
    kmeans_model, df_c, sil = train_kmeans_clustering(_df)
    return lr_model, y_pred, lr_metrics, kmeans_model, df_c, sil


try:
    df_raw, scaler = load_data()
    lr_model, y_pred, lr_metrics, kmeans_model, df, sil_score = (
        train_models(df_raw)
    )
except Exception as e:
    st.error(f"Gagal memuat data atau melatih model: {e}")
    st.stop()


# ============================================================
# 4. SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-wordmark">
        <div class="sb-wordmark-title">Dashboard Analisis<br>Sosio-Kesehatan Indonesia</div>
        <div class="sb-wordmark-sub">BPS 2024 &nbsp;&middot;&nbsp; Proyek Data Sains</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sb-section-label">Menu Navigasi</div>',
        unsafe_allow_html=True,
    )

    menu = st.radio(
        "Navigasi dashboard",
        options=[
            "Profil & Preprocessing",
            "Eksplorasi Data (EDA)",
            "Regresi Linear",
            "K-Means Clustering",
            "Simulator Prediksi",
        ],
        label_visibility="collapsed",
        key="main_navigation",
    )

    st.markdown(
        '<div class="sb-footer">Teknik Informatika &nbsp;&middot;&nbsp; Proyek Data Sains</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================
module_names = {
    "Profil & Preprocessing": "MODUL 1",
    "Eksplorasi Data (EDA)": "MODUL 2",
    "Regresi Linear": "MODUL 3",
    "K-Means Clustering": "MODUL 4",
    "Simulator Prediksi": "MODUL 5",
}


def sec(title):
    st.markdown(
        f'<div class="sec-title">{title}</div>',
        unsafe_allow_html=True,
    )


def kpi(col, label, value, sub, accent="#16a34a"):
    col.markdown(
        f"""
        <div class="kpi-card" style="border-top-color:{accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 6. HERO BANNER
# ============================================================
active_mod = module_names.get(menu, "")

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-badge"> {active_mod} · BPS 2024</div>
        <div class="hero-title">
            Dashboard Analisis &amp; Prediksi<br>
            Kerawanan Sosio-Kesehatan Indonesia
        </div>
        <div class="hero-meta">
            <span class="hero-tag">38 Provinsi Indonesia</span>
            <span class="hero-tag">Linear Regression &amp; K-Means</span>
            <span class="hero-tag">Data BPS 2024</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODUL 1 — PROFIL & PREPROCESSING
# ============================================================
if menu == "Profil & Preprocessing":

    sec("Ringkasan Statistik Dataset")

    c1, c2, c3, c4 = st.columns(4)

    kpi(
        c1, "Total Wilayah", f"{len(df)}",
        "Provinsi Indonesia", "#16a34a",
    )
    kpi(
        c2, "Rata-rata Kemiskinan",
        f"{df['Kemiskinan_Pct'].mean():.2f}%",
        f"Min {df['Kemiskinan_Pct'].min():.2f}% · "
        f"Maks {df['Kemiskinan_Pct'].max():.2f}%",
        "#ef4444",
    )
    kpi(
        c3, "Rata-rata AHH",
        f"{df['AHH_Tahun'].mean():.2f}",
        f"Min {df['AHH_Tahun'].min():.2f} · "
        f"Maks {df['AHH_Tahun'].max():.2f} tahun",
        "#059669",
    )
    kpi(
        c4, "Sumber Data", "BPS 2024",
        "bps_2024_clean.xlsx", "#f59e0b",
    )

    sec("Algoritma Min-Max Normalization")

    st.markdown("""
    <div class="info-box">
        <b>Formula:</b>
        <code>X_norm = (X − X_min) / (X_max − X_min)</code><br><br>
        Normalisasi mengubah nilai numerik ke rentang 0 sampai 1.
        Langkah ini membantu menyamakan skala variabel saat proses
        K-Means Clustering. Setiap kolom dinormalisasi secara independen.
    </div>
    """, unsafe_allow_html=True)

    sec("Tabel Master Data — Aktual vs Normalisasi")

    display_df = df[
        [
            "Provinsi",
            "Kemiskinan_Pct",
            "AHH_Tahun",
            "Kemiskinan_Norm",
            "AHH_Norm",
        ]
    ].copy()

    display_df = display_df.rename(columns={
        "Kemiskinan_Pct": "Kemiskinan (%)",
        "AHH_Tahun": "AHH (Tahun)",
    })

    st.dataframe(
        display_df.style
        .format({
            "Kemiskinan (%)": "{:.2f}",
            "AHH (Tahun)": "{:.2f}",
            "Kemiskinan_Norm": "{:.4f}",
            "AHH_Norm": "{:.4f}",
        })
        .background_gradient(
            subset=["Kemiskinan (%)"], cmap="YlOrRd"
        )
        .background_gradient(
            subset=["AHH (Tahun)"], cmap="YlGn"
        )
        .background_gradient(
            subset=["Kemiskinan_Norm"], cmap="YlOrRd",
            vmin=0, vmax=1,
        )
        .background_gradient(
            subset=["AHH_Norm"], cmap="YlGn",
            vmin=0, vmax=1,
        ),
        use_container_width=True,
        height=480,
    )


# ============================================================
# MODUL 2 — EKSPLORASI DATA
# ============================================================
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
        if corr_val <= -0.7:
            corr_desc = "Negatif kuat"
        elif corr_val <= -0.3:
            corr_desc = "Negatif sedang"
        elif corr_val < 0:
            corr_desc = "Negatif lemah"
        elif corr_val >= 0.7:
            corr_desc = "Positif kuat"
        elif corr_val >= 0.3:
            corr_desc = "Positif sedang"
        else:
            corr_desc = "Positif lemah atau mendekati nol"

        st.markdown(
            f"""
            <div class="corr-box">
                <div class="corr-label">
                    Koefisien Korelasi Pearson (r)
                </div>
                <div class="corr-r">{corr_val:.4f}</div>
                <div class="corr-desc">
                    Hubungan <b>{corr_desc}</b> antara kemiskinan
                    dan angka harapan hidup.<br>
                    Korelasi menunjukkan hubungan statistik,
                    bukan bukti sebab-akibat.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    sec("Scatter Plot Sebaran Data Asli")

    st.plotly_chart(
        plot_raw_scatter(df),
        use_container_width=True,
    )

    sec("Statistik Deskriptif")

    desc = df[["Kemiskinan_Pct", "AHH_Tahun"]].describe().T
    desc.index = ["Kemiskinan (%)", "AHH (Tahun)"]

    st.dataframe(
        desc.style.format("{:.4f}"),
        use_container_width=True,
    )


# ============================================================
# MODUL 3 — REGRESI LINEAR
# ============================================================
elif menu == "Regresi Linear":

    sec("Persamaan Regresi Linear (OLS)")

    sign = "−" if lr_metrics["coef"] < 0 else "+"

    st.markdown(
        f"""
        <div class="eq-box">
            <div class="eq-label">
                Model Regresi Linear — Scikit-Learn
            </div>
            <div class="eq-formula">
                Ŷ (AHH) = {lr_metrics['intercept']:.4f}
                {sign} {abs(lr_metrics['coef']):.4f}
                × X (Kemiskinan)
            </div>
        </div>
        """,
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

    kpi(
        m1, "R² Score", f"{lr_metrics['r2']:.4f}",
        "Koefisien determinasi", "#16a34a",
    )
    kpi(
        m2, "MAE", f"{lr_metrics['mae']:.4f}",
        "Mean Absolute Error (tahun)", "#059669",
    )
    kpi(
        m3, "RMSE", f"{lr_metrics['rmse']:.4f}",
        "Root Mean Squared Error (tahun)", "#f59e0b",
    )

    sec("Plot Regresi")

    st.plotly_chart(
        plot_regression_scatter(df, lr_metrics, y_pred),
        use_container_width=True,
    )

    sec("Perbandingan Prediksi vs Aktual")

    df_pred = df[
        ["Provinsi", "Kemiskinan_Pct", "AHH_Tahun"]
    ].copy()

    df_pred["AHH_Prediksi"] = y_pred
    df_pred["Residual"] = (
        df_pred["AHH_Tahun"] - df_pred["AHH_Prediksi"]
    )

    df_pred["Akurasi (%)"] = (
        1 - abs(df_pred["Residual"]) / df_pred["AHH_Tahun"]
    ) * 100

    df_pred = df_pred.rename(columns={
        "Kemiskinan_Pct": "Kemiskinan (%)",
        "AHH_Tahun": "AHH Aktual (Thn)",
        "AHH_Prediksi": "AHH Prediksi (Thn)",
    })

    st.dataframe(
        df_pred.style.format({
            "Kemiskinan (%)": "{:.2f}",
            "AHH Aktual (Thn)": "{:.2f}",
            "AHH Prediksi (Thn)": "{:.2f}",
            "Residual": "{:+.4f}",
            "Akurasi (%)": "{:.2f}%",
        }).background_gradient(
            subset=["Akurasi (%)"],
            cmap="RdYlGn",
            vmin=95,
            vmax=100,
        ),
        use_container_width=True,
        height=440,
    )

    st.caption(
        "Catatan: persentase pada kolom Akurasi merupakan "
        "ukuran berbasis residual, bukan akurasi klasifikasi."
    )


# ============================================================
# MODUL 4 — K-MEANS CLUSTERING
# ============================================================
elif menu == "K-Means Clustering":

    sec("Parameter & Evaluasi Model")

    p1, p2, p3 = st.columns(3)

    kpi(
        p1, "Jumlah Kluster (K)", "3",
        "Kelompok risiko sosio-kesehatan", "#16a34a",
    )
    kpi(
        p2, "Silhouette Score", f"{sil_score:.4f}",
        "Mendekati 1 umumnya menunjukkan kluster lebih terpisah",
        "#059669",
    )

    rc = df["Kategori_Risiko"].value_counts()

    # Petakan label risiko ke warna & nama pendek
    risk_map = {
        "Kluster 3: Risiko Sangat Tinggi (Prioritas)": ("#ef4444", "Risiko Tinggi"),
        "Kluster 2: Risiko Sedang":                    ("#f59e0b", "Risiko Sedang"),
        "Kluster 1: Risiko Rendah (Kondisi Baik)":     ("#16a34a", "Risiko Rendah"),
    }

    dist_rows = ""
    for label, (color, short) in risk_map.items():
        count = rc.get(label, 0)
        dist_rows += (
            f'<div style="display:flex;align-items:center;gap:0.5rem;'
            f'padding:0.35rem 0;border-bottom:1px solid #f1f5f9;">'
            f'<span style="width:10px;height:10px;border-radius:50%;'
            f'background:{color};flex-shrink:0;"></span>'
            f'<span style="flex:1;font-size:0.78rem;color:#374151;">{short}</span>'
            f'<span style="font-size:0.9rem;font-weight:700;color:#111827;">{count}</span>'
            f'<span style="font-size:0.72rem;color:#94a3b8;">prov</span>'
            f'</div>'
        )

    p3.markdown(
        f"""
        <div class="kpi-card" style="border-top-color:#f59e0b">
            <div class="kpi-label">Distribusi Provinsi</div>
            <div style="margin:0.6rem 0 0.2rem">{dist_rows}</div>
            <div class="kpi-sub">Jumlah anggota per kategori risiko</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sec("Evaluasi Silhouette Score")

    gc1, gc2, gc3 = st.columns([1, 2, 1])

    with gc2:
        st.plotly_chart(
            plot_silhouette_gauge(sil_score),
            use_container_width=True,
        )

    sec("Peta Segmentasi Wilayah K-Means (K=3)")

    st.plotly_chart(
        plot_kmeans_clusters(df),
        use_container_width=True,
    )

    sec("Rincian Anggota Tiap Kluster")

    risk_order = [
        "Kluster 3: Risiko Sangat Tinggi (Prioritas)",
        "Kluster 2: Risiko Sedang",
        "Kluster 1: Risiko Rendah (Kondisi Baik)",
    ]

    tabs = st.tabs([
        "🔴 Risiko Sangat Tinggi",
        "🟡 Risiko Sedang",
        "🟢 Risiko Rendah",
    ])

    for tab, rkey in zip(tabs, risk_order):

        sub = df[
            df["Kategori_Risiko"] == rkey
        ][
            [
                "Provinsi",
                "Kemiskinan_Pct",
                "AHH_Tahun",
                "Kemiskinan_Norm",
                "AHH_Norm",
            ]
        ].sort_values(
            "Kemiskinan_Pct",
            ascending=False,
        ).reset_index(drop=True)

        sub.index += 1

        with tab:
            a, b, c = st.columns(3)

            a.metric("Jumlah Provinsi", len(sub))

            avg_poverty = (
                f"{sub['Kemiskinan_Pct'].mean():.2f}%"
                if not sub.empty else "—"
            )
            avg_ahh = (
                f"{sub['AHH_Tahun'].mean():.2f} tahun"
                if not sub.empty else "—"
            )

            b.metric("Rata-rata Kemiskinan", avg_poverty)
            c.metric("Rata-rata AHH", avg_ahh)

            if sub.empty:
                st.info(
                    "Belum ada provinsi pada kategori risiko ini."
                )
            else:
                sub = sub.rename(columns={
                    "Kemiskinan_Pct": "Kemiskinan (%)",
                    "AHH_Tahun": "AHH (Tahun)",
                })

                st.dataframe(
                    sub.style.format({
                        "Kemiskinan (%)": "{:.2f}",
                        "AHH (Tahun)": "{:.2f}",
                        "Kemiskinan_Norm": "{:.4f}",
                        "AHH_Norm": "{:.4f}",
                    }),
                    use_container_width=True,
                )


# ============================================================
# MODUL 5 — SIMULATOR PREDIKSI
# ============================================================
elif menu == "Simulator Prediksi":

    sec("Kalkulator Estimasi AHH & Kategori Risiko")

    st.markdown("""
    <div class="info-box">
        Masukkan persentase penduduk miskin melalui bilah geser.
        Sistem akan mengestimasi <b>Angka Harapan Hidup (AHH)</b>
        menggunakan Regresi Linear dan menentukan kategori risiko
        menggunakan K-Means Clustering.
    </div>
    """, unsafe_allow_html=True)

    sa, sb = st.columns([3, 1])

    with sa:
        val_poverty = st.slider(
            "Persentase Penduduk Miskin (%)",
            min_value=0.0,
            max_value=40.0,
            value=10.0,
            step=0.1,
            format="%.1f%%",
        )

    with sb:
        st.markdown(
            f"""
            <div class="input-display">
                <div class="input-display-label">Nilai Input</div>
                <div class="input-display-value">
                    {val_poverty:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Prediksi regresi
    pred_ahh = float(
        lr_model.predict([[val_poverty]])[0]
    )

    # Normalisasi menggunakan rentang data pelatihan
    k_min_v = df["Kemiskinan_Pct"].min()
    k_max_v = df["Kemiskinan_Pct"].max()

    ahh_min_v = df["AHH_Tahun"].min()
    ahh_max_v = df["AHH_Tahun"].max()

    pov_range = k_max_v - k_min_v
    ahh_range = ahh_max_v - ahh_min_v

    pov_norm = (
        float(np.clip(
            (val_poverty - k_min_v) / pov_range, 0, 1
        ))
        if pov_range != 0 else 0.0
    )

    ahh_norm_ = (
        float(np.clip(
            (pred_ahh - ahh_min_v) / ahh_range, 0, 1
        ))
        if ahh_range != 0 else 0.0
    )

    cluster_raw = int(
        kmeans_model.predict([[pov_norm, ahh_norm_]])[0]
    )

    # Pemetaan kluster berdasarkan urutan centroid kemiskinan.
    # Mengikuti pendekatan yang digunakan pada kode awal.
    centroids = kmeans_model.cluster_centers_

    sorted_by_c = sorted(
        range(len(centroids)),
        key=lambda i: centroids[i][0],
    )

    raw_to_label = {
        sorted_by_c[0]: (
            "Kluster 1: Risiko Rendah (Kondisi Baik)",
            "badge-low",
            "🟢",
            "#16a34a",
        ),
        sorted_by_c[1]: (
            "Kluster 2: Risiko Sedang",
            "badge-med",
            "🟡",
            "#f59e0b",
        ),
        sorted_by_c[2]: (
            "Kluster 3: Risiko Sangat Tinggi (Prioritas)",
            "badge-high",
            "🔴",
            "#ef4444",
        ),
    }

    kat_label, badge_cls, risk_emoji, accent_color = (
        raw_to_label[cluster_raw]
    )

    # Kartu hasil
    sec("Hasil Estimasi")

    r1, r2 = st.columns(2)

    sign_str = "−" if lr_metrics["coef"] < 0 else "+"

    with r1:
        st.markdown(
            f"""
            <div class="sim-card" style="border-top-color:#16a34a">
                <div class="sim-card-label">
                    Estimasi Angka Harapan Hidup
                </div>
                <div class="sim-val" style="color:#16a34a">
                    {pred_ahh:.2f}
                </div>
                <div class="sim-unit">Tahun</div>
                <div class="sim-formula">
                    AHH = {lr_metrics['intercept']:.4f}
                    {sign_str}
                    {abs(lr_metrics['coef']):.4f}
                    × {val_poverty:.1f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2:
        st.markdown(
            f"""
            <div class="sim-card"
                 style="border-top-color:{accent_color}">
                <div class="sim-card-label">
                    Kategori Risiko Sosio-Kesehatan
                </div>
                <div style="font-size:2.4rem;line-height:1.2">
                    {risk_emoji}
                </div>
                <span class="badge {badge_cls}">
                    {kat_label}
                </span>
                <div class="sim-formula">
                    Titik Normalisasi:
                    [{pov_norm:.3f}, {ahh_norm_:.3f}]
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    sec("Kalkulasi Detail")

    latex_sign = "-" if lr_metrics["coef"] < 0 else "+"

    st.latex(
        rf"\hat{{Y}} = {lr_metrics['intercept']:.4f}"
        rf"\; {latex_sign} \;"
        rf"{abs(lr_metrics['coef']):.4f}"
        rf"\times {val_poverty:.1f}"
        rf"= {pred_ahh:.4f}\text{{ Tahun}}"
    )

    sec("Konteks vs Data 38 Provinsi")

    n_below = int(
        (df["Kemiskinan_Pct"] <= val_poverty).sum()
    )

    st.info(
        f"Input kemiskinan {val_poverty:.1f}% berada di atas "
        f"atau setara dengan {n_below} dari {len(df)} provinsi "
        f"({n_below / len(df) * 100:.1f}%) dalam dataset BPS 2024."
    )

    df_ctx = df.copy()

    df_ctx["Δ Kemiskinan"] = (
        df_ctx["Kemiskinan_Pct"] - val_poverty
    ).abs()

    nearest = df_ctx.nsmallest(
        8, "Δ Kemiskinan"
    )[
        [
            "Provinsi",
            "Kemiskinan_Pct",
            "AHH_Tahun",
            "Kategori_Risiko",
            "Δ Kemiskinan",
        ]
    ].reset_index(drop=True)

    nearest.index += 1

    st.caption(
        "8 provinsi terdekat berdasarkan persentase kemiskinan:"
    )

    nearest = nearest.rename(columns={
        "Kemiskinan_Pct": "Kemiskinan (%)",
        "AHH_Tahun": "AHH Aktual (Thn)",
    })

    st.dataframe(
        nearest.style.format({
            "Kemiskinan (%)": "{:.2f}",
            "AHH Aktual (Thn)": "{:.2f}",
            "Δ Kemiskinan": "{:.2f}",
        }),
        use_container_width=True,
    )
