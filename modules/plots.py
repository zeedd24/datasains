import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ── Design Tokens — Green & White Theme ───────────────────────────────────
C_LOW   = "#16a34a"   # green-600  – risiko rendah
C_MED   = "#d97706"   # amber-600  – risiko sedang
C_HIGH  = "#dc2626"   # red-600    – risiko tinggi
C_REG   = "#2563eb"   # blue-600   – scatter aktual
C_TREND = "#16a34a"   # green-600  – garis regresi

COLOR_MAP = {
    "Kluster 1: Risiko Rendah (Kondisi Baik)":     C_LOW,
    "Kluster 2: Risiko Sedang":                    C_MED,
    "Kluster 3: Risiko Sangat Tinggi (Prioritas)": C_HIGH,
}

BG        = "rgba(0,0,0,0)"
GRID      = "rgba(0,0,0,0.05)"
FONT_CLR  = "#64748b"
TITLE_CLR = "#14532d"

_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor="rgba(255,255,255,0.7)",
    font=dict(family="Inter, sans-serif", color=FONT_CLR, size=12),
    margin=dict(l=10, r=10, t=48, b=10),
    xaxis=dict(
        gridcolor=GRID,
        zerolinecolor="rgba(0,0,0,0.08)",
        linecolor="rgba(0,0,0,0.08)",
        tickfont=dict(size=11, color=FONT_CLR),
        title_font=dict(size=12, color=FONT_CLR),
    ),
    yaxis=dict(
        gridcolor=GRID,
        zerolinecolor="rgba(0,0,0,0.08)",
        linecolor="rgba(0,0,0,0.08)",
        tickfont=dict(size=11, color=FONT_CLR),
        title_font=dict(size=12, color=FONT_CLR),
    ),
    title_font=dict(size=14, color=TITLE_CLR, family="Inter, sans-serif"),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(22,163,74,0.2)",
        borderwidth=1,
        font=dict(size=11, color=FONT_CLR),
    ),
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="rgba(22,163,74,0.3)",
        font=dict(size=12, color="#1e293b"),
    ),
)


# ── 1. Top-5 Bar Charts ────────────────────────────────────────────────────
def plot_top_5_bar(df):
    top_pov = df.nlargest(5, "Kemiskinan_Pct").sort_values("Kemiskinan_Pct")
    top_ahh = df.nlargest(5, "AHH_Tahun").sort_values("AHH_Tahun")

    fig1 = px.bar(
        top_pov, x="Kemiskinan_Pct", y="Provinsi", orientation="h",
        color="Kemiskinan_Pct",
        color_continuous_scale=["#fca5a5", "#ef4444", "#7f1d1d"],
        text="Kemiskinan_Pct",
        title="Top 5 — Kemiskinan Tertinggi (%)",
        labels={"Kemiskinan_Pct": "Kemiskinan (%)"},
    )
    fig1.update_traces(
        texttemplate="%{x:.2f}%",
        textposition="outside",
        textfont=dict(color=FONT_CLR, size=11),
        marker_line_width=0,
    )
    fig1.update_coloraxes(showscale=False)
    fig1.update_layout(**_BASE, xaxis_title="Persentase Kemiskinan (%)", yaxis_title=None)

    fig2 = px.bar(
        top_ahh, x="AHH_Tahun", y="Provinsi", orientation="h",
        color="AHH_Tahun",
        color_continuous_scale=["#bbf7d0", "#16a34a", "#14532d"],
        text="AHH_Tahun",
        title="Top 5 — AHH Tertinggi (Tahun)",
        labels={"AHH_Tahun": "AHH (Tahun)"},
    )
    fig2.update_traces(
        texttemplate="%{x:.2f}",
        textposition="outside",
        textfont=dict(color=FONT_CLR, size=11),
        marker_line_width=0,
    )
    fig2.update_coloraxes(showscale=False)
    fig2.update_layout(**_BASE, xaxis_title="Angka Harapan Hidup (Tahun)", yaxis_title=None)

    return fig1, fig2


# ── 2. Scatter Plot Sebaran Data Asli ─────────────────────────────────────
def plot_raw_scatter(df):
    fig = px.scatter(
        df, x="Kemiskinan_Pct", y="AHH_Tahun",
        hover_name="Provinsi",
        title="Sebaran Data Asli — 38 Provinsi",
        labels={"Kemiskinan_Pct": "Kemiskinan (%)", "AHH_Tahun": "AHH (Tahun)"},
    )
    fig.update_traces(
        marker=dict(
            size=10, color=C_REG, opacity=0.8,
            line=dict(width=1, color="rgba(37,99,235,0.35)"),
        ),
        hovertemplate="<b>%{hovertext}</b><br>Kemiskinan: %{x:.2f}%<br>AHH: %{y:.2f} Thn<extra></extra>",
    )
    fig.update_layout(**_BASE, xaxis_title="Persentase Kemiskinan (%)", yaxis_title="AHH (Tahun)")
    return fig


# ── 3. Scatter + Garis Regresi ────────────────────────────────────────────
def plot_regression_scatter(df, lr_metrics, y_pred):
    x_line = np.linspace(df["Kemiskinan_Pct"].min(), df["Kemiskinan_Pct"].max(), 300)
    y_line = lr_metrics["intercept"] + lr_metrics["coef"] * x_line

    fig = go.Figure()

    # Residual bars (subtle)
    for _, row in df.iterrows():
        fig.add_shape(
            type="line",
            x0=row["Kemiskinan_Pct"], y0=row["AHH_Tahun"],
            x1=row["Kemiskinan_Pct"], y1=lr_metrics["intercept"] + lr_metrics["coef"] * row["Kemiskinan_Pct"],
            line=dict(color="rgba(22,163,74,0.2)", width=1, dash="dot"),
        )

    # Titik data aktual
    fig.add_trace(go.Scatter(
        x=df["Kemiskinan_Pct"], y=df["AHH_Tahun"],
        mode="markers",
        name="Data Aktual",
        text=df["Provinsi"],
        hovertemplate="<b>%{text}</b><br>Kemiskinan: %{x:.2f}%<br>AHH Aktual: %{y:.2f} Thn<extra></extra>",
        marker=dict(size=10, color=C_REG, opacity=0.85, line=dict(width=1, color="rgba(37,99,235,0.4)")),
    ))

    # Garis regresi
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        name="Garis Regresi OLS",
        line=dict(color=C_TREND, width=2.5),
        hoverinfo="skip",
    ))

    # Titik prediksi
    fig.add_trace(go.Scatter(
        x=df["Kemiskinan_Pct"], y=y_pred,
        mode="markers",
        name="Nilai Prediksi",
        text=df["Provinsi"],
        hovertemplate="<b>%{text}</b><br>Kemiskinan: %{x:.2f}%<br>AHH Prediksi: %{y:.2f} Thn<extra></extra>",
        marker=dict(size=7, color=C_TREND, symbol="diamond", opacity=0.7,
                    line=dict(width=1, color="rgba(22,163,74,0.4)")),
    ))

    fig.update_layout(
        **_BASE,
        title="Scatter Plot & Garis Regresi Linear — Kemiskinan vs AHH",
        xaxis_title="Persentase Kemiskinan (%)",
        yaxis_title="AHH (Tahun)",
    )
    return fig


# ── 4. K-Means Cluster Scatter ────────────────────────────────────────────
def plot_kmeans_clusters(df):
    fig = px.scatter(
        df,
        x="Kemiskinan_Pct", y="AHH_Tahun",
        color="Kategori_Risiko",
        color_discrete_map=COLOR_MAP,
        hover_name="Provinsi",
        size="Kemiskinan_Pct",
        size_max=28,
        title="Segmentasi Risiko Sosio-Kesehatan — K-Means (K=3)",
        labels={
            "Kemiskinan_Pct":  "Kemiskinan (%)",
            "AHH_Tahun":       "AHH (Tahun)",
            "Kategori_Risiko": "Kategori Risiko",
        },
    )
    fig.update_traces(
        marker=dict(opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.6)")),
        hovertemplate="<b>%{hovertext}</b><br>Kemiskinan: %{x:.2f}%<br>AHH: %{y:.2f} Thn<extra></extra>",
    )
    fig.update_layout(
        **_BASE,
        xaxis_title="Persentase Kemiskinan (%)",
        yaxis_title="AHH (Tahun)",
        legend_title_text="Kategori Risiko",
    )
    return fig


# ── 5. Silhouette Gauge ───────────────────────────────────────────────────
def plot_silhouette_gauge(sil_score):
    bar_color = C_LOW if sil_score >= 0.5 else C_MED
    num_color = C_LOW if sil_score >= 0.5 else C_MED

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(sil_score, 4),
        delta={"reference": 0.5, "increasing": {"color": C_LOW}, "decreasing": {"color": C_HIGH}},
        title={
            "text": "Silhouette Score<br><span style='font-size:0.75em;color:#94a3b8'>Kualitas Kluster K-Means</span>",
            "font": {"color": TITLE_CLR, "size": 14},
        },
        number={"font": {"color": num_color, "size": 36, "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 1],
                "tickvals": [0, 0.25, 0.5, 0.75, 1],
                "ticktext": ["0", "Buruk", "Cukup", "Baik", "1"],
                "tickcolor": FONT_CLR,
                "tickfont": {"color": FONT_CLR, "size": 10},
            },
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0.00, 0.25], "color": "rgba(220,38,38,0.1)"},
                {"range": [0.25, 0.50], "color": "rgba(217,119,6,0.1)"},
                {"range": [0.50, 0.75], "color": "rgba(22,163,74,0.1)"},
                {"range": [0.75, 1.00], "color": "rgba(22,163,74,0.2)"},
            ],
            "threshold": {
                "line": {"color": C_HIGH, "width": 2},
                "thickness": 0.85,
                "value": sil_score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color=FONT_CLR),
        margin=dict(l=40, r=40, t=60, b=10),
        height=250,
    )
    return fig