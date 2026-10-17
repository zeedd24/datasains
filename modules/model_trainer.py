import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score


def train_linear_regression(df):
    """
    Melatih model Linear Regression untuk memprediksi AHH berdasarkan Kemiskinan.

    Returns:
        model: Objek LinearRegression yang sudah dilatih.
        y_pred (ndarray): Array prediksi AHH.
        metrics (dict): Kamus berisi r2, mae, rmse, intercept, coef.
    """
    X = df[["Kemiskinan_Pct"]]
    y = df["AHH_Tahun"]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = {
        "r2":        r2_score(y, y_pred),
        "mae":       mean_absolute_error(y, y_pred),
        "rmse":      np.sqrt(mean_squared_error(y, y_pred)),
        "intercept": model.intercept_,
        "coef":      model.coef_[0],
    }

    return model, y_pred, metrics


def train_kmeans_clustering(df, n_clusters=3):
    """
    Melatih model K-Means Clustering pada variabel ter-normalisasi (Kemiskinan_Norm, AHH_Norm).
    Memetakan label kluster secara dinamis berdasarkan rata-rata kemiskinan:
      • Kemiskinan RENDAH  → Kluster 1: Risiko Rendah (Kondisi Baik)
      • Kemiskinan SEDANG  → Kluster 2: Risiko Sedang
      • Kemiskinan TINGGI  → Kluster 3: Risiko Sangat Tinggi (Prioritas)

    Returns:
        kmeans: Objek KMeans yang sudah dilatih.
        df: DataFrame dengan kolom Cluster_Id dan Kategori_Risiko.
        sil (float): Silhouette Score.
    """
    X_cluster = df[["Kemiskinan_Norm", "AHH_Norm"]]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw_labels = kmeans.fit_predict(X_cluster)
    df = df.copy()
    df["_raw_cluster"] = raw_labels

    # Urutan kluster: rata-rata kemiskinan terendah → Risiko Rendah, tertinggi → Risiko Sangat Tinggi
    cluster_poverty_mean = (
        df.groupby("_raw_cluster")["Kemiskinan_Pct"]
        .mean()
        .sort_values()
    )

    risk_labels = [
        "Kluster 1: Risiko Rendah (Kondisi Baik)",
        "Kluster 2: Risiko Sedang",
        "Kluster 3: Risiko Sangat Tinggi (Prioritas)",
    ]
    risk_map   = {cluster_id: label for cluster_id, label in zip(cluster_poverty_mean.index, risk_labels)}
    id_map     = {cluster_id: idx   for idx, cluster_id in enumerate(cluster_poverty_mean.index)}

    df["Cluster_Id"]     = df["_raw_cluster"].map(id_map)
    df["Kategori_Risiko"] = df["_raw_cluster"].map(risk_map)
    df.drop(columns=["_raw_cluster"], inplace=True)

    sil = silhouette_score(X_cluster, raw_labels)

    return kmeans, df, sil