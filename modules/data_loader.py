import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_excel(file_path="data/bps_2024_clean.xlsx"):
    """
    Membaca file .xlsx BPS 2024 dari folder data/ dan melakukan Min-Max Scaling.
    Kolom yang diharapkan: Provinsi, Kemiskinan_Pct, AHH_Tahun
    Kolom normalisasi yang dibuat: Kemiskinan_Norm, AHH_Norm

    Returns:
        df (DataFrame): Data lengkap dengan kolom normalisasi tambahan.
        scaler (MinMaxScaler): Objek scaler yang sudah di-fit.
    """
    df = pd.read_excel(file_path)

    # Pastikan kolom yang dibutuhkan ada
    required_cols = ["Provinsi", "Kemiskinan_Pct", "AHH_Tahun"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ditemukan di file Excel. "
                f"Kolom yang tersedia: {df.columns.tolist()}"
            )

    # Hapus baris dengan nilai NaN pada kolom kritis
    df = df.dropna(subset=["Kemiskinan_Pct", "AHH_Tahun"]).reset_index(drop=True)

    # Min-Max Normalization (Skala 0 – 1)
    # Jika kolom normalisasi belum ada di Excel, hitung ulang dari nilai aktual
    scaler = MinMaxScaler()
    df[["Kemiskinan_Norm", "AHH_Norm"]] = scaler.fit_transform(
        df[["Kemiskinan_Pct", "AHH_Tahun"]]
    )

    return df, scaler