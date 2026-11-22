"""
Pelatihan model prediksi biaya asuransi kesehatan.

Cara menjalankan (dari folder root proyek):
    python ml_training/train_model.py

Keluaran:
    ml_training/model_asuransi.joblib  (pipeline lengkap: preprocessing + model)
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "insurance.csv"
MODEL_PATH = BASE_DIR / "ml_training" / "model_asuransi.joblib"

# Urutan kolom fitur ini HARUS identik dengan urutan di core_app/ml_fuzzy_engine.py
KOLOM_FITUR = ["age", "sex", "bmi", "children", "smoker", "region"]
KOLOM_TARGET = "charges"
KOLOM_KATEGORIKAL = ["sex", "smoker", "region"]
KOLOM_NUMERIK = ["age", "bmi", "children"]

RANDOM_STATE = 42
UKURAN_DATA_UJI = 0.2


def muat_dataset(path: Path) -> pd.DataFrame:
    """Membaca dataset dan memastikan seluruh kolom yang dibutuhkan tersedia."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di '{path}'. "
            "Letakkan file insurance.csv di folder dataset/."
        )

    df = pd.read_csv(path)

    kolom_wajib = KOLOM_FITUR + [KOLOM_TARGET]
    kolom_hilang = [k for k in kolom_wajib if k not in df.columns]
    if kolom_hilang:
        raise ValueError(f"Kolom berikut tidak ada di dataset: {kolom_hilang}")

    jumlah_kosong = int(df[kolom_wajib].isna().sum().sum())
    if jumlah_kosong > 0:
        print(f"Peringatan: ditemukan {jumlah_kosong} nilai kosong, baris terkait dibuang.")
        df = df.dropna(subset=kolom_wajib).reset_index(drop=True)

    return df


def bangun_pipeline() -> Pipeline:
    """Membangun pipeline: ColumnTransformer (encoding dan scaling) + RandomForestRegressor."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "kategorikal",
                OneHotEncoder(handle_unknown="ignore"),
                KOLOM_KATEGORIKAL,
            ),
            ("numerik", StandardScaler(), KOLOM_NUMERIK),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def latih_dan_evaluasi() -> Pipeline:
    df = muat_dataset(DATASET_PATH)
    print(f"Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom.")

    # Urutan kolom dikunci sesuai KOLOM_FITUR agar identik dengan saat prediksi.
    X = df[KOLOM_FITUR]
    y = df[KOLOM_TARGET]

    X_latih, X_uji, y_latih, y_uji = train_test_split(
        X, y, test_size=UKURAN_DATA_UJI, random_state=RANDOM_STATE
    )
    print(f"Data latih: {len(X_latih)} baris | Data uji: {len(X_uji)} baris.")

    pipeline = bangun_pipeline()
    pipeline.fit(X_latih, y_latih)

    y_prediksi = pipeline.predict(X_uji)
    skor_r2 = r2_score(y_uji, y_prediksi)
    skor_mae = mean_absolute_error(y_uji, y_prediksi)

    print("-" * 40)
    print("Evaluasi pada data uji")
    print(f"R2 score : {skor_r2:.4f}")
    print(f"MAE      : {skor_mae:,.2f}")
    print("-" * 40)

    return pipeline


def main() -> int:
    try:
        pipeline = latih_dan_evaluasi()
    except (FileNotFoundError, ValueError) as error:
        print(f"Gagal melatih model: {error}", file=sys.stderr)
        return 1

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline terlatih disimpan ke: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())