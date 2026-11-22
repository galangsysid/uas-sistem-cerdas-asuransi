"""
Mesin Machine Learning dan Fuzzy Logic.

Berisi dua komponen utama:
1. Prediksi biaya asuransi menggunakan pipeline scikit-learn yang sudah dilatih
   (ml_training/model_asuransi.joblib), dimuat satu kali saat modul diimpor.
2. Sistem Inferensi Fuzzy (FIS) Mamdani dengan dua input (BMI dan usia) yang
   ditulis dengan Python dan NumPy murni, tanpa library fuzzy eksternal.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml_training" / "model_asuransi.joblib"

# Urutan kolom ini HARUS identik dengan KOLOM_FITUR pada ml_training/train_model.py
URUTAN_KOLOM = ["age", "sex", "bmi", "children", "smoker", "region"]

PILIHAN_SEX = ("male", "female")
PILIHAN_SMOKER = ("yes", "no")
PILIHAN_REGION = ("northeast", "northwest", "southeast", "southwest")

BATAS_USIA = (17, 100)
BATAS_BMI = (10.0, 60.0)
BATAS_ANAK = (0, 10)


# ---------------------------------------------------------------------------
# Bagian 1: Pemuatan model dan prediksi biaya
# ---------------------------------------------------------------------------

# Model dimuat sekali saja saat modul ini pertama kali diimpor.
MODEL_PIPELINE = None
PESAN_ERROR_MODEL = ""

try:
    MODEL_PIPELINE = joblib.load(MODEL_PATH)
except FileNotFoundError:
    PESAN_ERROR_MODEL = (
        f"File model tidak ditemukan di '{MODEL_PATH}'. "
        "Jalankan 'python ml_training/train_model.py' terlebih dahulu."
    )
except Exception as error:  # Contoh: versi scikit-learn berbeda dengan saat pelatihan
    PESAN_ERROR_MODEL = (
        f"Model gagal dimuat dari '{MODEL_PATH}': {error}. "
        "Latih ulang model dengan versi scikit-learn yang sama seperti di server."
    )


def validasi_input(age, sex, bmi, children, smoker, region) -> dict:
    """Memvalidasi dan menormalkan masukan pengguna. Melempar ValueError jika tidak valid."""
    try:
        age = int(age)
        children = int(children)
        bmi = float(bmi)
    except (TypeError, ValueError):
        raise ValueError("Usia, BMI, dan jumlah anak harus berupa angka.")

    sex = str(sex).strip().lower()
    smoker = str(smoker).strip().lower()
    region = str(region).strip().lower()

    if not BATAS_USIA[0] <= age <= BATAS_USIA[1]:
        raise ValueError(f"Usia harus antara {BATAS_USIA[0]} dan {BATAS_USIA[1]} tahun.")
    if not BATAS_BMI[0] <= bmi <= BATAS_BMI[1]:
        raise ValueError(f"BMI harus antara {BATAS_BMI[0]:.0f} dan {BATAS_BMI[1]:.0f}.")
    if not BATAS_ANAK[0] <= children <= BATAS_ANAK[1]:
        raise ValueError(f"Jumlah anak harus antara {BATAS_ANAK[0]} dan {BATAS_ANAK[1]}.")
    if sex not in PILIHAN_SEX:
        raise ValueError(f"Jenis kelamin harus salah satu dari: {', '.join(PILIHAN_SEX)}.")
    if smoker not in PILIHAN_SMOKER:
        raise ValueError(f"Status perokok harus salah satu dari: {', '.join(PILIHAN_SMOKER)}.")
    if region not in PILIHAN_REGION:
        raise ValueError(f"Wilayah harus salah satu dari: {', '.join(PILIHAN_REGION)}.")

    return {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }


def prediksi_biaya(age, sex, bmi, children, smoker, region) -> float:
    """Memprediksi biaya asuransi kesehatan (dalam USD) berdasarkan profil pengguna."""
    if MODEL_PIPELINE is None:
        raise RuntimeError(PESAN_ERROR_MODEL)

    data = validasi_input(age, sex, bmi, children, smoker, region)

    # Kolom disusun ulang sesuai URUTAN_KOLOM agar identik dengan data latih.
    df_input = pd.DataFrame([data], columns=URUTAN_KOLOM)
    hasil = MODEL_PIPELINE.predict(df_input)[0]
    return round(float(hasil), 2)


# ---------------------------------------------------------------------------
# Bagian 2: Sistem Inferensi Fuzzy Mamdani (BMI x Usia -> Skor Risiko)
# ---------------------------------------------------------------------------

def fungsi_trapesium(x, a, b, c, d):
    """
    Fungsi keanggotaan trapesium dengan titik (a, b, c, d).

    Nilai 0 di luar [a, d], naik linear dari a ke b, bernilai 1 di antara b dan c,
    turun linear dari c ke d. Jika b == c bentuknya menjadi segitiga. Jika a == b
    atau c == d, sisi tersebut menjadi bahu (shoulder) yang langsung bernilai 1.
    """
    x = np.asarray(x, dtype=float)
    naik = (x - a) / (b - a) if b > a else np.ones_like(x)
    turun = (d - x) / (d - c) if d > c else np.ones_like(x)
    return np.clip(np.minimum(naik, turun), 0.0, 1.0)


# Himpunan keanggotaan input: (a, b, c, d)
HIMPUNAN_BMI = {
    "Kurus": (10.0, 10.0, 17.0, 19.0),
    "Normal": (17.5, 19.5, 23.0, 26.0),
    "Gemuk": (23.5, 26.5, 28.5, 31.5),
    "Obesitas": (28.5, 32.0, 60.0, 60.0),
}

HIMPUNAN_USIA = {
    "Muda": (0.0, 0.0, 25.0, 38.0),
    "Dewasa": (28.0, 38.0, 48.0, 58.0),
    "Lansia": (48.0, 60.0, 120.0, 120.0),
}

# Himpunan keanggotaan output (skor risiko 0 sampai 100)
HIMPUNAN_RISIKO = {
    "Rendah": (0.0, 0.0, 20.0, 45.0),
    "Sedang": (30.0, 50.0, 50.0, 70.0),
    "Tinggi": (55.0, 80.0, 100.0, 100.0),
}

# Basis aturan IF-THEN lengkap: seluruh 4 x 3 = 12 kombinasi BMI dan Usia.
ATURAN_FUZZY = [
    # (BMI, Usia, Risiko)
    ("Kurus", "Muda", "Rendah"),
    ("Kurus", "Dewasa", "Rendah"),
    ("Kurus", "Lansia", "Sedang"),
    ("Normal", "Muda", "Rendah"),
    ("Normal", "Dewasa", "Rendah"),
    ("Normal", "Lansia", "Sedang"),
    ("Gemuk", "Muda", "Rendah"),
    ("Gemuk", "Dewasa", "Sedang"),
    ("Gemuk", "Lansia", "Tinggi"),
    ("Obesitas", "Muda", "Sedang"),
    ("Obesitas", "Dewasa", "Tinggi"),
    ("Obesitas", "Lansia", "Tinggi"),
]

# Batas kategori berdasarkan skor hasil defuzzifikasi.
BATAS_RENDAH = 35.0
BATAS_SEDANG = 65.0

# Semesta pembicaraan output dan kurva keanggotaannya dihitung sekali saja.
SEMESTA_RISIKO = np.linspace(0.0, 100.0, 1001)
KURVA_RISIKO = {
    label: fungsi_trapesium(SEMESTA_RISIKO, *titik)
    for label, titik in HIMPUNAN_RISIKO.items()
}


def fuzzifikasi(nilai: float, himpunan: dict) -> dict:
    """Mengubah nilai tegas menjadi derajat keanggotaan pada setiap himpunan."""
    return {
        label: float(fungsi_trapesium(nilai, *titik))
        for label, titik in himpunan.items()
    }


def tentukan_kategori(skor: float) -> str:
    """Menentukan kategori risiko dari skor 0 sampai 100."""
    if skor <= BATAS_RENDAH:
        return "Rendah"
    if skor <= BATAS_SEDANG:
        return "Sedang"
    return "Tinggi"


def hitung_risiko_fuzzy(bmi: float, usia: float) -> dict:
    """
    Menghitung skor risiko dengan metode Mamdani.

    Tahapan:
    1. Fuzzifikasi BMI dan usia.
    2. Evaluasi aturan: kekuatan aturan = min(derajat BMI, derajat usia).
    3. Implikasi: kurva output dipotong (clip) setinggi kekuatan aturan.
    4. Agregasi: gabungan seluruh kurva dengan operator max.
    5. Defuzzifikasi: titik pusat (centroid).
    """
    bmi = float(bmi)
    usia = float(usia)

    derajat_bmi = fuzzifikasi(bmi, HIMPUNAN_BMI)
    derajat_usia = fuzzifikasi(usia, HIMPUNAN_USIA)

    # Kekuatan tertinggi untuk setiap label output (agregasi aturan dengan max)
    kekuatan_output = {label: 0.0 for label in HIMPUNAN_RISIKO}
    aturan_aktif = []

    for label_bmi, label_usia, label_risiko in ATURAN_FUZZY:
        kekuatan = min(derajat_bmi[label_bmi], derajat_usia[label_usia])
        if kekuatan <= 0.0:
            continue
        aturan_aktif.append(
            {
                "bmi": label_bmi,
                "usia": label_usia,
                "risiko": label_risiko,
                "kekuatan": round(kekuatan, 4),
            }
        )
        kekuatan_output[label_risiko] = max(kekuatan_output[label_risiko], kekuatan)

    # Agregasi kurva output yang sudah dipotong
    agregat = np.zeros_like(SEMESTA_RISIKO)
    for label, kekuatan in kekuatan_output.items():
        if kekuatan > 0.0:
            agregat = np.maximum(agregat, np.minimum(KURVA_RISIKO[label], kekuatan))

    total_luas = float(np.sum(agregat))
    if total_luas <= 0.0:
        skor = 50.0  # Cadangan jika tidak ada aturan yang aktif
    else:
        skor = float(np.sum(SEMESTA_RISIKO * agregat) / total_luas)

    skor = round(skor, 2)
    aturan_aktif.sort(key=lambda r: r["kekuatan"], reverse=True)

    return {
        "skor_fuzzy": skor,
        "kategori_risiko": tentukan_kategori(skor),
        "derajat_bmi": {k: round(v, 4) for k, v in derajat_bmi.items()},
        "derajat_usia": {k: round(v, 4) for k, v in derajat_usia.items()},
        "aturan_aktif": aturan_aktif,
    }


# ---------------------------------------------------------------------------
# Bagian 3: Fungsi gabungan untuk dipanggil dari views.py
# ---------------------------------------------------------------------------

def analisis_lengkap(age, sex, bmi, children, smoker, region) -> dict:
    """Menjalankan prediksi biaya dan inferensi fuzzy sekaligus."""
    data = validasi_input(age, sex, bmi, children, smoker, region)

    biaya = prediksi_biaya(**data)
    fuzzy = hitung_risiko_fuzzy(data["bmi"], data["age"])

    return {
        **data,
        "prediksi_biaya": biaya,
        "skor_fuzzy": fuzzy["skor_fuzzy"],
        "kategori_risiko": fuzzy["kategori_risiko"],
        "detail_fuzzy": fuzzy,
    }