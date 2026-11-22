# Sistem Prediksi Biaya Asuransi Kesehatan & Kategori Risiko

Aplikasi web berbasis Django untuk memprediksi estimasi biaya asuransi kesehatan tahunan menggunakan Machine Learning, mengevaluasi tingkat risiko kesehatan melalui Sistem Inferensi Fuzzy (FIS) 2-Input, memvisualisasikan perbandingan data dengan Matplotlib, serta menghasilkan interpretasi otomatis menggunakan Groq AI.

## Sumber dan Lisensi Dataset

- **Nama Dataset:** Medical Cost Personal Dataset (`insurance.csv`)
- **Sumber:** Kaggle Open Datasets (Miri Choi / Brett Lantz, *Machine Learning with R*)
- **Jumlah Data:** 1.338 baris, 7 kolom (`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`)
- **Lisensi:** CC0: Public Domain (bebas digunakan untuk keperluan akademik dan pengembangan)

## Arsitektur Sistem

1. **Framework Web & Database:** Django dengan database SQLite (`RiwayatPrediksi`) untuk menyimpan histori analisis pengguna.
2. **Machine Learning (Regresi):** Pipeline `scikit-learn` (`ColumnTransformer` + `RandomForestRegressor`) yang dilatih secara offline dan disimpan ke `ml_training/model_asuransi.joblib`.
3. **Fuzzy Inference System (FIS Mamdani 2-Input):** Implementasi Python & NumPy murni yang mengolah input **BMI** (Kurus, Normal, Gemuk, Obesitas) dan **Usia** (Muda, Dewasa, Lansia) menjadi skor risiko 0–100 dan kategori (`Rendah`, `Sedang`, `Tinggi`).
4. **Visualisasi & Interpretasi AI:** Grafik dua panel dirender menggunakan Matplotlib backend `Agg` ke format Base64, kemudian data statistiknya dianalisis oleh Groq API (`llama-3.3-70b-versatile`) dengan mekanisme fallback statistik otomatis.

## Cara Menjalankan di Lokal

1. Aktifkan virtual environment dan instal dependensi:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt