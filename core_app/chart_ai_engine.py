"""
Mesin visualisasi dan integrasi AI.

Berisi:
1. Pembuatan grafik dua panel dengan Matplotlib (backend Agg, tanpa GUI), dikembalikan
   sebagai string Base64 untuk ditampilkan di HTML.
2. Integrasi Groq API (SDK resmi groq) untuk menjelaskan makna grafik.
   Seluruh angka yang digambarkan pada grafik dikirim dalam bentuk data terstruktur.
3. Penjelasan cadangan (Fallback Statistik) jika API key kosong, tidak valid, limit habis,
   atau terjadi error lain.
"""

import base64
import io
import logging
import os
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Wajib dipanggil sebelum pyplot diimpor

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

try:
    from groq import Groq
except ImportError:  # Paket belum terpasang: aplikasi tetap jalan dengan fallback
    Groq = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "insurance.csv"

load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)

DAFTAR_MODEL_GROQ = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
TIMEOUT_GROQ_DETIK = 30.0

SUMBER_GROQ = "Groq AI"
SUMBER_FALLBACK = "Fallback Statistik"

BATAS_RENDAH = 35.0
BATAS_SEDANG = 65.0

# Warna grafik
WARNA_PENGGUNA = "#B45309"
WARNA_REFERENSI = "#898781"
WARNA_TEKS = "#0b0b0b"
WARNA_TEKS_SEKUNDER = "#52514e"
WARNA_GRID = "#e1e0d9"
WARNA_PERMUKAAN = "#fcfcfb"
WARNA_PITA_BMI = "#f0efec"

# Kategori BMI standar WHO untuk latar panel kedua
PITA_BMI = [
    ("Kurus", 0.0, 18.5),
    ("Normal", 18.5, 25.0),
    ("Gemuk", 25.0, 30.0),
    ("Obesitas", 30.0, 100.0),
]

LABEL_SEX = {"male": "laki-laki", "female": "perempuan"}


# ---------------------------------------------------------------------------
# Statistik pembanding dari dataset
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def statistik_dataset() -> dict:
    """Menghitung rata-rata biaya perokok, non-perokok, dan BMI dari dataset (di-cache)."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di '{DATASET_PATH}'.")

    df = pd.read_csv(DATASET_PATH)
    rata_biaya = df.groupby("smoker")["charges"].mean()
    return {
        "biaya_perokok": float(rata_biaya["yes"]),
        "biaya_non_perokok": float(rata_biaya["no"]),
        "bmi_rata_rata": float(df["bmi"].mean()),
    }


def kategori_bmi(bmi: float) -> str:
    """Kategori BMI standar WHO."""
    for nama, bawah, atas in PITA_BMI:
        if bawah <= bmi < atas:
            return nama
    return "Obesitas"


def _persen_selisih(nilai: float, acuan: float) -> float:
    return (nilai - acuan) / acuan * 100.0 if acuan else 0.0


def hitung_statistik_pembanding(ringkasan: dict) -> dict:
    """
    Menghitung seluruh angka perbandingan yang digambarkan pada grafik.

    Nilai persen positif berarti angka pengguna lebih tinggi dari acuan.
    """
    data = statistik_dataset()
    biaya = ringkasan["prediksi_biaya"]
    bmi = ringkasan["bmi"]

    return {
        "rata_biaya_perokok": data["biaya_perokok"],
        "rata_biaya_non_perokok": data["biaya_non_perokok"],
        "rata_bmi": data["bmi_rata_rata"],
        "selisih_vs_perokok": biaya - data["biaya_perokok"],
        "persen_vs_perokok": _persen_selisih(biaya, data["biaya_perokok"]),
        "selisih_vs_non_perokok": biaya - data["biaya_non_perokok"],
        "persen_vs_non_perokok": _persen_selisih(biaya, data["biaya_non_perokok"]),
        "selisih_bmi": bmi - data["bmi_rata_rata"],
        "kategori_bmi": kategori_bmi(bmi),
        "kategori_bmi_rata": kategori_bmi(data["bmi_rata_rata"]),
    }


def _format_usd(nilai: float) -> str:
    return f"USD {nilai:,.0f}"


def _label_perokok(smoker: str) -> str:
    return "perokok" if smoker == "yes" else "bukan perokok"


# ---------------------------------------------------------------------------
# Grafik dua panel
# ---------------------------------------------------------------------------

def buat_grafik(prediksi_biaya: float, bmi: float, skor_fuzzy: float, kategori_risiko: str) -> str:
    """
    Membuat grafik dua panel dan mengembalikannya sebagai string Base64 (PNG).

    Panel kiri : prediksi biaya pengguna dibandingkan rata-rata perokok dan non-perokok.
    Panel kanan: posisi BMI pengguna, rata-rata BMI dataset, dan skor risiko fuzzy.

    Gunakan di HTML: <img src="data:image/png;base64,{{ grafik_base64 }}">
    """
    statistik = statistik_dataset()
    rata_bmi = statistik["bmi_rata_rata"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    fig.patch.set_facecolor(WARNA_PERMUKAAN)

    for ax in (ax1, ax2):
        ax.set_facecolor(WARNA_PERMUKAAN)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=WARNA_TEKS_SEKUNDER, labelsize=10)

    # Panel 1: perbandingan biaya
    label_batang = ["Rata-rata\nnon-perokok", "Rata-rata\nperokok", "Prediksi\nAnda"]
    nilai_batang = [statistik["biaya_non_perokok"], statistik["biaya_perokok"], prediksi_biaya]
    warna_batang = [WARNA_REFERENSI, WARNA_REFERENSI, WARNA_PENGGUNA]

    batang = ax1.bar(label_batang, nilai_batang, color=warna_batang, width=0.55, zorder=3)
    ax1.set_title("Perbandingan biaya asuransi", fontsize=12, color=WARNA_TEKS, loc="left")
    ax1.set_ylabel("Biaya (USD)", color=WARNA_TEKS_SEKUNDER)
    ax1.yaxis.set_major_formatter(lambda nilai, _: f"{nilai:,.0f}")
    ax1.grid(axis="y", color=WARNA_GRID, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)

    batas_atas = max(nilai_batang) * 1.15
    ax1.set_ylim(0, batas_atas)
    for rect, nilai in zip(batang, nilai_batang):
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            nilai + batas_atas * 0.015,
            f"{nilai:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=WARNA_TEKS,
        )

    # Panel 2: posisi BMI dan skor risiko
    x_maks = max(50.0, bmi + 4.0)
    x_min = 10.0
    ax2.set_xlim(x_min, x_maks)
    ax2.set_ylim(0, 100)

    for indeks, (nama, bawah, atas) in enumerate(PITA_BMI):
        kiri = max(bawah, x_min)
        kanan = min(atas, x_maks)
        if kanan <= kiri:
            continue
        if indeks % 2 == 0:
            ax2.axvspan(kiri, kanan, color=WARNA_PITA_BMI, zorder=0)
        ax2.text(
            (kiri + kanan) / 2,
            97,
            nama,
            ha="center",
            va="top",
            fontsize=9,
            color=WARNA_TEKS_SEKUNDER,
        )

    for batas in (BATAS_RENDAH, BATAS_SEDANG):
        ax2.axhline(batas, color="#c3c2b7", linestyle="--", linewidth=1, zorder=1)

    # Garis rata-rata BMI dataset sebagai pembanding posisi BMI pengguna
    ax2.axvline(rata_bmi, color=WARNA_REFERENSI, linestyle=":", linewidth=1.6, zorder=2)
    di_sisi_kiri = rata_bmi > x_maks - 14
    ax2.text(
        rata_bmi + (-0.5 if di_sisi_kiri else 0.5),
        2,
        f"Rata-rata BMI dataset: {rata_bmi:.1f}",
        ha="right" if di_sisi_kiri else "left",
        va="bottom",
        fontsize=8.5,
        color=WARNA_TEKS_SEKUNDER,
    )

    # Label zona risiko diletakkan di luar sumbu kanan agar tidak bertabrakan dengan titik pengguna
    transform_label = ax2.get_yaxis_transform()
    zona_risiko = [
        ("Risiko\nRendah", BATAS_RENDAH / 2),
        ("Risiko\nSedang", (BATAS_RENDAH + BATAS_SEDANG) / 2),
        ("Risiko\nTinggi", (BATAS_SEDANG + 100) / 2),
    ]
    for teks_zona, posisi_y in zona_risiko:
        ax2.text(1.015, posisi_y, teks_zona, transform=transform_label, ha="left",
                 va="center", fontsize=9, color=WARNA_TEKS_SEKUNDER, clip_on=False)

    ax2.scatter(
        [bmi], [skor_fuzzy], s=150, color=WARNA_PENGGUNA,
        edgecolors=WARNA_PERMUKAAN, linewidths=2, zorder=5,
    )

    # Posisi teks anotasi disesuaikan agar tidak keluar area grafik
    di_kanan = bmi > (x_min + x_maks) / 2
    di_atas = skor_fuzzy > 80
    ax2.annotate(
        f"Anda: BMI {bmi:.1f}, skor {skor_fuzzy:.1f}\n({kategori_risiko})",
        xy=(bmi, skor_fuzzy),
        xytext=(-14 if di_kanan else 14, -26 if di_atas else 18),
        textcoords="offset points",
        ha="right" if di_kanan else "left",
        va="top" if di_atas else "bottom",
        fontsize=10,
        color=WARNA_TEKS,
        zorder=6,
    )

    ax2.set_title("Posisi BMI dan skor risiko fuzzy", fontsize=12, color=WARNA_TEKS, loc="left")
    ax2.set_xlabel("BMI (latar: kategori WHO)", color=WARNA_TEKS_SEKUNDER)
    ax2.set_ylabel("Skor risiko (0 sampai 100)", color=WARNA_TEKS_SEKUNDER)

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(
        buffer, format="png", dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight"
    )
    plt.close(fig)  # Mencegah kebocoran memori pada server yang berjalan lama

    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Penjelasan Fallback Statistik
# ---------------------------------------------------------------------------

def buat_penjelasan_fallback(ringkasan: dict, alasan: str = "") -> str:
    """Penjelasan berbasis statistik sederhana, dipakai jika Groq tidak dapat dipanggil."""
    stat = hitung_statistik_pembanding(ringkasan)
    biaya = ringkasan["prediksi_biaya"]
    bmi = ringkasan["bmi"]
    skor = ringkasan["skor_fuzzy"]
    kategori = ringkasan["kategori_risiko"]

    arah_perokok = "lebih tinggi" if stat["selisih_vs_perokok"] >= 0 else "lebih rendah"
    arah_non_perokok = "lebih tinggi" if stat["selisih_vs_non_perokok"] >= 0 else "lebih rendah"
    arah_bmi = "di atas" if stat["selisih_bmi"] >= 0 else "di bawah"

    paragraf_biaya = (
        f"Biaya asuransi tahunan Anda diperkirakan sebesar {_format_usd(biaya)}. "
        f"Angka ini {arah_non_perokok} {abs(stat['persen_vs_non_perokok']):.0f}% dari rata-rata "
        f"non-perokok di dataset ({_format_usd(stat['rata_biaya_non_perokok'])}) dan "
        f"{arah_perokok} {abs(stat['persen_vs_perokok']):.0f}% dari rata-rata perokok "
        f"({_format_usd(stat['rata_biaya_perokok'])})."
    )

    paragraf_risiko = (
        f"BMI Anda {bmi:.1f} termasuk kategori {stat['kategori_bmi'].lower()}, "
        f"{abs(stat['selisih_bmi']):.1f} poin {arah_bmi} rata-rata dataset ({stat['rata_bmi']:.1f}). "
        f"Dengan usia {ringkasan['age']} tahun, sistem fuzzy memberi skor risiko {skor:.1f} dari 100, "
        f"yaitu kategori risiko {kategori.lower()}."
    )

    keterangan_alasan = f" ({alasan})" if alasan else ""
    catatan = (
        f"Catatan: penjelasan ini dibuat otomatis dari perhitungan statistik karena layanan "
        f"Groq AI tidak dapat digunakan{keterangan_alasan}. Hasil ini hanya estimasi edukatif, "
        "bukan saran medis maupun penawaran asuransi."
    )

    return "\n\n".join([paragraf_biaya, paragraf_risiko, catatan])


# ---------------------------------------------------------------------------
# Penjelasan AI (Groq)
# ---------------------------------------------------------------------------

PROMPT_SISTEM = (
    "Anda adalah asisten yang menjelaskan hasil sistem prediksi biaya asuransi kesehatan kepada "
    "pengguna awam. Tulis dalam bahasa Indonesia yang ringkas, jelas, dan mudah dipahami. "
    "Gunakan teks biasa tanpa markdown, tanpa daftar, dan tanpa emoji. Hanya gunakan angka yang "
    "diberikan, jangan mengarang data baru, dan jangan memberi diagnosis atau saran medis."
)


def _susun_prompt_pengguna(ringkasan: dict) -> str:
    stat = hitung_statistik_pembanding(ringkasan)
    detail = ringkasan.get("detail_fuzzy") or {}

    derajat_bmi = ", ".join(
        f"{nama} {nilai:.2f}" for nama, nilai in detail.get("derajat_bmi", {}).items() if nilai > 0
    )
    derajat_usia = ", ".join(
        f"{nama} {nilai:.2f}" for nama, nilai in detail.get("derajat_usia", {}).items() if nilai > 0
    )

    baris = [
        "Sebuah grafik dua panel dibuat dari data berikut. Jelaskan maknanya kepada pengguna.",
        "",
        "Profil pengguna:",
        f"- Usia: {ringkasan['age']} tahun",
        f"- Jenis kelamin: {LABEL_SEX.get(ringkasan['sex'], ringkasan['sex'])}",
        f"- Jumlah anak: {ringkasan['children']}",
        f"- Status: {_label_perokok(ringkasan['smoker'])}",
        f"- Wilayah: {ringkasan['region']}",
        "",
        "Panel 1 (perbandingan biaya asuransi tahunan, dalam USD):",
        f"- Estimasi biaya pengguna (Random Forest): {_format_usd(ringkasan['prediksi_biaya'])}",
        f"- Rata-rata biaya non-perokok pada dataset: {_format_usd(stat['rata_biaya_non_perokok'])}",
        f"- Rata-rata biaya perokok pada dataset: {_format_usd(stat['rata_biaya_perokok'])}",
        f"- Selisih terhadap rata-rata non-perokok: {stat['persen_vs_non_perokok']:+.1f}% "
        f"({stat['selisih_vs_non_perokok']:+,.0f} USD)",
        f"- Selisih terhadap rata-rata perokok: {stat['persen_vs_perokok']:+.1f}% "
        f"({stat['selisih_vs_perokok']:+,.0f} USD)",
        "",
        "Panel 2 (posisi BMI dan skor risiko fuzzy):",
        f"- BMI pengguna: {ringkasan['bmi']:.1f} (kategori WHO: {stat['kategori_bmi']})",
        f"- Rata-rata BMI dataset: {stat['rata_bmi']:.1f} (kategori WHO: {stat['kategori_bmi_rata']})",
        f"- Selisih BMI pengguna terhadap rata-rata dataset: {stat['selisih_bmi']:+.1f} poin",
        f"- Skor risiko fuzzy: {ringkasan['skor_fuzzy']:.1f} dari 100, kategori {ringkasan['kategori_risiko']}",
        f"- Batas kategori skor: Rendah sampai {BATAS_RENDAH:.0f}, Sedang sampai {BATAS_SEDANG:.0f}, "
        "Tinggi di atasnya",
    ]
    if derajat_bmi:
        baris.append(f"- Derajat keanggotaan BMI yang aktif: {derajat_bmi}")
    if derajat_usia:
        baris.append(f"- Derajat keanggotaan usia yang aktif: {derajat_usia}")

    baris += [
        "",
        "Tugas: tulis maksimal 3 paragraf pendek (sekitar 120 kata). Paragraf pertama menjelaskan "
        "posisi estimasi biaya terhadap kedua rata-rata. Paragraf kedua menjelaskan arti posisi BMI "
        "terhadap rata-rata dataset dan arti skor serta kategori risiko fuzzy. Paragraf ketiga "
        "berupa satu atau dua kalimat penutup yang menegaskan bahwa hasil ini hanya estimasi edukatif.",
    ]
    return "\n".join(baris)


def _tentukan_alasan_error(error: Exception) -> str:
    """Mengubah error dari SDK Groq menjadi alasan singkat yang aman ditampilkan."""
    kode = getattr(error, "status_code", None)
    nama = type(error).__name__

    if kode in (401, 403) or nama in ("AuthenticationError", "PermissionDeniedError"):
        return "API key tidak valid atau tidak memiliki izin"
    if kode == 429 or nama == "RateLimitError":
        return "batas penggunaan atau kuota API sudah habis"
    if nama == "APITimeoutError":
        return "permintaan ke Groq melewati batas waktu"
    if nama == "APIConnectionError":
        return "tidak dapat terhubung ke server Groq"
    return "terjadi kesalahan saat menghubungi Groq"


def buat_penjelasan_ai(ringkasan: dict):
    """
    Meminta Groq menjelaskan makna grafik menggunakan model aktif pada free tier.

    Mengembalikan tuple (teks_penjelasan, sumber) dengan sumber bernilai "Groq AI" atau
    "Fallback Statistik".
    """
    if Groq is None:
        logger.warning("Paket groq belum terpasang, memakai Fallback Statistik.")
        return (
            buat_penjelasan_fallback(ringkasan, "paket groq belum terpasang"),
            SUMBER_FALLBACK,
        )

    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        logger.warning("GROQ_API_KEY belum diisi, memakai Fallback Statistik.")
        return (
            buat_penjelasan_fallback(ringkasan, "API key Groq belum diisi"),
            SUMBER_FALLBACK,
        )

    client = Groq(api_key=api_key, timeout=TIMEOUT_GROQ_DETIK, max_retries=1)
    pesan = [
        {"role": "system", "content": PROMPT_SISTEM},
        {"role": "user", "content": _susun_prompt_pengguna(ringkasan)},
    ]

    error_terakhir = None
    for nama_model in DAFTAR_MODEL_GROQ:
        try:
            respons = client.chat.completions.create(
                model=nama_model,
                messages=pesan,
                temperature=0.4,
                max_tokens=1200,
            )
            teks = (respons.choices[0].message.content or "").strip()
            if teks:
                return teks, SUMBER_GROQ
        except Exception as error:
            error_terakhir = error
            logger.warning("Model Groq %s gagal (%s): %s", nama_model, type(error).__name__, error)
            if getattr(error, "status_code", None) in (401, 403):
                break

    alasan = _tentukan_alasan_error(error_terakhir) if error_terakhir else "respons dari Groq kosong"
    return buat_penjelasan_fallback(ringkasan, alasan), SUMBER_FALLBACK


# ---------------------------------------------------------------------------
# Fungsi gabungan untuk dipanggil dari views.py
# ---------------------------------------------------------------------------

def proses_visual_dan_penjelasan(hasil: dict) -> dict:
    """
    Membuat grafik lalu penjelasannya dari hasil analisis_lengkap().

    Mengembalikan dict berisi:
    - grafik_base64    : untuk ditampilkan di HTML (<img src="data:image/png;base64,...">)
    - penjelasan_ai    : teks penjelasan (Groq AI atau Fallback Statistik)
    - sumber_penjelasan: "Groq AI" atau "Fallback Statistik"
    """
    grafik_base64 = buat_grafik(
        prediksi_biaya=hasil["prediksi_biaya"],
        bmi=hasil["bmi"],
        skor_fuzzy=hasil["skor_fuzzy"],
        kategori_risiko=hasil["kategori_risiko"],
    )
    penjelasan, sumber = buat_penjelasan_ai(hasil)

    return {
        "grafik_base64": grafik_base64,
        "penjelasan_ai": penjelasan,
        "sumber_penjelasan": sumber,
    }