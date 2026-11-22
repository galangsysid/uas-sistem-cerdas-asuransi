from django.db import models


class RiwayatPrediksi(models.Model):
    """Menyimpan setiap hasil prediksi biaya asuransi beserta analisis risikonya."""

    PILIHAN_SEX = [
        ("male", "Laki-laki"),
        ("female", "Perempuan"),
    ]
    PILIHAN_SMOKER = [
        ("yes", "Perokok"),
        ("no", "Bukan perokok"),
    ]
    PILIHAN_REGION = [
        ("northeast", "Northeast"),
        ("northwest", "Northwest"),
        ("southeast", "Southeast"),
        ("southwest", "Southwest"),
    ]
    PILIHAN_KATEGORI_RISIKO = [
        ("Rendah", "Rendah"),
        ("Sedang", "Sedang"),
        ("Tinggi", "Tinggi"),
    ]

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Waktu dibuat")

    # Data masukan pengguna
    age = models.PositiveIntegerField(verbose_name="Usia")
    sex = models.CharField(max_length=10, choices=PILIHAN_SEX, verbose_name="Jenis kelamin")
    bmi = models.FloatField(verbose_name="BMI")
    children = models.PositiveIntegerField(verbose_name="Jumlah anak")
    smoker = models.CharField(max_length=3, choices=PILIHAN_SMOKER, verbose_name="Status perokok")
    region = models.CharField(max_length=20, choices=PILIHAN_REGION, verbose_name="Wilayah")

    # Hasil mesin Machine Learning dan Fuzzy Logic
    prediksi_biaya = models.FloatField(verbose_name="Prediksi biaya")
    skor_fuzzy = models.FloatField(verbose_name="Skor risiko fuzzy")
    kategori_risiko = models.CharField(
        max_length=20,
        choices=PILIHAN_KATEGORI_RISIKO,
        verbose_name="Kategori risiko",
    )

    # Hasil penjelasan AI (atau fallback statistik)
    penjelasan_ai = models.TextField(blank=True, default="", verbose_name="Penjelasan AI")

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Riwayat Prediksi"
        verbose_name_plural = "Riwayat Prediksi"

    def __str__(self) -> str:
        waktu = self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "baru"
        return f"Prediksi {waktu} | usia {self.age} | risiko {self.kategori_risiko}"