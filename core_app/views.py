from django.shortcuts import render

from .chart_ai_engine import proses_visual_dan_penjelasan
from .ml_fuzzy_engine import analisis_lengkap
from .models import RiwayatPrediksi


def index(request):
    form_data = {
        "age": "30",
        "sex": "male",
        "mode_bmi": "langsung",
        "bmi": "24.5",
        "tinggi_cm": "170",
        "berat_kg": "70",
        "children": "0",
        "smoker": "no",
        "region": "southeast",
    }
    hasil = None
    pesan_error = ""

    if request.method == "POST":
        mode_bmi = request.POST.get("mode_bmi", "langsung").strip()
        age_raw = request.POST.get("age", "").strip()
        sex_raw = request.POST.get("sex", "").strip()
        bmi_raw = request.POST.get("bmi", "").strip()
        tinggi_raw = request.POST.get("tinggi_cm", "").strip()
        berat_raw = request.POST.get("berat_kg", "").strip()
        children_raw = request.POST.get("children", "").strip()
        smoker_raw = request.POST.get("smoker", "").strip()
        region_raw = request.POST.get("region", "").strip()

        form_data = {
            "age": age_raw,
            "sex": sex_raw,
            "mode_bmi": mode_bmi,
            "bmi": bmi_raw,
            "tinggi_cm": tinggi_raw,
            "berat_kg": berat_raw,
            "children": children_raw,
            "smoker": smoker_raw,
            "region": region_raw,
        }

        try:
            if mode_bmi == "tinggi_berat":
                try:
                    tinggi_cm = float(tinggi_raw)
                    berat_kg = float(berat_raw)
                except (TypeError, ValueError):
                    raise ValueError("Tinggi badan dan berat badan harus berupa angka.")

                if not (100.0 <= tinggi_cm <= 250.0):
                    raise ValueError("Tinggi badan harus antara 100 cm dan 250 cm.")
                if not (25.0 <= berat_kg <= 300.0):
                    raise ValueError("Berat badan harus antara 25 kg dan 300 kg.")

                tinggi_m = tinggi_cm / 100.0
                bmi_hitung = round(berat_kg / (tinggi_m * tinggi_m), 2)
                bmi_raw = str(bmi_hitung)
                form_data["bmi"] = bmi_raw

            analisis = analisis_lengkap(
                age=age_raw,
                sex=sex_raw,
                bmi=bmi_raw,
                children=children_raw,
                smoker=smoker_raw,
                region=region_raw,
            )

            visual_ai = proses_visual_dan_penjelasan(analisis)

            RiwayatPrediksi.objects.create(
                age=analisis["age"],
                sex=analisis["sex"],
                bmi=analisis["bmi"],
                children=analisis["children"],
                smoker=analisis["smoker"],
                region=analisis["region"],
                prediksi_biaya=analisis["prediksi_biaya"],
                skor_fuzzy=analisis["skor_fuzzy"],
                kategori_risiko=analisis["kategori_risiko"],
                penjelasan_ai=visual_ai["penjelasan_ai"],
            )

            hasil = {
                **analisis,
                **visual_ai,
            }

        except (ValueError, RuntimeError, FileNotFoundError) as err:
            pesan_error = str(err)
        except Exception as err:
            pesan_error = f"Terjadi kesalahan sistem saat memproses data: {err}"

    riwayat_list = RiwayatPrediksi.objects.all()[:10]

    context = {
        "form_data": form_data,
        "hasil": hasil,
        "pesan_error": pesan_error,
        "riwayat_list": riwayat_list,
    }
    return render(request, "index.html", context)