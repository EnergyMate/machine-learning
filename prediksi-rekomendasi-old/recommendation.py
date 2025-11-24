# To run this code you need to install the following dependencies:
# pip install google-genai numpy tensorflow joblib

import joblib
import numpy as np
import os
import json
import subprocess
from tensorflow.keras.models import load_model

# --- Konfigurasi Model & Data ---
try:
    # Asumsikan model dan scaler ada di lokasi ini
    model = load_model('./models/best_model.keras')
    X_scaler = joblib.load("./models/X_scaler.save")
    if hasattr(X_scaler, 'feature_names_in_'):
        del X_scaler.feature_names_in_
    y_scaler = joblib.load("./models/y_scaler.save")
except Exception as e:
    print(f"Peringatan: Gagal memuat model/scaler ({e}). Menggunakan nilai dummy untuk prediksi.")
    # Dummy objects untuk memungkinkan kode berjalan tanpa file model yang sebenarnya
    class DummyScaler:
        def transform(self, X): return X
        def inverse_transform(self, X): return X
    model = type('DummyModel', (object,), {'predict': lambda self, X, verbose=0: np.array([[3.5]])})()
    X_scaler = DummyScaler()
    y_scaler = DummyScaler()

Q1 = 2.50  # Kuartil 1 (Batas Rendah/Sedang)
Q3 = 5.00  # Kuartil 3 (Batas Sedang/Tinggi)

SUB_LABELS = {
    "Sub_metering_1": "Peralatan Dapur & Elektronik Kecil",
    "Sub_metering_2": "Peralatan Laundry & Pemanas",
    "Sub_metering_3": "Pendingin, Penerangan & Pembersih"
}

# --- Fungsi Penggunaan Energi Pengguna ---

def get_user_consumption():
    """Meminta input penggunaan alat dari pengguna."""
    appliance_power_watt = {
        "microwave": 1000, "penanak nasi": 300, "blender": 250, "dispenser air": 200, "pemanggang roti": 850,
        "mesin cuci": 500, "mesin pengering": 3000, "setrika": 1000, "pompa air": 750, "pengering rambut": 600,
        "pemanas air": 1500, "AC": 800, "penyedot debu": 1200, "kipas angin": 100, "lampu LED": 20
    }

    appliance_to_sub = {
        "microwave": "Sub_metering_1", "penanak nasi": "Sub_metering_1", "blender": "Sub_metering_1", "dispenser air": "Sub_metering_1", "pemanggang roti": "Sub_metering_1",
        "mesin cuci": "Sub_metering_2", "mesin pengering": "Sub_metering_2", "setrika": "Sub_metering_2", "pompa air": "Sub_metering_2", "pengering rambut": "Sub_metering_2",
        "pemanas air": "Sub_metering_3", "AC": "Sub_metering_3", "penyedot debu": "Sub_metering_3", "kipas angin": "Sub_metering_3", "lampu LED": "Sub_metering_3"
    }

    submeter_usage = {"Sub_metering_1": 0.0, "Sub_metering_2": 0.0, "Sub_metering_3": 0.0}

    print("--- Input Penggunaan Energi Saat Ini ---")
    while True:
        item = input("Nama alat (atau ketik 'selesai'): ").strip().lower()
        if item == "selesai":
            break

        watt = appliance_power_watt.get(item, 20)
        
        try:
            count = int(input(f"Jumlah unit '{item}' (default: 1): ") or 1)
            if count < 1:
                count = 1
        except ValueError:
            count = 1

        energy_kwh = (watt * count) / 1000
        sub_key = appliance_to_sub.get(item, "Sub_metering_1")
        submeter_usage[sub_key] += energy_kwh

        print(f"► {item}: {count} × {watt} watt = {energy_kwh:.2f} kWh pada {SUB_LABELS.get(sub_key, sub_key)}\n")

    try:
        hour = int(input("Jam sekarang (0-23, default 12): ") or 12)
        hour = max(0, min(23, hour))
    except ValueError:
        hour = 12

    total_kw = sum(submeter_usage.values())
    voltage = 230
    # Global_intensity (Ampere) dihitung dari daya (Watt=Total_kW*1000) dibagi Tegangan
    global_intensity = round((total_kw * 1000) / voltage, 2)

    return {
        "Global_intensity": global_intensity,
        "Sub_metering_1": round(submeter_usage["Sub_metering_1"], 2),
        "Sub_metering_2": round(submeter_usage["Sub_metering_2"], 2),
        "Sub_metering_3": round(submeter_usage["Sub_metering_3"], 2),
        "hour": hour
    }

# --- Fungsi Gemini LLM untuk Rekomendasi Dinamis ---

def llm_based_recommendation(pred_kw, usage_kws, category, max_label):
    """
    Menghasilkan rekomendasi spesifik via REST API Gemini (curl).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        return f"Area Fokus: {max_label}. Saran: API Key belum diatur (gunakan rekomendasi default)."

    breakdown_text = "\n".join([f"- {k}: {v:.2f} kWh" for k, v in usage_kws.items()])

    system_prompt = (
        "Anda adalah ahli efisiensi energi profesional. "
        "Berikan rekomendasi sangat spesifik dalam BAHASA INDONESIA, "
        "maksimal 3 poin, dalam bentuk bullet list yang actionable."
    )

    user_query = (
        f"{system_prompt}\n\n"
        f"Data konsumsi energi:\n"
        f"Kategori: {category} (Prediksi: {pred_kw:.2f} kWh)\n"
        f"Area Fokus: {max_label}\n"
        f"Rincian Penggunaan:\n{breakdown_text}\n"
        "Berikan rekomendasi."
    )

    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_query}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7
        }
    }

    MODEL_ID = "gemini-2.0-flash"
    GENERATE_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={api_key}"

    try:
        # Menjalankan curl melalui subprocess
        process = subprocess.Popen(
            [
                "curl", "-s", "-X", "POST",
                "-H", "Content-Type: application/json",
                GENERATE_ENDPOINT,
                "-d", json.dumps(request_body)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output, error = process.communicate()

        if error:
            return f"Area Fokus: {max_label}. (Error API: {error.decode('utf-8')})"

        response_json = json.loads(output.decode("utf-8"))

        # Ambil teks hasil rekomendasi
        text = ""
        if "candidates" in response_json:
            parts = response_json["candidates"][0]["content"]["parts"]
            for p in parts:
                text += p.get("text", "")

        return text.strip() if text else f"Area Fokus: {max_label}. Tidak ada respon dari API."

    except Exception as e:
        return f"Area Fokus: {max_label}. (Error: {e})"

# --- Fungsi Prediksi dan Rekomendasi Utama ---

def predict_and_recommend(user_input):
    """Melakukan prediksi dan menghasilkan rekomendasi (Rule-Based + LLM)."""
    
    # 1. Persiapan Input dan Prediksi Model
    
    if user_input.get("Global_intensity") is None:
        total_kw = (user_input["Sub_metering_1"] +
                    user_input["Sub_metering_2"] +
                    user_input["Sub_metering_3"])
        user_input["Global_intensity"] = (total_kw * 1000) / 230

    X_input = np.array([
        user_input["Global_intensity"],
        user_input["Sub_metering_1"],
        user_input["Sub_metering_2"],
        user_input["Sub_metering_3"],
        user_input["hour"]
    ])
    
    # Model membutuhkan input urutan 60 langkah waktu
    past_60_input = np.tile(X_input, (60, 1)) 

    X_scaled_sequence = X_scaler.transform(past_60_input)
    X_seq = X_scaled_sequence.reshape(1, 60, -1)

    pred_scaled = model.predict(X_seq, verbose=0)[0][0]
    pred_kw = y_scaler.inverse_transform([[pred_scaled]])[0][0]

    usage_kws = {
        "Sub_metering_1": user_input["Sub_metering_1"],
        "Sub_metering_2": user_input["Sub_metering_2"],
        "Sub_metering_3": user_input["Sub_metering_3"]
    }
    
    total_usage = sum(usage_kws.values())
    max_sub = max(usage_kws, key=usage_kws.get)
    max_label = SUB_LABELS.get(max_sub, max_sub)

    # 2. Rekomendasi Berbasis Aturan (Rule-Based)
    if pred_kw < Q1:
        category = "Rendah"
        base_rec = "Konsumsi umum rendah—bagus, pertahankan pola ini."
    elif pred_kw <= Q3:
        category = "Sedang"
        base_rec = "Konsumsi sedang—perhatikan peralatan yang tidak dipakai."
    else:
        category = "Tinggi"
        base_rec = "Konsumsi tinggi—kurangi beban puncak."
        
    # 3. Rekomendasi Spesifik Berbasis LLM
    print("\n[Membuat rekomendasi dinamis dengan LLM...]")
    specific_llm_rec = llm_based_recommendation(
        pred_kw=pred_kw, 
        usage_kws=usage_kws, 
        category=category, 
        max_label=max_label
    )

    return {
        "total_usage_kw": round(total_usage, 2),
        "prediction_kw": round(pred_kw, 2),
        "category": category,
        "breakdown": {SUB_LABELS[k]: round(v, 2) for k, v in usage_kws.items()},
        "general_recommendation": base_rec,
        "focus_area": max_label,
        # Mengganti rekomendasi spesifik berbasis aturan dengan output dari LLM
        "specific_recommendation": specific_llm_rec 
    }


if __name__ == "__main__":
    user_input = get_user_consumption()
    recommendation = predict_and_recommend(user_input)

    print("\n==================================")
    print("=== HASIL & REKOMENDASI ENERGI ===")
    print("==================================")
    print(f"Total Penggunaan Saat Ini: {recommendation['total_usage_kw']} kWh")
    print(f"Prediksi Konsumsi (Model): {recommendation['prediction_kw']:.2f} kWh ({recommendation['category']})")
    print(f"Area Fokus Tertinggi: {recommendation['breakdown'].get(recommendation['focus_area'], recommendation['focus_area'])}")
    print("\n--- Detail Penggunaan ---")
    for label, usage in recommendation['breakdown'].items():
        print(f"  {label.ljust(40)}: {usage:.2f} kWh")
        
    print("\n--- Rekomendasi Umum ---")
    print(recommendation['general_recommendation'])
    
    print("\n--- Rekomendasi Spesifik ---")
    print(recommendation['specific_recommendation'])
    print("==================================")
