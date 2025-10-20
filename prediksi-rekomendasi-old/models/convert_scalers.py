from joblib import load
import json

def scaler_to_json(input_path, output_path):
    """Konversi scaler .save/.joblib ke file JSON."""
    scaler = load(input_path)
    scaler_data = {
        "mean_": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
        "scale_": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
        "min_": scaler.min_.tolist() if hasattr(scaler, "min_") else None,
        "data_min_": scaler.data_min_.tolist() if hasattr(scaler, "data_min_") else None,
        "data_max_": scaler.data_max_.tolist() if hasattr(scaler, "data_max_") else None,
    }

    with open(output_path, "w") as f:
        json.dump(scaler_data, f, indent=4)
    
    print(f"✅ {input_path} berhasil dikonversi ke {output_path}")

# Konversi dua scaler
scaler_to_json("X_scaler.save", "X_scaler.json")
scaler_to_json("y_scaler.save", "y_scaler.json")
