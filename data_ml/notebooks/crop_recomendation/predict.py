import joblib
import pandas as pd
import sys

# Load trained model and label encoder
model = joblib.load("../../models/crop_recommendation/crop_recommendation_model.pkl")
le = joblib.load("../../models/crop_recommendation/label_encoder.pkl")

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = pd.DataFrame([{
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }])
    predicted_label = model.predict(features)[0]
    crop_name = le.inverse_transform([predicted_label])[0]
    return crop_name

def get_input(prompt, min_val, max_val):
    value = float(input(prompt))
    if value < min_val or value > max_val:
        print(f"❌ Invalid input! Please enter a value between {min_val} and {max_val}.")
        sys.exit(1)  # Exit the program if invalid input
    return value

if __name__ == "__main__":
    print("🌱 Enter soil and climate parameters for crop recommendation:")

    N = get_input("Nitrogen (N) [0 - 140]: ", 0, 140)
    P = get_input("Phosphorus (P) [5 - 145]: ", 5, 145)
    K = get_input("Potassium (K) [5 - 205]: ", 5, 205)
    temperature = get_input("Temperature (°C) [8 - 43]: ", 8, 43)
    humidity = get_input("Humidity (%) [14 - 100]: ", 14, 100)
    ph = get_input("pH value [3.5 - 9.9]: ", 3.5, 9.9)
    rainfall = get_input("Rainfall (mm) [20 - 300]: ", 20, 300)

    crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
    print("\n✅ Recommended Crop:", crop)
