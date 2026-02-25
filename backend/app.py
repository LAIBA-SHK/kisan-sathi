
import requests
import io
import os
import json
import colorsys
import traceback

from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
from PIL import Image as PILImage
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

app = Flask(__name__)

# ═══════════════════════════════════════════════
# CROP RECOMMENDATION MODEL
# ═══════════════════════════════════════════════

def load_crop_model():
    base = os.path.dirname(__file__)
    model_path   = os.path.join(base, '../data_ml/models/crop_recommendation/crop_recommendation_model.pkl')
    encoder_path = os.path.join(base, '../data_ml/models/crop_recommendation/label_encoder.pkl')

    if not os.path.exists(model_path):
        print(f'❌ Crop model NOT found: {model_path}')
        return None, None
    if not os.path.exists(encoder_path):
        print(f'❌ Label encoder NOT found: {encoder_path}')
        return None, None

    try:
        crop_model   = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        print(f'✅ Crop model loaded successfully')
        print(f'✅ Crops supported: {list(label_encoder.classes_)}')
        return crop_model, label_encoder
    except Exception:
        print(f'❌ Crop model load failed:\n{traceback.format_exc()}')
        return None, None

crop_model, label_encoder = load_crop_model()

# Full crop info for all 22 crops the model supports
CROP_INFO = {
    "rice":        {"name":"Rice",        "season":"Kharif",       "duration":"120-150 days", "yield":"40-60 q/ha",  "description":"Rice needs flooded fields and warm, humid climate. It is the staple food for billions.", "tips":"Prepare nursery, transplant after 25-30 days, maintain 5cm water level, apply DAP at sowing"},
    "maize":       {"name":"Maize",       "season":"Kharif/Rabi",  "duration":"90-120 days",  "yield":"50-70 q/ha",  "description":"Maize is a versatile crop used for food, fodder and industry.", "tips":"Sow in rows 60cm apart, irrigate at silking stage, apply urea in splits, harvest when husks dry"},
    "chickpea":    {"name":"Chickpea",    "season":"Rabi",         "duration":"90-120 days",  "yield":"12-20 q/ha",  "description":"Chickpea is a protein-rich legume crop grown in dry cool conditions.", "tips":"Sow in October-November, avoid waterlogging, use Rhizobium seed treatment, harvest when pods turn brown"},
    "kidneybeans": {"name":"Kidney Beans","season":"Kharif",       "duration":"90-120 days",  "yield":"10-15 q/ha",  "description":"Kidney beans are high protein legumes grown in warm moist conditions.", "tips":"Sow after last frost, provide support, irrigate regularly, harvest when pods dry"},
    "pigeonpeas":  {"name":"Pigeon Peas", "season":"Kharif",       "duration":"150-180 days", "yield":"10-15 q/ha",  "description":"Pigeon peas are drought-tolerant legumes widely grown in dry areas.", "tips":"Sow at start of monsoon, tolerates poor soils, minimal irrigation needed, harvest when 80% pods mature"},
    "mothbeans":   {"name":"Moth Beans",  "season":"Kharif",       "duration":"75-90 days",   "yield":"5-8 q/ha",    "description":"Moth bean is an arid-zone legume, very drought resistant.", "tips":"Sow in sandy loam soil, minimal water needed, harvest in stages as pods mature"},
    "mungbean":    {"name":"Mung Bean",   "season":"Kharif/Rabi",  "duration":"60-90 days",   "yield":"8-12 q/ha",   "description":"Mung bean is a short-duration legume with high protein content.", "tips":"Sow in well-drained soil, 2-3 irrigations sufficient, harvest when 80% pods turn black"},
    "blackgram":   {"name":"Black Gram",  "season":"Kharif/Rabi",  "duration":"70-90 days",   "yield":"8-12 q/ha",   "description":"Black gram is a nutritious pulse crop grown in warm humid conditions.", "tips":"Sow in loamy soil, avoid waterlogging, 3-4 irrigations needed, harvest when pods turn black"},
    "lentil":      {"name":"Lentil",      "season":"Rabi",         "duration":"90-120 days",  "yield":"10-15 q/ha",  "description":"Lentil is a cool-season legume rich in protein and fiber.", "tips":"Sow October-November, minimal water, inoculate with Rhizobium, harvest when lower pods turn yellow"},
    "pomegranate": {"name":"Pomegranate", "season":"Perennial",    "duration":"3-4 years",    "yield":"150-200 q/ha","description":"Pomegranate is a drought-tolerant fruit crop suitable for arid regions.", "tips":"Plant in well-drained soil, prune regularly, drip irrigation preferred, harvest when fruits turn red"},
    "banana":      {"name":"Banana",      "season":"Year-round",   "duration":"12-18 months", "yield":"300-400 q/ha","description":"Banana is a high-yield tropical fruit crop needing warm humid conditions.", "tips":"Plant tissue culture suckers, drip irrigate, apply heavy potassium doses, harvest at 75% maturity"},
    "mango":       {"name":"Mango",       "season":"Summer",       "duration":"5-8 years",    "yield":"100-200 q/ha","description":"Mango is the king of fruits, requiring dry winters and hot summers.", "tips":"Plant in deep well-drained soil, minimal irrigation after establishment, spray urea at flowering"},
    "grapes":      {"name":"Grapes",      "season":"Perennial",    "duration":"3-4 years",    "yield":"200-300 q/ha","description":"Grapes are high-value fruit crops requiring specific pruning and training.", "tips":"Train on trellises, prune in January, regulate water after fruit set, harvest at correct brix level"},
    "watermelon":  {"name":"Watermelon",  "season":"Kharif/Summer","duration":"80-110 days",  "yield":"200-350 q/ha","description":"Watermelon is a warm-season crop needing full sun and well-drained sandy soil.", "tips":"Sow on raised beds, provide frequent irrigation, use black mulch, harvest when tendril near fruit dries"},
    "muskmelon":   {"name":"Muskmelon",   "season":"Summer",       "duration":"75-100 days",  "yield":"150-200 q/ha","description":"Muskmelon is a warm-season fruit with high water needs during fruit development.", "tips":"Sow in sandy loam, irrigate frequently, reduce water near maturity, harvest when aroma develops"},
    "apple":       {"name":"Apple",       "season":"Winter/Spring","duration":"5-8 years",    "yield":"200-400 q/ha","description":"Apple is a temperate fruit crop requiring cold winters for dormancy.", "tips":"Plant in well-drained loamy soil, require chilling hours, prune annually, thin fruits for size"},
    "orange":      {"name":"Orange",      "season":"Winter",       "duration":"3-5 years",    "yield":"150-250 q/ha","description":"Orange is a citrus fruit crop needing warm days and cool nights.", "tips":"Plant in deep soil, drip irrigate, apply micronutrients, harvest when color fully develops"},
    "papaya":      {"name":"Papaya",      "season":"Year-round",   "duration":"10-14 months", "yield":"400-500 q/ha","description":"Papaya is a fast-growing tropical fruit with very high yields.", "tips":"Plant 2x2m spacing, provide regular water and fertilizer, protect from frost and waterlogging"},
    "coconut":     {"name":"Coconut",     "season":"Perennial",    "duration":"5-7 years",    "yield":"80-100 nuts/tree","description":"Coconut is a tropical palm providing nuts, oil, and fiber.", "tips":"Plant in coastal sandy soil, needs high rainfall, apply salt and potassium, harvest every 45 days"},
    "cotton":      {"name":"Cotton",      "season":"Kharif",       "duration":"180-200 days", "yield":"15-25 q/ha",  "description":"Cotton is the most important fiber crop providing raw material to the textile industry.", "tips":"Deep ploughing, use Bt cotton varieties, integrated pest management, pick when bolls open fully"},
    "jute":        {"name":"Jute",        "season":"Kharif",       "duration":"120-150 days", "yield":"20-30 q/ha",  "description":"Jute is a fiber crop requiring hot humid climate and flooded conditions.", "tips":"Sow April-May, needs frequent irrigation, ret stalks in water after harvest, dry before storage"},
    "coffee":      {"name":"Coffee",      "season":"Perennial",    "duration":"3-4 years",    "yield":"8-15 q/ha",   "description":"Coffee is a high-value beverage crop grown in shade at medium altitudes.", "tips":"Grow under shade trees, needs well-distributed rainfall, pulp and ferment after harvest"},
    "wheat":       {"name":"Wheat",       "season":"Rabi",         "duration":"120-150 days", "yield":"35-50 q/ha",  "description":"Wheat is the major rabi cereal grown in cool dry conditions.", "tips":"Sow October-November, 4-6 irrigations, apply DAP at sowing and urea in 2 splits, harvest at golden stage"},
    "tobacco":     {"name":"Tobacco",     "season":"Rabi/Kharif",  "duration":"90-120 days",  "yield":"15-20 q/ha",  "description":"Tobacco is a commercial crop requiring specific curing after harvest.", "tips":"Transplant seedlings, avoid excess nitrogen, top plants at flowering, cure in barn after harvest"},
    "tomato":      {"name":"Tomato",      "season":"Kharif/Rabi",  "duration":"90-120 days",  "yield":"400-600 q/ha","description":"Tomato is a popular vegetable rich in vitamins and minerals.", "tips":"Use disease-resistant varieties, provide stake support, regular watering, harvest at proper maturity"},
    "potato":      {"name":"Potato",      "season":"Rabi",         "duration":"70-120 days",  "yield":"200-400 q/ha","description":"Potato is an important food crop and major source of carbohydrates.", "tips":"Plant in well-drained soil, earthing up is essential, control late blight, harvest when plants mature"},
}

MARKET_PRICES = {
    "rice":        {"price": 2443, "trend": "+5.2%", "market": "Lahore"},
    "wheat":       {"price": 2167, "trend": "+2.1%", "market": "Karachi"},
    "cotton":      {"price": 5906, "trend": "-1.8%", "market": "Multan"},
    "tomato":      {"price": 2000, "trend": "+8.5%", "market": "Faisalabad"},
    "potato":      {"price": 1200, "trend": "-3.2%", "market": "Lahore"},
    "maize":       {"price": 1800, "trend": "+3.1%", "market": "Rawalpindi"},
    "chickpea":    {"price": 3200, "trend": "+1.5%", "market": "Lahore"},
    "mango":       {"price": 4500, "trend": "+12%",  "market": "Multan"},
    "banana":      {"price": 1500, "trend": "+4.2%", "market": "Karachi"},
    "orange":      {"price": 2800, "trend": "+6.1%", "market": "Sargodha"},
    "sugarcane":   {"price": 350,  "trend": "+2.0%", "market": "Lahore"},
    "onion":       {"price": 1800, "trend": "-5.1%", "market": "Lahore"},
}

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON or missing Content-Type'}), 400

    if crop_model is None or label_encoder is None:
        print('❌ Crop model not loaded — cannot predict')
        return jsonify({'error': 'Crop model not loaded on server. Check file paths.'}), 500

    def clamp(val, minv, maxv):
        try:
            return max(minv, min(maxv, float(val)))
        except:
            return minv

    try:
        features = pd.DataFrame([{
            'N':           clamp(data.get('N', 0),           0,    140),
            'P':           clamp(data.get('P', 0),           5,    145),
            'K':           clamp(data.get('K', 0),           5,    205),
            'temperature': clamp(data.get('temperature', 0), 8,    43),
            'humidity':    clamp(data.get('humidity', 0),    14,   100),
            'ph':          clamp(data.get('ph', 0),          3.5,  9.9),
            'rainfall':    clamp(data.get('rainfall', 0),    20,   300),
        }])
        print('📥 Crop input:', features.to_dict(orient='records'))

        predicted_label = crop_model.predict(features)[0]
        crop_name = label_encoder.inverse_transform([predicted_label])[0]
        print(f'🌾 Predicted crop: {crop_name}')

        # Get probability if model supports it
        confidence = 100
        if hasattr(crop_model, 'predict_proba'):
            probs = crop_model.predict_proba(features)[0]
            confidence = round(float(max(probs)) * 100, 1)

        info   = CROP_INFO.get(crop_name.lower(), {})
        market = MARKET_PRICES.get(crop_name.lower(), {})

        return jsonify({
            "recommendation": crop_name,
            "name":           info.get("name", crop_name),
            "season":         info.get("season", "N/A"),
            "duration":       info.get("duration", "N/A"),
            "yield":          info.get("yield", "N/A"),
            "description":    info.get("description", f"{crop_name} is recommended for your soil and climate conditions."),
            "tips":           info.get("tips", "Consult your local agriculture officer for best practices."),
            "marketPrice":    market,
            "confidence":     confidence,
        })

    except Exception:
        print(f'❌ Crop prediction error:\n{traceback.format_exc()}')
        return jsonify({'error': 'Prediction failed. Check server logs.'}), 500


# ═══════════════════════════════════════════════
# STATIC FILE SERVING (single catch-all route)
# ═══════════════════════════════════════════════

@app.route('/')
def home():
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    # Search in frontend folder
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    fp = os.path.join(frontend_dir, filename)
    if os.path.exists(fp):
        return send_from_directory(frontend_dir, filename)
    return f'{filename} not found', 404


# ═══════════════════════════════════════════════
# WEATHER
# ═══════════════════════════════════════════════

@app.route('/api/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city', 'Lahore')
    geo_headers = {"User-Agent": "KisanSathi/1.0"}
    geocode_url = f'https://nominatim.openstreetmap.org/search?format=json&q={city}'
    try:
        geo_resp = requests.get(geocode_url, headers=geo_headers, timeout=10)
        geo_data = geo_resp.json()
        if not geo_data:
            return jsonify({'error': 'City not found'}), 404
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        weather_url = (
            f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}'
            '&hourly=temperature_2m,relative_humidity_2m'
            '&daily=temperature_2m_max,temperature_2m_min,precipitation_sum'
            '&current_weather=true&timezone=auto'
        )
        weather_data = requests.get(weather_url, timeout=10).json()
        daily = weather_data.get('daily', {})
        humidity, rainfall = None, None
        try:
            if 'relative_humidity_2m' in weather_data.get('hourly', {}):
                h = weather_data['hourly']['relative_humidity_2m'][:24]
                humidity = round(sum(h) / len(h), 1)
            if 'precipitation_sum' in daily:
                rainfall = daily['precipitation_sum'][0]
        except Exception:
            pass
        return jsonify({
            'city': city, 'latitude': lat, 'longitude': lon,
            'current': weather_data.get('current_weather', {}),
            'daily': daily, 'hourly': weather_data.get('hourly', {}),
            'humidity': humidity, 'rainfall': rainfall,
        })
    except Exception:
        print(f'❌ Weather error:\n{traceback.format_exc()}')
        return jsonify({'error': 'Weather fetch failed'}), 500


# ═══════════════════════════════════════════════
# DISEASE DETECTION MODEL
# ═══════════════════════════════════════════════

def load_disease_model():
    base = os.path.dirname(__file__)
    model_path  = os.path.join(base, '../data_ml/models/mobilenetv3_plant_disease.pth')
    labels_path = os.path.join(base, '../data_ml/models/class_labels.json')

    if not os.path.exists(model_path):
        print(f'❌ Disease model NOT found: {model_path}')
        return None, None
    if not os.path.exists(labels_path):
        print(f'❌ Class labels NOT found: {labels_path}')
        return None, None
    try:
        with open(labels_path, 'r') as f:
            class_labels = json.load(f)
        num_classes = len(class_labels)
        print(f'✅ Disease classes loaded: {num_classes}')

        disease_model = models.mobilenet_v3_small(weights=None)
        disease_model.classifier[3] = nn.Linear(disease_model.classifier[3].in_features, num_classes)
        disease_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        disease_model.eval()
        print('✅ Disease model loaded successfully')
        return disease_model, class_labels
    except Exception:
        print(f'❌ Disease model load failed:\n{traceback.format_exc()}')
        return None, None

disease_model, disease_labels = load_disease_model()

def is_plant_image(img):
    img_small = img.resize((100, 100))
    pixels = list(img_small.getdata())
    green_count = 0
    for r, g, b in pixels:
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        if 0.22 <= h <= 0.45 and s > 0.15 and v > 0.1:
            green_count += 1
    return (green_count / len(pixels)) >= 0.06

DISEASE_TREATMENTS = {
    'scab':      {'en': 'Remove infected leaves. Spray Captan or Mancozeb every 7-10 days.',          'ur': 'متاثرہ پتے ہٹائیں۔ کیپٹان یا مانکوزیب ہر 7-10 دن سپرے کریں۔',       'sev': 'Medium'},
    'black rot': {'en': 'Prune infected areas. Apply Copper fungicide immediately.',                   'ur': 'متاثرہ حصے کاٹیں۔ فوری کاپر فنگسائیڈ لگائیں۔',                       'sev': 'High'},
    'rust':      {'en': 'Apply Propiconazole or Tebuconazole. Remove infected leaves.',               'ur': 'پروپیکونازول سپرے کریں۔ متاثرہ پتے جلا دیں۔',                       'sev': 'High'},
    'blight':    {'en': 'Apply Mancozeb 75WP (2g/L water). Avoid overhead irrigation.',              'ur': 'مانکوزیب 2 گرام فی لیٹر سپرے کریں۔ اوپر سے پانی نہ دیں۔',           'sev': 'High'},
    'mildew':    {'en': 'Spray Sulfur or Triadimefon. Improve air circulation between plants.',       'ur': 'گندھک کا سپرے کریں۔ پودوں کے درمیان ہوا آنے دیں۔',                 'sev': 'Medium'},
    'mosaic':    {'en': 'Control aphids with Imidacloprid. Remove and burn infected plants.',         'ur': 'چیچڑ کو امیڈاکلوپرڈ سے کنٹرول کریں۔ متاثرہ پودے جلائیں۔',           'sev': 'High'},
    'spot':      {'en': 'Apply Copper Oxychloride (3g/L). Remove fallen infected leaves.',           'ur': 'کاپر آکسی کلورائیڈ 3 گرام فی لیٹر سپرے کریں۔',                     'sev': 'Medium'},
    'curl':      {'en': 'Control whitefly with Imidacloprid. Use virus-resistant varieties.',        'ur': 'سفید مکھی کنٹرول کریں۔ مزاحم اقسام لگائیں۔',                        'sev': 'High'},
    'greening':  {'en': 'No cure available. Remove infected trees to stop spread.',                  'ur': 'کوئی علاج نہیں۔ پھیلاؤ روکنے کو متاثرہ درخت کاٹیں۔',               'sev': 'High'},
    'mites':     {'en': 'Spray Abamectin or Spiromesifen. Maintain field moisture.',                 'ur': 'ابامیکٹن سپرے کریں۔ کھیت میں نمی برقرار رکھیں۔',                   'sev': 'Medium'},
    'bacterial': {'en': 'Apply Copper-based bactericide. Avoid working when plants are wet.',        'ur': 'کاپر بیکٹریسائیڈ سپرے کریں۔ گیلے وقت کام نہ کریں۔',               'sev': 'Medium'},
}

@app.route('/api/disease', methods=['POST'])
def detect_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    try:
        if disease_model is None or disease_labels is None:
            return jsonify({'error': 'Disease model not loaded on server'}), 500

        img_bytes = file.read()
        img = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')

        if not is_plant_image(img):
            resp = jsonify({'error': 'not_plant', 'message': 'No plant leaf detected in image'})
            resp.status_code = 422
            return resp

        # ✅ No Normalize — matches training exactly
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = disease_model(input_tensor)
            probs  = torch.softmax(output, dim=1)
            pred_idx   = probs.argmax(dim=1).item()
            confidence = round(probs[0][pred_idx].item() * 100, 1)

        label      = disease_labels.get(str(pred_idx), f'Unknown (class {pred_idx})')
        is_healthy = 'healthy' in label.lower()
        print(f'🔬 Disease detected: {label} ({confidence}%)')

        treatment_en = 'Consult your local agriculture officer.'
        treatment_ur = 'مقامی زراعت افسر سے رابطہ کریں۔'
        severity     = 'Medium'

        if is_healthy:
            severity     = 'None'
            treatment_en = 'Plant is healthy. Maintain regular watering and fertilization.'
            treatment_ur = 'پودا صحت مند ہے۔ باقاعدہ پانی اور کھاد دیتے رہیں۔'
        else:
            for key, val in DISEASE_TREATMENTS.items():
                if key in label.lower():
                    treatment_en = val['en']
                    treatment_ur = val['ur']
                    severity     = val['sev']
                    break

        return jsonify({
            'label':          label,
            'urdu':           label,
            'description':    f'{"Healthy plant." if is_healthy else "Disease detected."} Confidence: {confidence}%',
            'treatment':      treatment_en,
            'treatment_urdu': treatment_ur,
            'severity':       severity,
            'confidence':     confidence,
            'is_healthy':     is_healthy,
        })

    except Exception:
        print(f'❌ Disease detection error:\n{traceback.format_exc()}')
        return jsonify({'error': 'Detection failed. Check server logs.'}), 500


# ═══════════════════════════════════════════════
# CORS + RUN
# ═══════════════════════════════════════════════

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

if __name__ == "__main__":
    print('\n🌿 Kisan Sathi Backend Starting...')
    app.run(debug=True, port=5000)
