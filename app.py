from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import re
import csv
from db import create_user, authenticate_user, get_user_by_email, update_user, log_activity, get_user_activity, get_user_stats, get_user_monthly_activity, get_recent_soil_analyses, MONGO_CONNECTED

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('fertilizer_model.pkl', 'rb') as f:
    fert_model = pickle.load(f)

with open('fertilizer_scaler.pkl', 'rb') as f:
    fert_scaler = pickle.load(f)

with open('fertilizer_encoders.pkl', 'rb') as f:
    fert_encoders = pickle.load(f)

# Load fertilizer CSV data
FERTILIZER_CSV_DATA = []
try:
    with open('FertilizerPrediction.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            FERTILIZER_CSV_DATA.append({
                'temperature': int(row['Temparature']),
                'humidity': int(row['Humidity '].strip()),
                'moisture': int(row['Moisture']),
                'soil_type': row['Soil Type'],
                'crop_type': row['Crop Type'],
                'nitrogen': int(row['Nitrogen']),
                'potassium': int(row['Potassium']),
                'phosphorous': int(row['Phosphorous']),
                'fertilizer': row['Fertilizer Name']
            })
    print(f"[DATA] Loaded {len(FERTILIZER_CSV_DATA)} fertilizer records from CSV")
except Exception as e:
    print(f"[DATA] Could not load fertilizer CSV: {e}")

# Extract unique values from CSV
SOIL_TYPES = sorted(set(r['soil_type'] for r in FERTILIZER_CSV_DATA))
CROP_TYPES = sorted(set(r['crop_type'] for r in FERTILIZER_CSV_DATA))
FERTILIZER_NAMES = sorted(set(r['fertilizer'] for r in FERTILIZER_CSV_DATA))

FEATURES = ['n', 'p', 'k', 'ph', 'ec', 'oc', 's', 'zn', 'fe', 'cu', 'mn', 'b']

LABELS = {
    0: 'Not Fertile',
    1: 'Fertile',
    2: 'Highly Fertile'
}

DESCRIPTIONS = {
    0: 'The soil lacks essential nutrients. Consider adding organic compost, NPK fertilizers, and micronutrient supplements to improve fertility.',
    1: 'The soil has moderate fertility. With some targeted nutrient management and organic amendments, crop yields can be improved.',
    2: 'The soil is rich in nutrients and well-suited for crop production. Maintain current practices with periodic soil testing.'
}

# Reference ranges: (low, optimal_low, optimal_high, high) for each nutrient
REFERENCE_RANGES = {
    'n':  {'low': 140, 'opt_low': 280, 'opt_high': 560, 'high': 800,  'unit': 'kg/ha',  'name': 'Nitrogen (N)'},
    'p':  {'low': 5,   'opt_low': 10,  'opt_high': 25,  'high': 50,   'unit': 'kg/ha',  'name': 'Phosphorus (P)'},
    'k':  {'low': 100, 'opt_low': 200, 'opt_high': 500, 'high': 700,  'unit': 'kg/ha',  'name': 'Potassium (K)'},
    'ph': {'low': 5.5, 'opt_low': 6.0, 'opt_high': 7.5, 'high': 8.5,  'unit': '',       'name': 'pH'},
    'ec': {'low': 0.0, 'opt_low': 0.2, 'opt_high': 0.8, 'high': 1.5,  'unit': 'dS/m',   'name': 'Electrical Conductivity (EC)'},
    'oc': {'low': 0.3, 'opt_low': 0.5, 'opt_high': 1.0, 'high': 1.5,  'unit': '%',      'name': 'Organic Carbon (OC)'},
    's':  {'low': 5,   'opt_low': 10,  'opt_high': 30,  'high': 50,   'unit': 'ppm',    'name': 'Sulfur (S)'},
    'zn': {'low': 0.5, 'opt_low': 1.0, 'opt_high': 3.0, 'high': 5.0,  'unit': 'ppm',    'name': 'Zinc (Zn)'},
    'fe': {'low': 2.0, 'opt_low': 4.0, 'opt_high': 10.0,'high': 15.0, 'unit': 'ppm',    'name': 'Iron (Fe)'},
    'cu': {'low': 0.2, 'opt_low': 0.5, 'opt_high': 1.5, 'high': 3.0,  'unit': 'ppm',    'name': 'Copper (Cu)'},
    'mn': {'low': 2.0, 'opt_low': 4.0, 'opt_high': 15.0,'high': 25.0, 'unit': 'ppm',    'name': 'Manganese (Mn)'},
    'b':  {'low': 0.3, 'opt_low': 0.5, 'opt_high': 2.0, 'high': 3.5,  'unit': 'ppm',    'name': 'Boron (B)'}
}

RECOMMENDATIONS = {
    'n':  {'deficient': 'Apply urea (46-0-0) at 100-150 kg/ha or ammonium sulfate. Incorporate leguminous crops in rotation to fix atmospheric nitrogen.', 'excessive': 'Reduce nitrogen fertilizer application. Grow nitrogen-consuming crops and avoid adding N-based fertilizers for 1-2 seasons.'},
    'p':  {'deficient': 'Apply single super phosphate (SSP) or DAP at 50-75 kg/ha. Add farmyard manure to improve phosphorus availability.', 'excessive': 'Stop phosphorus fertilization. Excess P can lock out zinc and iron. Focus on crop removal to reduce levels.'},
    'k':  {'deficient': 'Apply muriate of potash (MOP) at 50-100 kg/ha. Add wood ash or banana peels as organic potassium sources.', 'excessive': 'Reduce potassium fertilizers. Excess K can interfere with magnesium and calcium uptake.'},
    'ph': {'deficient': 'Soil is too acidic. Apply agricultural lime (calcium carbonate) at 1-2 tonnes/ha to raise pH gradually.', 'excessive': 'Soil is too alkaline. Apply eleite sulfur or aluminum sulfate. Use acidic organic mulches like pine needles.'},
    'ec': {'deficient': 'EC is very low indicating low salt content. This is generally acceptable for most crops.', 'excessive': 'High salinity detected. Leach soil with good quality water. Add gypsum to improve soil structure and reduce salt concentration.'},
    'oc': {'deficient': 'Apply farmyard manure at 10-15 tonnes/ha or vermicompost. Practice green manuring and mulching.', 'excessive': 'Organic carbon is high which is beneficial. Maintain current organic matter management practices.'},
    's':  {'deficient': 'Apply gypsum (calcium sulfate) at 200-300 kg/ha or ammonium sulfate. Sulfur is often overlooked but critical.', 'excessive': 'Reduce sulfur-containing fertilizers. High sulfur can lower pH and affect other nutrient availability.'},
    'zn': {'deficient': 'Apply zinc sulfate at 25 kg/ha as soil application or 0.5% as foliar spray. Critical for crop growth.', 'excessive': 'Stop zinc application. Excess Zn can cause iron and copper deficiency. Time will naturally reduce levels.'},
    'fe': {'deficient': 'Apply ferrous sulfate at 50 kg/ha or use chelated iron (Fe-EDDHA) for faster correction. Foliar spray at 0.5%.', 'excessive': 'Iron toxicity is rare. Ensure proper drainage as waterlogged conditions increase Fe availability.'},
    'cu': {'deficient': 'Apply copper sulfate at 10-15 kg/ha. Deficiency is common in organic and sandy soils.', 'excessive': 'Stop copper application. Excess Cu is toxic to plants and soil organisms. Add organic matter to bind excess copper.'},
    'mn': {'deficient': 'Apply manganese sulfate at 20-25 kg/ha or foliar spray at 0.5%. Check pH as high pH reduces availability.', 'excessive': 'Reduce manganese application. Ensure pH is not too low as acidic soil increases Mn availability to toxic levels.'},
    'b':  {'deficient': 'Apply borax at 10-15 kg/ha. Boron has a narrow range between deficient and toxic, so apply carefully.', 'excessive': 'Stop boron application immediately. Leach soil with water. Boron toxicity is serious and damages root systems.'}
}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        values = [float(data[f]) for f in FEATURES]
        values_scaled = scaler.transform([values])
        prediction = int(model.predict(values_scaled)[0])
        probabilities = model.predict_proba(values_scaled)[0]

        email = data.get('email', '')
        if email:
            log_activity(email, 'soil_prediction', {
                'label': LABELS[prediction],
                'confidence': round(float(max(probabilities)) * 100, 1),
                'inputs': {f: data[f] for f in FEATURES}
            })

        return jsonify({
            'success': True,
            'prediction': prediction,
            'label': LABELS[prediction],
            'description': DESCRIPTIONS[prediction],
            'confidence': round(float(max(probabilities)) * 100, 1),
            'probabilities': {
                LABELS[i]: round(float(p) * 100, 1) for i, p in enumerate(probabilities)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/report', methods=['POST'])
def report():
    try:
        data = request.get_json()
        nutrient_analysis = []
        issues = []

        for feat in FEATURES:
            val = float(data[feat])
            ref = REFERENCE_RANGES[feat]

            if val < ref['low']:
                status = 'Deficient'
                severity = 'critical'
                rec = RECOMMENDATIONS[feat]['deficient']
                issues.append({'nutrient': ref['name'], 'status': 'Deficient', 'severity': 'critical', 'recommendation': rec})
            elif val < ref['opt_low']:
                status = 'Low'
                severity = 'warning'
                rec = RECOMMENDATIONS[feat]['deficient']
                issues.append({'nutrient': ref['name'], 'status': 'Low', 'severity': 'warning', 'recommendation': rec})
            elif val <= ref['opt_high']:
                status = 'Optimal'
                severity = 'good'
                rec = 'Level is within the optimal range. No corrective action needed.'
            elif val <= ref['high']:
                status = 'High'
                severity = 'warning'
                rec = RECOMMENDATIONS[feat]['excessive']
                issues.append({'nutrient': ref['name'], 'status': 'High', 'severity': 'warning', 'recommendation': rec})
            else:
                status = 'Excessive'
                severity = 'critical'
                rec = RECOMMENDATIONS[feat]['excessive']
                issues.append({'nutrient': ref['name'], 'status': 'Excessive', 'severity': 'critical', 'recommendation': rec})

            # Calculate percentage of optimal range
            opt_mid = (ref['opt_low'] + ref['opt_high']) / 2
            pct = round((val / ref['high']) * 100, 1) if ref['high'] > 0 else 0

            nutrient_analysis.append({
                'feature': feat,
                'name': ref['name'],
                'value': val,
                'unit': ref['unit'],
                'status': status,
                'severity': severity,
                'recommendation': rec,
                'optimal_range': f"{ref['opt_low']} - {ref['opt_high']} {ref['unit']}",
                'percentage': min(pct, 100)
            })

        # Overall score (0-100)
        score = sum(1 for n in nutrient_analysis if n['status'] == 'Optimal') / len(nutrient_analysis) * 100

        email = data.get('email', '')
        if email:
            log_activity(email, 'soil_report', {
                'overall_score': round(score, 1),
                'total_issues': len(issues),
                'critical_issues': len([i for i in issues if i['severity'] == 'critical'])
            })

        return jsonify({
            'success': True,
            'nutrient_analysis': nutrient_analysis,
            'issues': issues,
            'overall_score': round(score, 1),
            'total_issues': len(issues),
            'critical_issues': len([i for i in issues if i['severity'] == 'critical']),
            'warning_issues': len([i for i in issues if i['severity'] == 'warning'])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


FERTILIZER_INFO = {
    'Urea': {
        'npk': '46-0-0',
        'description': 'Urea is the most concentrated nitrogen fertilizer. It provides a quick boost of nitrogen for leafy growth and overall plant vigor.',
        'application': 'Apply 100-150 kg/ha in split doses - at sowing and during vegetative growth. Best applied in moist soil, avoid surface application in hot weather.',
        'crops': 'Best for: Wheat, Rice, Maize, Sugarcane, Cotton - crops with high nitrogen demand.',
        'tips': 'Incorporate into soil within 24 hours to prevent nitrogen loss through volatilization.'
    },
    'DAP': {
        'npk': '18-46-0',
        'description': 'Diammonium Phosphate provides both nitrogen and phosphorus. Excellent for root development and early crop establishment.',
        'application': 'Apply 50-75 kg/ha at the time of sowing/planting. Place near the seed for best results.',
        'crops': 'Best for: All crops, especially during early growth stages. Ideal for Paddy, Wheat, Cotton.',
        'tips': 'DAP has a high pH around the granule which can help reduce aluminum toxicity in acidic soils.'
    },
    '10-26-26': {
        'npk': '10-26-26',
        'description': 'A complex fertilizer with balanced phosphorus and potassium. Good for crops needing strong root and fruit development.',
        'application': 'Apply 100-125 kg/ha as basal dose before sowing.',
        'crops': 'Best for: Pulses, Oil seeds, Vegetables - crops needing good P and K.',
        'tips': 'Use in soils that already have adequate nitrogen but need P and K supplementation.'
    },
    '14-35-14': {
        'npk': '14-35-14',
        'description': 'High phosphorus fertilizer ideal for root establishment and flowering stages.',
        'application': 'Apply 75-100 kg/ha at sowing or transplanting.',
        'crops': 'Best for: Cotton, Ground Nuts, Sugarcane - crops with long duration needing strong roots.',
        'tips': 'Especially useful in newly reclaimed or low-phosphorus soils.'
    },
    '17-17-17': {
        'npk': '17-17-17',
        'description': 'A perfectly balanced NPK fertilizer providing equal parts of all three macronutrients.',
        'application': 'Apply 100-150 kg/ha as a general-purpose basal fertilizer.',
        'crops': 'Best for: All crops as a balanced starter. Works well for Maize, Barley, Millets.',
        'tips': 'Versatile fertilizer suitable when soil test shows balanced deficiency across N, P, and K.'
    },
    '20-20': {
        'npk': '20-20-0',
        'description': 'Provides equal nitrogen and phosphorus without potassium. Good when soil K is already sufficient.',
        'application': 'Apply 75-100 kg/ha at sowing and top-dress with urea if needed.',
        'crops': 'Best for: Crops in potassium-rich soils. Wheat, Maize, Paddy.',
        'tips': 'Check soil K levels before use. Add MOP separately if potassium is low.'
    },
    '28-28': {
        'npk': '28-28-0',
        'description': 'High concentration NPK without potassium. Efficient for quick N and P supply.',
        'application': 'Apply 50-75 kg/ha. Lower dose needed due to high concentration.',
        'crops': 'Best for: Millets, Tobacco, Maize - crops with moderate K needs.',
        'tips': 'Cost-effective option when only N and P supplementation is required.'
    }
}


@app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_predict():
    try:
        data = request.get_json()
        soil_type = data.get('soil_type', '')
        crop_type = data.get('crop_type', '')
        temperature = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 50))
        moisture = float(data.get('moisture', 40))
        nitrogen = float(data.get('nitrogen', 20))
        phosphorus = float(data.get('phosphorus', 10))
        potassium = float(data.get('potassium', 5))
        soil_ph = data.get('soil_ph', 'neutral')
        growth_stage = data.get('growth_stage', '')
        preference = data.get('preference', 'chemical')

        # Encode categorical
        soil_encoded = fert_encoders['soil'].transform([soil_type])[0]
        crop_encoded = fert_encoders['crop'].transform([crop_type])[0]

        # Prepare features
        features = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorus]])
        features_scaled = fert_scaler.transform(features)

        # Predict
        prediction_idx = int(fert_model.predict(features_scaled)[0])
        probabilities = fert_model.predict_proba(features_scaled)[0]
        fertilizer_name = fert_encoders['fertilizer'].inverse_transform([prediction_idx])[0]

        # Get top 3 recommendations
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_recommendations = []
        for idx in top_indices:
            fname = fert_encoders['fertilizer'].inverse_transform([idx])[0]
            prob = round(float(probabilities[idx]) * 100, 1)
            if prob > 0:
                info = FERTILIZER_INFO.get(fname, {})
                top_recommendations.append({
                    'name': fname,
                    'npk': info.get('npk', ''),
                    'confidence': prob,
                    'description': info.get('description', ''),
                    'application': info.get('application', ''),
                    'crops': info.get('crops', ''),
                    'tips': info.get('tips', '')
                })

        # Generate pH-based advice
        ph_advice = ''
        if soil_ph == 'acidic':
            ph_advice = 'Your soil is acidic. Consider applying agricultural lime (1-2 tonnes/ha) before fertilizer application to improve nutrient availability.'
        elif soil_ph == 'alkaline':
            ph_advice = 'Your soil is alkaline. Apply eleite sulfur or gypsum to lower pH. Micronutrients like iron and zinc may be less available.'
        else:
            ph_advice = 'Your soil pH is neutral, which is ideal for most nutrient availability.'

        # Growth stage advice
        stage_advice = ''
        if growth_stage == 'seedling':
            stage_advice = 'At seedling stage, phosphorus is critical for root establishment. The recommended fertilizer will support early growth.'
        elif growth_stage == 'vegetative':
            stage_advice = 'During vegetative growth, nitrogen is essential for leaf and stem development. Split nitrogen applications are recommended.'
        elif growth_stage == 'flowering':
            stage_advice = 'At flowering stage, phosphorus and potassium are important for flower development and pollination success.'
        elif growth_stage == 'fruiting':
            stage_advice = 'During fruiting, potassium is critical for fruit quality and size. Avoid excess nitrogen which can reduce fruit set.'
        else:
            stage_advice = 'Select a growth stage for more specific timing advice.'

        # Organic preference note
        organic_note = ''
        if preference == 'organic':
            organic_note = 'Since you prefer organic farming, supplement the recommended fertilizer with: Farmyard Manure (10-15 t/ha), Vermicompost (2-5 t/ha), and Neem Cake (250 kg/ha) for sustained nutrient release and soil health improvement.'
        elif preference == 'both':
            organic_note = 'For integrated nutrient management, combine the recommended chemical fertilizer with organic sources like FYM (5-10 t/ha) or vermicompost for better soil biology and sustained nutrition.'

        email = data.get('email', '')
        if email:
            log_activity(email, 'fertilizer_prediction', {
                'fertilizer': fertilizer_name,
                'soil_type': soil_type,
                'crop_type': crop_type
            })

        return jsonify({
            'success': True,
            'fertilizer': fertilizer_name,
            'info': FERTILIZER_INFO.get(fertilizer_name, {}),
            'top_recommendations': top_recommendations,
            'ph_advice': ph_advice,
            'stage_advice': stage_advice,
            'organic_note': organic_note,
            'inputs': {
                'soil_type': soil_type,
                'crop_type': crop_type,
                'temperature': temperature,
                'humidity': humidity,
                'moisture': moisture,
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


CHATBOT_KB = [
    {
        'keywords': ['hello', 'hi', 'hey', 'namaste', 'good morning', 'good evening'],
        'response': "Hello! I'm your Soil Health Assistant. Ask me anything about soil fertility, fertilizers, crop management, or nutrient deficiencies. How can I help you today?"
    },
    {
        'keywords': ['nitrogen', 'n deficiency', 'nitrogen deficient', 'yellow leaves', 'chlorosis'],
        'response': "**Nitrogen (N) Deficiency:**\n- Symptoms: Yellowing of older/lower leaves, stunted growth, pale green color.\n- Solution: Apply urea (46-0-0) at 100-150 kg/ha or ammonium sulfate (21-0-0) at 200-300 kg/ha.\n- Organic: Use farmyard manure, compost, or grow legumes (moong, urad) to fix atmospheric nitrogen.\n- Ideal range: 280-560 kg/ha."
    },
    {
        'keywords': ['phosphorus', 'p deficiency', 'phosphorus deficient', 'purple leaves'],
        'response': "**Phosphorus (P) Deficiency:**\n- Symptoms: Purple/reddish discoloration of leaves, poor root development, delayed maturity.\n- Solution: Apply DAP (18-46-0) at 50-75 kg/ha or Single Super Phosphate (SSP) at 150-200 kg/ha.\n- Organic: Add bone meal, rock phosphate, or farmyard manure.\n- Ideal range: 10-25 kg/ha."
    },
    {
        'keywords': ['potassium', 'k deficiency', 'potassium deficient', 'brown edges', 'leaf burn'],
        'response': "**Potassium (K) Deficiency:**\n- Symptoms: Brown/scorched leaf edges (leaf burn), weak stems, poor fruit quality.\n- Solution: Apply Muriate of Potash (MOP) at 50-100 kg/ha or Sulfate of Potash (SOP).\n- Organic: Use wood ash, banana peels, or green manure crops.\n- Ideal range: 200-500 kg/ha."
    },
    {
        'keywords': ['ph', 'acidic', 'alkaline', 'lime', 'soil acidity', 'soil alkalinity'],
        'response': "**Soil pH Management:**\n- Ideal pH: 6.0-7.5 for most crops.\n- Too Acidic (pH < 5.5): Apply agricultural lime (CaCO3) at 1-2 tonnes/ha. Dolomite lime also adds magnesium.\n- Too Alkaline (pH > 8.5): Apply elemental sulfur, aluminum sulfate, or use acidic mulches (pine needles).\n- Most nutrients are best available at pH 6.5-7.0."
    },
    {
        'keywords': ['organic', 'organic farming', 'compost', 'vermicompost', 'natural'],
        'response': "**Organic Soil Management:**\n- **Compost:** Apply 5-10 tonnes/ha of well-decomposed compost.\n- **Vermicompost:** Apply 2-5 tonnes/ha for rich micronutrients.\n- **Green Manuring:** Grow dhaincha, sunhemp, or cowpea and incorporate before flowering.\n- **Farmyard Manure:** Apply 10-15 tonnes/ha during field preparation.\n- **Mulching:** Use crop residue mulch to retain moisture and add organic matter."
    },
    {
        'keywords': ['fertilizer', 'npk', 'urea', 'dap', 'mop', 'fertiliser', 'chemical'],
        'response': "**NPK Fertilizer Guide:**\n- **Urea (46-0-0):** Primary nitrogen source. Apply in split doses.\n- **DAP (18-46-0):** Good source of N and P. Apply at sowing.\n- **MOP (0-0-60):** Potassium source. Apply before last irrigation.\n- **SSP (0-16-0-11):** Provides P and S. Good for deficient soils.\n- Always do soil testing before applying fertilizers."
    },
    {
        'keywords': ['wheat', 'rice', 'paddy', 'maize', 'corn', 'cotton', 'sugarcane', 'crop'],
        'response': "**Crop-Specific Tips:**\n- **Wheat:** Needs good N and P. Apply 120 kg N + 60 kg P2O5 + 40 kg K2O per ha.\n- **Rice/Paddy:** Needs flooded conditions. Apply N in 3 splits. Zinc is critical.\n- **Maize:** Heavy feeder. Needs 150 kg N + 75 kg P2O5 + 50 kg K2O per ha.\n- **Cotton:** Long duration crop. Needs balanced NPK + micronutrients.\n- **Sugarcane:** Very heavy feeder. Needs 250-300 kg N per ha in splits."
    },
    {
        'keywords': ['micronutrient', 'zinc', 'iron', 'manganese', 'copper', 'boron', 'micronutrients'],
        'response': "**Micronutrient Guide:**\n- **Zinc (Zn):** Apply ZnSO4 at 25 kg/ha. Critical for rice, wheat, maize.\n- **Iron (Fe):** Apply FeSO4 at 50 kg/ha or chelated iron spray.\n- **Manganese (Mn):** Apply MnSO4 at 20-25 kg/ha. Spray at 0.5%.\n- **Copper (Cu):** Apply CuSO4 at 10-15 kg/ha.\n- **Boron (B):** Apply borax at 10-15 kg/ha."
    },
    {
        'keywords': ['irrigation', 'water', 'drip', 'flood', 'rain', 'moisture'],
        'response': "**Irrigation & Soil Health:**\n- Drip irrigation saves 40-60% water and improves nutrient uptake.\n- Apply fertilizers with irrigation (fertigation) for better efficiency.\n- Maintain soil moisture at 50-70% field capacity.\n- Mulching reduces water evaporation by 25-50%."
    },
    {
        'keywords': ['test', 'soil test', 'testing', 'sample', 'lab', 'analysis'],
        'response': "**How to Get Your Soil Tested:**\n1. Collect samples from 15-20 spots across your field (0-15 cm depth).\n2. Mix well and take 500g as representative sample.\n3. Send to nearest soil testing lab.\n4. Test for: pH, EC, OC, N, P, K, S, Zn, Fe, Cu, Mn, B.\n5. Results take 7-15 days."
    },
    {
        'keywords': ['season', 'kharif', 'rabi', 'summer', 'winter', 'monsoon', 'rainy'],
        'response': "**Seasonal Soil Management:**\n- **Kharif (Monsoon):** Focus on drainage, apply P & K before sowing.\n- **Rabi (Winter):** Good time for lime application. Apply full dose of P & K at sowing.\n- **Summer:** Minimal tillage, mulching to conserve moisture, grow green manure crops."
    },
    {
        'keywords': ['pest', 'disease', 'insect', 'fungus', 'weed', 'protection'],
        'response': "**Soil Health & Pest Management:**\n- Healthy soil = fewer pests and diseases.\n- Crop rotation breaks pest cycles.\n- Neem cake (250 kg/ha) is both organic fertilizer and pest repellent.\n- Excessive nitrogen attracts pests."
    },
    {
        'keywords': ['thank', 'thanks', 'dhanyavaad', 'shukriya'],
        'response': "You're welcome! Happy to help with your farming needs. Feel free to ask more questions about soil health anytime."
    },
    {
        'keywords': ['help', 'what can you do', 'how to use', 'features'],
        'response': "**I can help you with:**\n- Nutrient deficiencies (N, P, K, micronutrients)\n- Fertilizer recommendations\n- Soil pH management\n- Crop-specific guidance\n- Irrigation tips\n- Seasonal soil care\n- Pest & disease prevention\n\nJust type your question naturally!"
    }
]


def get_chatbot_response(message):
    msg = message.lower().strip()
    scores = []
    for entry in CHATBOT_KB:
        score = sum(1 for kw in entry['keywords'] if kw in msg)
        if score > 0:
            scores.append((score, entry['response']))
    if scores:
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]
    return "I'm not sure about that. I specialize in soil fertility and farming topics. Try asking about:\n- Nutrient deficiencies\n- Fertilizer recommendations\n- Soil pH management\n- Crop-specific advice\n- Organic farming methods\n\nType 'help' to see all topics!"


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400

        email = data.get('email', '')
        if email:
            log_activity(email, 'chat', {'message': message})

        reply = get_chatbot_response(message)
        return jsonify({'success': True, 'response': reply, 'source': 'local'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        first_name = data.get('firstName', '').strip()
        last_name = data.get('lastName', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        occupation = data.get('occupation', '').strip()
        country = data.get('country', '').strip()
        phone = data.get('phone', '').strip()

        if not all([first_name, last_name, email, password, country]):
            return jsonify({'success': False, 'error': 'All required fields must be filled'}), 400

        if len(password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400

        user, error = create_user(first_name, last_name, email, password, occupation, country, phone)
        if error:
            status = 409 if 'already' in error.lower() else 500
            return jsonify({'success': False, 'error': error}), status

        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': user
        }), 201

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password are required'}), 400

        user, error = authenticate_user(email, password)
        if error:
            return jsonify({'success': False, 'error': error}), 401

        log_activity(email, 'login', {'email': email})

        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': user
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        if email:
            log_activity(email, 'logout', {'email': email})
        return jsonify({'success': True, 'message': 'Logged out'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/profile/<email>', methods=['GET'])
def get_profile(email):
    try:
        user, error = get_user_by_email(email.strip().lower())
        if error:
            return jsonify({'success': False, 'error': error}), 404

        return jsonify({'success': True, 'user': user})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/profile/<email>', methods=['PUT'])
def update_profile(email):
    try:
        data = request.get_json()
        user, error = update_user(email.strip().lower(), data)
        if error:
            status = 404 if 'not found' in error.lower() else 400
            return jsonify({'success': False, 'error': error}), status

        return jsonify({
            'success': True,
            'message': 'Profile updated',
            'user': user
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/activity/<email>', methods=['GET'])
def get_activity(email):
    try:
        limit = request.args.get('limit', 50, type=int)
        activities, error = get_user_activity(email.strip().lower(), limit)
        if error:
            return jsonify({'success': False, 'error': error}), 500

        return jsonify({'success': True, 'activities': activities})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats/<email>', methods=['GET'])
def get_stats(email):
    try:
        email_lower = email.strip().lower()
        stats, error = get_user_stats(email_lower)
        if error:
            return jsonify({'success': False, 'error': error}), 500

        trend, trend_err = get_user_monthly_activity(email_lower)
        if trend_err:
            trend = {'labels': [], 'soil_analysis': [], 'fertilizer': [], 'chatbot': []}

        return jsonify({'success': True, 'stats': stats, 'trend': trend})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recent-analyses/<email>', methods=['GET'])
def get_recent_analyses(email):
    try:
        analyses, error = get_recent_soil_analyses(email.strip().lower(), limit=5)
        if error:
            return jsonify({'success': False, 'error': error}), 500

        return jsonify({'success': True, 'analyses': analyses})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fertilizer/options', methods=['GET'])
def fertilizer_options():
    return jsonify({
        'success': True,
        'soil_types': SOIL_TYPES,
        'crop_types': CROP_TYPES,
        'fertilizer_names': FERTILIZER_NAMES
    })


@app.route('/api/fertilizer/recommend', methods=['POST'])
def fertilizer_recommend():
    try:
        data = request.get_json()
        soil_type = data.get('soil_type', '')
        crop_type = data.get('crop_type', '')
        nitrogen = data.get('nitrogen')
        potassium = data.get('potassium')
        phosphorous = data.get('phosphorous')

        # Filter CSV data based on selections
        matches = FERTILIZER_CSV_DATA
        if soil_type:
            matches = [r for r in matches if r['soil_type'].lower() == soil_type.lower()]
        if crop_type:
            matches = [r for r in matches if r['crop_type'].lower() == crop_type.lower()]
        if nitrogen is not None:
            n = int(nitrogen)
            matches = [r for r in matches if abs(r['nitrogen'] - n) <= 10]
        if potassium is not None:
            k = int(potassium)
            matches = [r for r in matches if abs(r['potassium'] - k) <= 10]
        if phosphorous is not None:
            p = int(phosphorous)
            matches = [r for r in matches if abs(r['phosphorous'] - p) <= 10]

        if not matches:
            return jsonify({'success': True, 'recommendations': [], 'message': 'No matching records found. Try different filters.'})

        # Count fertilizer occurrences
        fert_count = {}
        for r in matches:
            fname = r['fertilizer']
            if fname not in fert_count:
                fert_count[fname] = {'count': 0, 'avg_n': 0, 'avg_k': 0, 'avg_p': 0, 'records': []}
            fert_count[fname]['count'] += 1
            fert_count[fname]['records'].append(r)

        # Calculate averages and build recommendations
        recommendations = []
        for fname, info in sorted(fert_count.items(), key=lambda x: x[1]['count'], reverse=True):
            recs = info['records']
            avg_n = sum(r['nitrogen'] for r in recs) / len(recs)
            avg_k = sum(r['potassium'] for r in recs) / len(recs)
            avg_p = sum(r['phosphorous'] for r in recs) / len(recs)
            confidence = round(info['count'] / len(matches) * 100, 1)
            recommendations.append({
                'fertilizer': fname,
                'confidence': confidence,
                'match_count': info['count'],
                'avg_nitrogen': round(avg_n, 1),
                'avg_potassium': round(avg_k, 1),
                'avg_phosphorous': round(avg_p, 1),
                'sample_crops': list(set(r['crop_type'] for r in recs))[:5],
                'sample_soils': list(set(r['soil_type'] for r in recs))[:5]
            })

        email = data.get('email', '')
        if email:
            log_activity(email, 'fertilizer_selector', {
                'soil_type': soil_type,
                'crop_type': crop_type,
                'top_result': recommendations[0]['fertilizer'] if recommendations else None,
                'total_matches': len(matches)
            })

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'total_matches': len(matches),
            'filters': {'soil_type': soil_type, 'crop_type': crop_type}
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
