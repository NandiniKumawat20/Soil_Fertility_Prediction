from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import re
import csv
import traceback
from db import create_user, authenticate_user, get_user_by_email, update_user, log_activity, get_user_activity, get_user_stats, get_user_monthly_activity, get_recent_soil_analyses, save_feedback, get_all_feedback, MONGO_CONNECTED

# Dataset-based fertilizer predictor (NPK_Distance_Score)
try:
    from fertilizer_predictor_dataset_model import FertilizerDatasetPredictor
except Exception:
    FertilizerDatasetPredictor = None


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'mongo': MONGO_CONNECTED})


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Fertilizer dataset predictor (optional)
_fert_dataset_predictor = None

# Fertilizer ML feature removed



# Fertilizer selector feature removed (CSV-based + ML-based) 
SOIL_TYPES = []
CROP_TYPES = []
FERTILIZER_NAMES = []
FERTILIZER_CSV_DATA = []


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
        data = request.get_json() or {}

        # Build feature vector matching the scaler/model.
        # If some features are missing from the frontend payload,
        # fill them with 0 (safer than crashing) so analysis can still run.
        values = []
        for f in FEATURES:
            raw = data.get(f, 0)
            values.append(float(raw))
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

        # Optional: also provide fertilizer suggestion from dataset model
        fertilizer_suggestion = None
        try:
            if _fert_dataset_predictor is not None:
                # Use provided soil/crop context if available; otherwise fall back.
                # Frontend for soil analysis may not send these keys.
                soil_type = data.get('soil_type', '') or 'Loamy'
                crop_type = data.get('crop_type', '') or 'Wheat'
                # If temperature/humidity/moisture are not provided, use reasonable defaults.
                temperature = float(data.get('temperature', 25))
                humidity = float(data.get('humidity', 50))
                moisture = float(data.get('moisture', 40))

                fert_result = _fert_dataset_predictor.predict_best_fertilizer(
                    soil_type=soil_type,
                    crop_type=crop_type,
                    temperature=temperature,
                    humidity=humidity,
                    moisture=moisture,
                )
                fertilizer_suggestion = {
                    'fertilizer_name': fert_result.get('fertilizer_name'),
                    'predicted_score': fert_result.get('predicted_score'),
                }
        except Exception:
            fertilizer_suggestion = None

        return jsonify({
            'success': True,
            'prediction': prediction,
            'label': LABELS[prediction],
            'description': DESCRIPTIONS[prediction],
            'confidence': round(float(max(probabilities)) * 100, 1),
            'probabilities': {
                LABELS[i]: round(float(p) * 100, 1) for i, p in enumerate(probabilities)
            },
            'fertilizer_suggestion': fertilizer_suggestion
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_predict():
    try:
        data = request.get_json() or {}
        email = data.get('email', '')

        if _fert_dataset_predictor is None:
            return jsonify({'success': False, 'error': 'Fertility dataset model not available'}), 500

        soil_type = data.get('soil_type', '')
        crop_type = data.get('crop_type', '')

        # Temperature/Humidity/Moisture are numeric
        temperature = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 50))
        moisture = float(data.get('moisture', 40))

        result = _fert_dataset_predictor.predict_best_fertilizer(
            soil_type=soil_type,
            crop_type=crop_type,
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
        )

        # Build response compatible with existing frontend
        top = {
            'name': result['fertilizer_name'],
            'npk': None,
            'confidence': round(result['predicted_score'], 2),
            'description': '',
            'application': '',
            'crops': '',
            'tips': ''
        }

        # Alternatives: we simply reuse top_k filtering by asking predictor model many times.
        # For simplicity (and to keep it stable), we return duplicates if not available.
        response = {
            'success': True,
            'top_recommendations': [top],
            'ph_advice': '',
            'stage_advice': '',
            'organic_note': ''
        }

        if email:
            log_activity(email, 'fertilizer_prediction', {
                'fertilizer': top['name'],
                'top_result': top['name'],
                'score': result['predicted_score']
            })

        return jsonify(response)
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



# Fertilizer ML feature removed
FERTILIZER_INFO = {}

# Dataset-based fertilizer recommendation (from Fertility_Dataset.csv)
try:
    if FertilizerDatasetPredictor is not None:
        _fert_dataset_predictor = FertilizerDatasetPredictor(
            model_path="fertility_dataset_npk_distance_rf.joblib",
            dataset_path="Fertility_Dataset.csv",
        )
    else:
        _fert_dataset_predictor = None
except Exception:
    _fert_dataset_predictor = None




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
            trend = {'labels': [], 'soil_analysis': [], 'chatbot': []}


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


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        rating = data.get('rating', 0)
        features = data.get('features', [])
        ease = data.get('ease', '')
        recommend = data.get('recommend', '')
        comment = data.get('comment', '')

        if email:
            log_activity(email, 'feedback', {'rating': rating})

        success, error = save_feedback(email, rating, features, ease, recommend, comment)
        if error:
            return jsonify({'success': False, 'error': error}), 500

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    try:
        feedbacks, error = get_all_feedback(limit=50)
        if error:
            return jsonify({'success': False, 'error': error}), 500

        return jsonify({'success': True, 'feedbacks': feedbacks})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
