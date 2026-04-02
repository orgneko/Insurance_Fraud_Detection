"""
Flask Web Application cho hệ thống phát hiện gian lận bảo hiểm
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import shap
import os
from tensorflow import keras

app = Flask(__name__)

# Global variables
models = {}
preprocessor = None
feature_names = None
explainers = {}


# =========================
# LOAD MODELS
# =========================
def load_all_models():
    global models, preprocessor, feature_names, explainers

    model_dir = '../models'

    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        data = joblib.load(preprocessor_path)
        preprocessor = data
        feature_names = data['feature_names']
        print("[OK] Loaded preprocessor")

    # Load models
    if os.path.exists(os.path.join(model_dir, 'xgboost_model.pkl')):
        models['xgboost'] = joblib.load(
            os.path.join(model_dir, 'xgboost_model.pkl'))
        print("[OK] Loaded XGBoost")

    if os.path.exists(os.path.join(model_dir, 'random_forest_model.pkl')):
        models['random_forest'] = joblib.load(
            os.path.join(model_dir, 'random_forest_model.pkl'))
        print("[OK] Loaded Random Forest")

    if os.path.exists(os.path.join(model_dir, 'ann_model.h5')):
        models['ann'] = keras.models.load_model(
            os.path.join(model_dir, 'ann_model.h5'))
        print("[OK] Loaded ANN")

    # SHAP explainers
    if 'xgboost' in models:
        explainers['xgboost'] = shap.TreeExplainer(models['xgboost'])

    if 'random_forest' in models:
        explainers['random_forest'] = shap.TreeExplainer(
            models['random_forest'])


# =========================
# RULE-BASED REASONS
# =========================
def explain_rule_based(data):
    reasons = []

    if data['claim_amount'] > 0.5 * data['income']:
        reasons.append("Số tiền yêu cầu quá cao so với thu nhập")

    if data['num_claims'] > 3:
        reasons.append("Số lần yêu cầu bảo hiểm nhiều bất thường")

    if data['credit_score'] < 500:
        reasons.append("Điểm tín dụng thấp")

    if data['policy_duration'] < 12:
        reasons.append("Thời gian tham gia bảo hiểm ngắn")

    if data['vehicle_age'] > 10:
        reasons.append("Xe đã quá cũ")

    if data['employment_status'] == 'unemployed':
        reasons.append("Không có việc làm ổn định")

    if len(reasons) == 0:
        reasons.append("Không phát hiện dấu hiệu bất thường rõ ràng")

    return reasons


# =========================
# SHAP EXPLAIN
# =========================
def get_shap_explanation(model_name, feature_scaled):
    if model_name not in explainers:
        return []

    try:
        explainer = explainers[model_name]
        shap_values = explainer.shap_values(feature_scaled)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        values = shap_values[0]

        feature_importance = []
        for i, val in enumerate(values):
            feature_importance.append({
                "feature": feature_names[i],
                "impact": float(val)
            })

        feature_importance = sorted(
            feature_importance,
            key=lambda x: abs(x["impact"]),
            reverse=True
        )

        return feature_importance[:5]

    except Exception as e:
        print("SHAP error:", e)
        return []


# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # ===== Parse input =====
        features = {
            'age': float(data['age']),
            'income': float(data['income']),
            'claim_amount': float(data['claim_amount']),
            'num_claims': int(data['num_claims']),
            'policy_duration': int(data['policy_duration']),
            'num_dependents': int(data['num_dependents']),
            'vehicle_age': int(data['vehicle_age']),
            'credit_score': int(data['credit_score']),
            'employment_status': data['employment_status'],
            'education': data['education'],
            'marital_status': data['marital_status'],
            'claim_type': data['claim_type']
        }

        # ===== Feature engineering =====
        features['claim_to_income_ratio'] = features['claim_amount'] / \
            (features['income'] + 1e-6)
        features['avg_claim_per_num_claim'] = features['claim_amount'] / \
            (features['num_claims'] + 1e-6)
        features['age_x_policy_duration'] = features['age'] * \
            features['policy_duration']
        features['high_claim_relative_to_income'] = int(
            features['claim_amount'] > 0.5 * features['income'])

        # Encode categorical
        for col in ['employment_status', 'education', 'marital_status', 'claim_type']:
            le = preprocessor['label_encoders'][col]
            features[col] = le.transform([features[col]])[0]

        # ===== Feature array =====
        feature_array = np.array([[features[f] for f in feature_names]])
        feature_scaled = preprocessor['scaler'].transform(feature_array)

        # ===== Predict =====
        predictions = {}
        shap_details = {}

        for name, model in models.items():
            if name == 'ann':
                proba = model.predict(feature_scaled, verbose=0)[0][0]
                pred = 1 if proba > 0.5 else 0
            else:
                pred = model.predict(feature_scaled)[0]
                proba = model.predict_proba(feature_scaled)[0][1]

            predictions[name] = {
                'prediction': int(pred),
                'probability': float(proba)
            }

            # SHAP cho từng model
            shap_details[name] = get_shap_explanation(name, feature_scaled)

        # ===== Ensemble =====
        fraud_votes = sum(1 for p in predictions.values()
                          if p['prediction'] == 1)
        avg_probability = np.mean([p['probability']
                                  for p in predictions.values()])
        final_prediction = 1 if fraud_votes >= 2 else 0

        # ===== Reasons =====
        reasons = explain_rule_based(features)

        return jsonify({
            'success': True,
            'predictions': predictions,
            'final_prediction': final_prediction,
            'final_probability': float(avg_probability),
            'fraud_votes': fraud_votes,
            'total_models': len(predictions),
            'reasons': reasons,
            'shap': shap_details
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/models-info')
def models_info():
    """API lấy thông tin các models"""
    info = {
        'loaded_models': list(models.keys()),
        'total_models': len(models),
        'feature_count': len(feature_names) if feature_names else 0
    }
    return jsonify(info)


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("KHOI DONG WEB APPLICATION")
    print("="*60)

    # Load models
    load_all_models()

    # In thông tin hướng dẫn
    print("\n[INFO] Huong dan su dung:")
    print("1. Mo trinh duyet (Chrome, Edge, Firefox, ...)")
    print("2. Truy cap dia chi: http://localhost:5000")
    print("3. Nhap thong tin bao hiem va nhan 'Phat hien Gian lan'")
    print("\nDang khoi dong server...")

    # Chạy server
    # Note: debug=True sẽ tự động reload khi sửa code
    app.run(debug=True, host='0.0.0.0', port=5000)

    # In thông tin hướng dẫn
    print("\n[INFO] Huong dan su dung:")
    print("1. Mo trinh duyet (Chrome, Edge, Firefox, ...)")
    print("2. Truy cap dia chi: http://localhost:5000")
    print("3. Nhap thong tin bao hiem va nhan 'Phat hien Gian lan'")
    print("\nDang khoi dong server...")

    # Chạy server
    # Note: debug=True sẽ tự động reload khi sửa code
    app.run(debug=True, host='0.0.0.0', port=5000)
