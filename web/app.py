"""
Flask Web Application cho hệ thống phát hiện gian lận bảo hiểm
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
from tensorflow import keras

app = Flask(__name__)

# Global variables cho models
models = {}
preprocessor = None
feature_names = None


def load_all_models():
    """Load tất cả models và preprocessor"""
    global models, preprocessor, feature_names

    model_dir = '../models'

    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        data = joblib.load(preprocessor_path)
        preprocessor = data
        feature_names = data['feature_names']
        print("[OK] Loaded preprocessor")

    # Load XGBoost
    xgb_path = os.path.join(model_dir, 'xgboost_model.pkl')
    if os.path.exists(xgb_path):
        models['xgboost'] = joblib.load(xgb_path)
        print("[OK] Loaded XGBoost")

    # Load Random Forest
    rf_path = os.path.join(model_dir, 'random_forest_model.pkl')
    if os.path.exists(rf_path):
        models['random_forest'] = joblib.load(rf_path)
        print("[OK] Loaded Random Forest")

    # Load ANN
    ann_path = os.path.join(model_dir, 'ann_model.h5')
    if os.path.exists(ann_path):
        models['ann'] = keras.models.load_model(ann_path)
        print("[OK] Loaded ANN")


@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def explain_prediction(data):
    reasons = []

    if data['claim_amount'] > 0.5 * data['income']:
        reasons.append("Số tiền yêu cầu bảo hiểm quá cao so với thu nhập")

    if data['num_claims'] > 3:
        reasons.append("Số lần yêu cầu bảo hiểm nhiều bất thường")

    if data['credit_score'] < 500:
        reasons.append("Điểm tín dụng thấp")

    if data['policy_duration'] < 12:
        reasons.append("Thời gian tham gia bảo hiểm ngắn")

    if data['vehicle_age'] > 10:
        reasons.append("Xe quá cũ")

    if data['employment_status'] == 'unemployed':
        reasons.append("Không có việc làm ổn định")

    return reasons


def predict():
    """API dự đoán gian lận"""
    try:
        # Lấy dữ liệu từ form
        data = request.json

        # Chuẩn bị features
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

        # Encode categorical features
        # Feature engineering (GIỐNG Y HỆT TRAIN)
        features['claim_to_income_ratio'] = features['claim_amount'] / \
            (features['income'] + 1e-6)

        features['avg_claim_per_num_claim'] = features['claim_amount'] / \
            (features['num_claims'] + 1e-6)

        features['age_x_policy_duration'] = features['age'] * \
            features['policy_duration']

        features['high_claim_relative_to_income'] = int(
            features['claim_amount'] > 0.5 * features['income']
        )
        for col in ['employment_status', 'education', 'marital_status', 'claim_type']:
            le = preprocessor['label_encoders'][col]
            features[col] = le.transform([features[col]])[0]

        # Tạo feature array
        feature_array = np.array([[
            features['age'],
            features['income'],
            features['claim_amount'],
            features['num_claims'],
            features['policy_duration'],
            features['num_dependents'],
            features['vehicle_age'],
            features['credit_score'],
            features['employment_status'],
            features['education'],
            features['marital_status'],
            features['claim_type'],
            features['claim_to_income_ratio'],
            features['avg_claim_per_num_claim'],
            features['age_x_policy_duration'],
            features['high_claim_relative_to_income']
        ]])

        # Scale features
        feature_scaled = preprocessor['scaler'].transform(feature_array)

        # Dự đoán với tất cả models
        predictions = {}

        if 'xgboost' in models:
            pred = models['xgboost'].predict(feature_scaled)[0]
            proba = models['xgboost'].predict_proba(feature_scaled)[0][1]
            predictions['xgboost'] = {
                'prediction': int(pred),
                'probability': float(proba),
                'confidence': float(proba * 100)
            }

        if 'random_forest' in models:
            pred = models['random_forest'].predict(feature_scaled)[0]
            proba = models['random_forest'].predict_proba(feature_scaled)[0][1]
            predictions['random_forest'] = {
                'prediction': int(pred),
                'probability': float(proba),
                'confidence': float(proba * 100)
            }

        if 'ann' in models:
            proba = models['ann'].predict(feature_scaled, verbose=0)[0][0]
            pred = 1 if proba > 0.5 else 0
            predictions['ann'] = {
                'prediction': int(pred),
                'probability': float(proba),
                'confidence': float(proba * 100)
            }

        # Tính kết quả tổng hợp (voting)
        fraud_votes = sum(1 for p in predictions.values()
                          if p['prediction'] == 1)
        avg_probability = np.mean([p['probability']
                                  for p in predictions.values()])

        final_prediction = 1 if fraud_votes >= 2 else 0
        reasons = explain_prediction(features)
        if len(reasons) == 0:
            reasons.append("Không phát hiện dấu hiệu bất thường rõ ràng")

        return jsonify({
            'success': True,
            'predictions': predictions,
            'final_prediction': final_prediction,
            'final_probability': float(avg_probability),
            'fraud_votes': fraud_votes,
            'total_models': len(predictions),
            'reasons': reasons
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/models-info')
def models_info():
    """API lấy thông tin các models"""
    info = {
        'loaded_models': list(models.keys()),
        'total_models': len(models),
        'feature_count': len(feature_names) if feature_names else 0
    }
    return jsonify(info)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("KHOI DONG WEB APPLICATION")
    print("="*60)

    # Load models
    load_all_models()
    """
Flask Web Application cho hệ thống phát hiện gian lận bảo hiểm
"""


app = Flask(__name__)

# Global variables cho models
models = {}
preprocessor = None
feature_names = None


def load_all_models():
    """Load tất cả models và preprocessor"""
    global models, preprocessor, feature_names

    model_dir = '../models'

    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        data = joblib.load(preprocessor_path)
        preprocessor = data
        feature_names = data['feature_names']
        print("[OK] Loaded preprocessor")

    # Load XGBoost
    xgb_path = os.path.join(model_dir, 'xgboost_model.pkl')
    if os.path.exists(xgb_path):
        models['xgboost'] = joblib.load(xgb_path)
        print("[OK] Loaded XGBoost")

    # Load Random Forest
    rf_path = os.path.join(model_dir, 'random_forest_model.pkl')
    if os.path.exists(rf_path):
        models['random_forest'] = joblib.load(rf_path)
        print("[OK] Loaded Random Forest")

    # Load ANN
    ann_path = os.path.join(model_dir, 'ann_model.h5')
    if os.path.exists(ann_path):
        models['ann'] = keras.models.load_model(ann_path)
        print("[OK] Loaded ANN")


@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API dự đoán gian lận"""
    try:
        # Lấy dữ liệu từ form
        data = request.json

        # Chuẩn bị features
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

        # Encode categorical features
        # Feature engineering (GIỐNG Y HỆT TRAIN)
        features['claim_to_income_ratio'] = features['claim_amount'] / \
            (features['income'] + 1e-6)

        features['avg_claim_per_num_claim'] = features['claim_amount'] / \
            (features['num_claims'] + 1e-6)

        features['age_x_policy_duration'] = features['age'] * \
            features['policy_duration']

        features['high_claim_relative_to_income'] = int(
            features['claim_amount'] > 0.5 * features['income']
        )
        for col in ['employment_status', 'education', 'marital_status', 'claim_type']:
            le = preprocessor['label_encoders'][col]
            features[col] = le.transform([features[col]])[0]

        # Tạo feature array
        feature_array = np.array([[
            features['age'],
            features['income'],
            features['claim_amount'],
            features['num_claims'],
            features['policy_duration'],
            features['num_dependents'],
            features['vehicle_age'],
            features['credit_score'],
            features['employment_status'],
            features['education'],
            features['marital_status'],
            features['claim_type'],
            features['claim_to_income_ratio'],
            features['avg_claim_per_num_claim'],
            features['age_x_policy_duration'],
            features['high_claim_relative_to_income']
        ]])

        # Scale features
        feature_scaled = preprocessor['scaler'].transform(feature_array)

        # Dự đoán với tất cả models
        predictions = {}

        if 'xgboost' in models:
            pred = models['xgboost'].predict(feature_scaled)[0]
            proba = models['xgboost'].predict_proba(feature_scaled)[0][1]
            predictions['xgboost'] = {
                'prediction': int(pred),
                'probability': float(proba),
                'confidence': float(proba * 100)
            }

        if 'random_forest' in models:
            pred = models['random_forest'].predict(feature_scaled)[0]
            proba = models['random_forest'].predict_proba(feature_scaled)[0][1]
            predictions['random_forest'] = {
                'prediction': int(pred),
                'probability': float(proba),
                'confidence': float(proba * 100)
            }

        if 'ann' in models:
            proba = models['ann'].predict(feature_scaled, verbose=0)[0][0]
            pred = 1 if proba > 0.5 else 0
            predictions['ann'] = {
                'prediction': int(pred),
                'probability': float(proba),
                'confidence': float(proba * 100)
            }

        # Tính kết quả tổng hợp (voting)
        fraud_votes = sum(1 for p in predictions.values()
                          if p['prediction'] == 1)
        avg_probability = np.mean([p['probability']
                                  for p in predictions.values()])

        final_prediction = 1 if fraud_votes >= 2 else 0

        return jsonify({
            'success': True,
            'predictions': predictions,
            'final_prediction': final_prediction,
            'final_probability': float(avg_probability),
            'fraud_votes': fraud_votes,
            'total_models': len(predictions)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/models-info')
def models_info():
    """API lấy thông tin các models"""
    info = {
        'loaded_models': list(models.keys()),
        'total_models': len(models),
        'feature_count': len(feature_names) if feature_names else 0
    }
    return jsonify(info)


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
