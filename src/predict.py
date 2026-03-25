"""
Script dự đoán đơn lẻ - Sử dụng models đã train để dự đoán một trường hợp cụ thể
"""

import numpy as np
import joblib
from tensorflow import keras
import os


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


class FraudPredictor:
    """Class để dự đoán gian lận"""

    def __init__(self, model_dir='models'):
        self.models = {}
        self.preprocessor = None
        self.load_models(model_dir)

    def load_models(self, model_dir):
        """Load tất cả models"""
        print("📂 Loading models...")

        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            print("✅ Loaded preprocessor")

        # Load XGBoost
        xgb_path = os.path.join(model_dir, 'xgboost_model.pkl')
        if os.path.exists(xgb_path):
            self.models['xgboost'] = joblib.load(xgb_path)
            print("✅ Loaded XGBoost")

        # Load Random Forest
        rf_path = os.path.join(model_dir, 'random_forest_model.pkl')
        if os.path.exists(rf_path):
            self.models['random_forest'] = joblib.load(rf_path)
            print("✅ Loaded Random Forest")

        # Load ANN
        ann_path = os.path.join(model_dir, 'ann_model.h5')
        if os.path.exists(ann_path):
            self.models['ann'] = keras.models.load_model(ann_path)
            print("✅ Loaded ANN")

    def predict(self, case_data):
        """
        Dự đoán một trường hợp

        Args:
            case_data: dict với các keys:
                - age, income, claim_amount, num_claims, policy_duration,
                - num_dependents, vehicle_age, credit_score,
                - employment_status, education, marital_status, claim_type

        Returns:
            dict với predictions từ tất cả models
        """
        # Encode categorical features
        encoded_data = case_data.copy()
        encoded_data['claim_to_income_ratio'] = encoded_data['claim_amount'] / \
            (encoded_data['income'] + 1e-6)
        encoded_data['avg_claim_per_num_claim'] = encoded_data['claim_amount'] / \
            (encoded_data['num_claims'] + 1e-6)
        encoded_data['age_x_policy_duration'] = encoded_data['age'] * \
            (encoded_data['policy_duration'])
        encoded_data['high_claim_relative_to_income'] = int(
            encoded_data['claim_amount'] > 0.5 * encoded_data['income'])

        for col in ['employment_status', 'education', 'marital_status', 'claim_type']:
            le = self.preprocessor['label_encoders'][col]
            encoded_data[col] = le.transform([encoded_data[col]])[0]

        # Tạo feature array
        feature_array = np.array([[
            encoded_data['age'],
            encoded_data['income'],
            encoded_data['claim_amount'],
            encoded_data['num_claims'],
            encoded_data['policy_duration'],
            encoded_data['num_dependents'],
            encoded_data['vehicle_age'],
            encoded_data['credit_score'],
            encoded_data['employment_status'],
            encoded_data['education'],
            encoded_data['marital_status'],
            encoded_data['claim_type'],
            encoded_data['claim_to_income_ratio'],
            encoded_data['avg_claim_per_num_claim'],
            encoded_data['age_x_policy_duration'],
            encoded_data['high_claim_relative_to_income']
        ]])

        # Scale features
        feature_scaled = self.preprocessor['scaler'].transform(feature_array)

        # Dự đoán với tất cả models
        predictions = {}

        for name, model in self.models.items():
            if name == 'ann':
                proba = model.predict(feature_scaled, verbose=0)[0][0]
                pred = 1 if proba > 0.5 else 0
            else:
                pred = model.predict(feature_scaled)[0]
                proba = model.predict_proba(feature_scaled)[0][1]

            predictions[name] = {
                'prediction': int(pred),
                'probability': float(proba),
                'label': 'GIAN LẬN' if pred == 1 else 'BÌNH THƯỜNG'
            }

        # Ensemble prediction
        fraud_votes = sum(1 for p in predictions.values()
                          if p['prediction'] == 1)
        avg_probability = np.mean([p['probability']
                                  for p in predictions.values()])

        final_prediction = 1 if fraud_votes >= 2 else 0
        reasons = explain_prediction(case_data)

        return {
            'predictions': predictions,
            'ensemble': {
                'prediction': final_prediction,
                'probability': avg_probability,
                'fraud_votes': fraud_votes,
                'total_models': len(predictions),
                'label': 'GIAN LẬN' if final_prediction == 1 else 'BÌNH THƯỜNG'
            },
            'reasons': reasons
        }


def print_results(results):
    """In kết quả đẹp"""
    print("\n" + "="*80)
    print("KẾT QUẢ DỰ ĐOÁN")
    print("="*80)

    # Kết quả từng model
    print("\nDự đoán từng Model:")
    for name, pred in results['predictions'].items():
        print(f"\n  {name.upper()}:")
        print(f"    Kết quả: {pred['label']}")
        print(f"    Xác suất gian lận: {pred['probability']*100:.2f}%")

    # Kết quả ensemble
    ensemble = results['ensemble']
    print("\n" + "-"*80)
    print("KẾT QUẢ TỔNG HỢP (Ensemble):")
    print(f"  Kết luận: {ensemble['label']}")
    print(f"  Xác suất gian lận: {ensemble['probability']*100:.2f}%")
    print(
        f"  Số models phát hiện gian lận: {ensemble['fraud_votes']}/{ensemble['total_models']}")
    print("="*80 + "\n")

    print("\nLý do nghi ngờ:")
    for r in results.get('reasons', []):
        print(f"  - {r}")


def main():
    """Ví dụ sử dụng"""
    print("\n" + "="*80)
    print("FRAUD DETECTION - SINGLE PREDICTION")
    print("="*80)

    # Khởi tạo predictor
    predictor = FraudPredictor()
    print("Số feature scaler expect:",
          predictor.preprocessor['scaler'].n_features_in_)

    # Test case 1: Trường hợp BÌNH THƯỜNG
    print("\nTEST CASE 1: Trường hợp BÌNH THƯỜNG")
    case1 = {
        'age': 40,
        'income': 20000000,
        'claim_amount': 3000000,
        'num_claims': 1,
        'policy_duration': 36,
        'num_dependents': 2,
        'vehicle_age': 3,
        'credit_score': 750,
        'employment_status': 'employed',
        'education': 'bachelor',
        'marital_status': 'married',
        'claim_type': 'accident'
    }

    results1 = predictor.predict(case1)
    print_results(results1)

    # Test case 2: Trường hợp NGHI NGỜ GIAN LẬN
    print("\nTEST CASE 2: Trường hợp NGHI NGỜ GIAN LẬN")
    case2 = {
        'age': 30,
        'income': 8000000,
        'claim_amount': 25000000,
        'num_claims': 6,
        'policy_duration': 6,
        'num_dependents': 4,
        'vehicle_age': 12,
        'credit_score': 480,
        'employment_status': 'unemployed',
        'education': 'high_school',
        'marital_status': 'divorced',
        'claim_type': 'theft'
    }

    results2 = predictor.predict(case2)
    print_results(results2)


if __name__ == "__main__":
    main()
