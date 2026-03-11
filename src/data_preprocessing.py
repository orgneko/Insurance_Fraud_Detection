"""
Module xử lý và chuẩn bị dữ liệu cho mô hình phát hiện gian lận bảo hiểm
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os


class DataPreprocessor:
    """Class xử lý dữ liệu bảo hiểm"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def generate_sample_data(self, n_samples=10000, fraud_ratio=0.15):
        """
        Tạo dữ liệu mẫu về gian lận bảo hiểm
        
        Args:
            n_samples: Số lượng mẫu
            fraud_ratio: Tỷ lệ gian lận
        """
        np.random.seed(42)
        
        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud
        
        # Tạo dữ liệu cho trường hợp bình thường
        normal_data = {
            'age': np.random.normal(40, 12, n_normal).clip(18, 80),
            'income': np.random.normal(15000000, 8000000, n_normal).clip(5000000, 100000000),
            'claim_amount': np.random.normal(5000000, 3000000, n_normal).clip(500000, 50000000),
            'num_claims': np.random.poisson(1.5, n_normal).clip(0, 10),
            'policy_duration': np.random.normal(36, 24, n_normal).clip(1, 120),
            'num_dependents': np.random.poisson(2, n_normal).clip(0, 8),
            'vehicle_age': np.random.normal(5, 3, n_normal).clip(0, 20),
            'credit_score': np.random.normal(700, 80, n_normal).clip(300, 850),
            'employment_status': np.random.choice(['employed', 'self_employed', 'unemployed'], 
                                                  n_normal, p=[0.7, 0.2, 0.1]),
            'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], 
                                         n_normal, p=[0.3, 0.5, 0.15, 0.05]),
            'marital_status': np.random.choice(['single', 'married', 'divorced'], 
                                              n_normal, p=[0.3, 0.6, 0.1]),
            'claim_type': np.random.choice(['accident', 'theft', 'health', 'property'], 
                                          n_normal, p=[0.4, 0.2, 0.3, 0.1]),
            'is_fraud': 0
        }
        
        # Tạo dữ liệu cho trường hợp gian lận (có đặc điểm bất thường)
        fraud_data = {
            'age': np.random.normal(35, 10, n_fraud).clip(18, 80),
            'income': np.random.normal(8000000, 5000000, n_fraud).clip(3000000, 50000000),
            'claim_amount': np.random.normal(15000000, 8000000, n_fraud).clip(5000000, 100000000),
            'num_claims': np.random.poisson(4, n_fraud).clip(2, 15),
            'policy_duration': np.random.normal(12, 8, n_fraud).clip(1, 48),
            'num_dependents': np.random.poisson(3, n_fraud).clip(0, 10),
            'vehicle_age': np.random.normal(8, 4, n_fraud).clip(0, 25),
            'credit_score': np.random.normal(550, 100, n_fraud).clip(300, 750),
            'employment_status': np.random.choice(['employed', 'self_employed', 'unemployed'], 
                                                  n_fraud, p=[0.4, 0.3, 0.3]),
            'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], 
                                         n_fraud, p=[0.5, 0.4, 0.08, 0.02]),
            'marital_status': np.random.choice(['single', 'married', 'divorced'], 
                                              n_fraud, p=[0.4, 0.4, 0.2]),
            'claim_type': np.random.choice(['accident', 'theft', 'health', 'property'], 
                                          n_fraud, p=[0.3, 0.4, 0.2, 0.1]),
            'is_fraud': 1
        }
        
        # Kết hợp dữ liệu
        df_normal = pd.DataFrame(normal_data)
        df_fraud = pd.DataFrame(fraud_data)
        df = pd.concat([df_normal, df_fraud], ignore_index=True)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode các features dạng categorical
        
        Args:
            df: DataFrame
            fit: Nếu True, fit encoder mới; nếu False, dùng encoder đã fit
        """
        df_encoded = df.copy()
        categorical_cols = ['employment_status', 'education', 'marital_status', 'claim_type']
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        return df_encoded
    
    def prepare_features(self, df, fit=True):
        """
        Chuẩn bị features cho training
        
        Args:
            df: DataFrame
            fit: Nếu True, fit scaler mới; nếu False, dùng scaler đã fit
        """
        # Encode categorical
        df_encoded = self.encode_categorical_features(df, fit=fit)
        
        # Tách features và target
        X = df_encoded.drop('is_fraud', axis=1)
        y = df_encoded['is_fraud']
        
        if fit:
            self.feature_names = X.columns.tolist()
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def balance_dataset(self, X, y):
        """
        Cân bằng dataset sử dụng SMOTE
        
        Args:
            X: Features
            y: Target
        """
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Lưu preprocessor"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)
        print(f"[OK] Da luu preprocessor tai {path}")
    
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """Load preprocessor"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        print(f"[OK] Da load preprocessor tu {path}")


def main():
    """Hàm chính để test preprocessing"""
    print("--- Bat dau xu ly du lieu ---")
    
    # Khởi tạo preprocessor
    preprocessor = DataPreprocessor()
    
    # Tạo dữ liệu mẫu
    print("\nTao du lieu mau...")
    df = preprocessor.generate_sample_data(n_samples=10000, fraud_ratio=0.15)
    
    # Lưu raw data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/insurance_data.csv', index=False)
    print(f"[OK] Da luu du lieu tai data/insurance_data.csv")
    
    # Thống kê
    print(f"\nThong ke du lieu:")
    print(f"   - Tong so mau: {len(df)}")
    print(f"   - So mau gian lan: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"   - So mau binh thuong: {(1-df['is_fraud']).sum()} ({(1-df['is_fraud'].mean())*100:.2f}%)")
    
    # Prepare features
    print("\nChuan bi features...")
    X, y = preprocessor.prepare_features(df, fit=True)
    
    # Balance dataset
    print("\nCan bang dataset voi SMOTE...")
    X_balanced, y_balanced = preprocessor.balance_dataset(X, y)
    print(f"   - So mau sau khi balance: {len(X_balanced)}")
    
    # Split data
    print("\nChia du lieu train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    print(f"   - Train set: {len(X_train)} mau")
    print(f"   - Test set: {len(X_test)} mau")
    
    # Lưu processed data
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    print("\n[OK] Da luu du lieu da xu ly")
    
    # Lưu preprocessor
    preprocessor.save_preprocessor()
    
    print("\n--- Hoan thanh xu ly du lieu ---")


if __name__ == "__main__":
    main()
