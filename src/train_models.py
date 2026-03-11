"""
Module training các mô hình ML: XGBoost, Random Forest, ANN
"""

import numpy as np
import pandas as pd
import joblib
import os
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """Class để train và quản lý các mô hình"""
    
    def __init__(self):
        self.models = {}
        self.training_history = {}
        
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train mô hình XGBoost
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        start_time = time.time()
        
        # Cấu hình model
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Train model
        print("\nDang training...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Đánh giá
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"\n[OK] Training hoan thanh!")
        print(f"   Thoi gian: {training_time:.2f}s")
        print(f"   Train Accuracy: {train_score*100:.2f}%")
        print(f"   Test Accuracy: {test_score*100:.2f}%")
        
        # Lưu model
        self.models['xgboost'] = model
        self.training_history['xgboost'] = {
            'training_time': training_time,
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
        
        return model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train mô hình Random Forest
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        start_time = time.time()
        
        # Khởi tạo model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        print("\nDang training...")
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Đánh giá
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"\n[OK] Training hoan thanh!")
        print(f"   Thoi gian: {training_time:.2f}s")
        print(f"   Train Accuracy: {train_score*100:.2f}%")
        print(f"   Test Accuracy: {test_score*100:.2f}%")
        
        # Lưu model
        self.models['random_forest'] = model
        self.training_history['random_forest'] = {
            'training_time': training_time,
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
        
        return model
    
    def train_ann(self, X_train, y_train, X_test, y_test):
        """
        Train mô hình ANN (Artificial Neural Network)
        """
        print("\n" + "="*60)
        print("TRAINING ANN MODEL")
        print("="*60)
        
        start_time = time.time()
        
        # Khởi tạo model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nDang training...")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        training_time = time.time() - start_time
        
        # Đánh giá
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n[OK] Training hoan thanh!")
        print(f"   Thoi gian: {training_time:.2f}s")
        print(f"   Train Accuracy: {train_acc*100:.2f}%")
        print(f"   Test Accuracy: {test_acc*100:.2f}%")
        
        # Lưu model
        self.models['ann'] = model
        self.training_history['ann'] = {
            'training_time': training_time,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'history': history.history
        }
        
        return model
    
    def save_models(self, output_dir='models'):
        """Lưu tất cả các models"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        # Lưu XGBoost
        if 'xgboost' in self.models:
            path = os.path.join(output_dir, 'xgboost_model.pkl')
            joblib.dump(self.models['xgboost'], path)
            print(f"[OK] XGBoost saved: {path}")
        
        # Lưu Random Forest
        if 'random_forest' in self.models:
            path = os.path.join(output_dir, 'random_forest_model.pkl')
            joblib.dump(self.models['random_forest'], path)
            print(f"[OK] Random Forest saved: {path}")
        
        # Lưu ANN
        if 'ann' in self.models:
            path = os.path.join(output_dir, 'ann_model.h5')
            self.models['ann'].save(path)
            print(f"[OK] ANN saved: {path}")
        
        # Lưu training history
        history_path = os.path.join(output_dir, 'training_history.pkl')
        joblib.dump(self.training_history, history_path)
        print(f"[OK] Training history saved: {history_path}")
    
    def plot_training_comparison(self, output_dir='results'):
        """Vẽ biểu đồ so sánh các models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Chuẩn bị dữ liệu
        models_names = list(self.training_history.keys())
        train_accs = [self.training_history[m]['train_accuracy'] * 100 for m in models_names]
        test_accs = [self.training_history[m]['test_accuracy'] * 100 for m in models_names]
        times = [self.training_history[m]['training_time'] for m in models_names]
        
        # Tạo figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Accuracy comparison
        x = np.arange(len(models_names))
        width = 0.35
        
        axes[0].bar(x - width/2, train_accs, width, label='Train Accuracy', color='#4CAF50')
        axes[0].bar(x + width/2, test_accs, width, label='Test Accuracy', color='#2196F3')
        axes[0].set_xlabel('Models', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.upper() for m in models_names])
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Training time comparison
        axes[1].bar(models_names, times, color=['#FF9800', '#9C27B0', '#F44336'])
        axes[1].set_xlabel('Models', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        axes[1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        # Sửa lỗi set_xticklabels
        axes[1].set_xticks(range(len(models_names)))
        axes[1].set_xticklabels([m.upper() for m in models_names])
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"[OK] Da luu bieu do so sanh tai {output_dir}/training_comparison.png")
        plt.close()


def main():
    """Hàm chính để train tất cả models"""
    print("\n" + "="*60)
    print("BAT DAU TRAINING MODELS")
    print("="*60)
    
    # Load dữ liệu
    print("\nLoading du lieu...")
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    
    print(f"[OK] Da load du lieu:")
    print(f"   - X_train shape: {X_train.shape}")
    print(f"   - X_test shape: {X_test.shape}")
    
    # Khởi tạo trainer
    trainer = ModelTrainer()
    
    # Train các models
    trainer.train_xgboost(X_train, y_train, X_test, y_test)
    trainer.train_random_forest(X_train, y_train, X_test, y_test)
    trainer.train_ann(X_train, y_train, X_test, y_test)
    
    # Lưu models
    trainer.save_models()
    
    # Vẽ biểu đồ so sánh
    trainer.plot_training_comparison()
    
    print("\n" + "="*60)
    print("HOAN THANH TRAINING TAT CA MODELS!")
    print("="*60)


if __name__ == "__main__":
    main()
