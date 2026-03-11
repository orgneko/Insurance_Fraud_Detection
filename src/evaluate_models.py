"""
Module đánh giá và so sánh hiệu suất các mô hình
"""

import numpy as np
import pandas as pd
import joblib
import os
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Class đánh giá các mô hình"""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def load_models(self, model_dir='models'):
        """Load các models đã train"""
        print("\nLoading models...")
        
        # Load XGBoost
        xgb_path = os.path.join(model_dir, 'xgboost_model.pkl')
        if os.path.exists(xgb_path):
            self.models['xgboost'] = joblib.load(xgb_path)
            print("[OK] Loaded XGBoost")
        
        # Load Random Forest
        rf_path = os.path.join(model_dir, 'random_forest_model.pkl')
        if os.path.exists(rf_path):
            self.models['random_forest'] = joblib.load(rf_path)
            print("[OK] Loaded Random Forest")
        
        # Load ANN
        ann_path = os.path.join(model_dir, 'ann_model.h5')
        if os.path.exists(ann_path):
            self.models['ann'] = keras.models.load_model(ann_path)
            print("[OK] Loaded ANN")

    def predict_all(self, X_test):
        """Dự đoán với tất cả models"""
        print("\nMaking predictions...")
        
        for name, model in self.models.items():
            if name == 'ann':
                # ANN trả về probabilities
                y_pred_proba = model.predict(X_test, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                self.predictions[name] = {
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba.flatten()
                }
            else:
                # XGBoost và Random Forest
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                self.predictions[name] = {
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
            print(f"[OK] Predicted with {name.upper()}")
    
    def calculate_metrics(self, y_test):
        """Tính toán các metrics cho tất cả models"""
        print("\nCalculating metrics...")
        
        for name, preds in self.predictions.items():
            y_pred = preds['y_pred']
            y_pred_proba = preds['y_pred_proba']
            
            self.metrics[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"[OK] Calculated metrics for {name.upper()}")
    
    def print_metrics_table(self):
        """In bảng so sánh metrics"""
        print("\n" + "="*80)
        print("BANG SO SANH HIEU SUAT CAC MO HINH")
        print("="*80)
        
        # Tạo DataFrame
        metrics_data = []
        for name, metrics in self.metrics.items():
            metrics_data.append({
                'Model': name.upper(),
                'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                'Precision': f"{metrics['precision']*100:.2f}%",
                'Recall': f"{metrics['recall']*100:.2f}%",
                'F1-Score': f"{metrics['f1_score']*100:.2f}%",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        df = pd.DataFrame(metrics_data)
        print("\n" + df.to_string(index=False))
        
        # Tìm model tốt nhất
        best_model = max(self.metrics.items(), key=lambda x: x[1]['f1_score'])
        print(f"\nMO HINH TOT NHAT: {best_model[0].upper()} (F1-Score: {best_model[1]['f1_score']*100:.2f}%)")
        print("="*80)
    
    def plot_confusion_matrices(self, output_dir='results'):
        """Vẽ confusion matrices cho tất cả models"""
        os.makedirs(output_dir, exist_ok=True)
        
        n_models = len(self.metrics)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, metrics) in enumerate(self.metrics.items()):
            cm = metrics['confusion_matrix']
            
            # Vẽ heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{name.upper()}\nConfusion Matrix', 
                               fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=12)
            axes[idx].set_ylabel('Actual', fontsize=12)
            axes[idx].set_xticklabels(['Normal', 'Fraud'])
            axes[idx].set_yticklabels(['Normal', 'Fraud'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"[OK] Da luu confusion matrices tai {output_dir}/confusion_matrices.png")
        plt.close()
    
    def plot_roc_curves(self, y_test, output_dir='results'):
        """Vẽ ROC curves cho tất cả models"""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (name, preds) in enumerate(self.predictions.items()):
            y_pred_proba = preds['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = self.metrics[name]['roc_auc']
            
            plt.plot(fpr, tpr, color=colors[idx], lw=2,
                    label=f'{name.upper()} (AUC = {auc:.4f})')
        
        # Đường diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"[OK] Da luu ROC curves tai {output_dir}/roc_curves.png")
        plt.close()
    
    def plot_metrics_comparison(self, output_dir='results'):
        """Vẽ biểu đồ so sánh các metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Chuẩn bị dữ liệu
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        models_names = list(self.metrics.keys())
        
        data = []
        for metric in metrics_names:
            for model in models_names:
                data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Model': model.upper(),
                    'Score': self.metrics[model][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Vẽ biểu đồ
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(metrics_names))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, model in enumerate(models_names):
            model_data = df[df['Model'] == model.upper()]
            scores = model_data['Score'].values
            ax.bar(x + idx*width, scores, width, label=model.upper(), 
                  color=colors[idx], alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"[OK] Da luu metrics comparison tai {output_dir}/metrics_comparison.png")
        plt.close()
    
    def save_evaluation_report(self, output_dir='results'):
        """Lưu báo cáo đánh giá chi tiết"""
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BAO CAO DANH GIA MO HINH PHAT HIEN GIAN LAN BAO HIEM\n")
            f.write("="*80 + "\n\n")
            
            for name, metrics in self.metrics.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"MO HINH: {name.upper()}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"Accuracy:  {metrics['accuracy']*100:.2f}%\n")
                f.write(f"Precision: {metrics['precision']*100:.2f}%\n")
                f.write(f"Recall:    {metrics['recall']*100:.2f}%\n")
                f.write(f"F1-Score:  {metrics['f1_score']*100:.2f}%\n")
                f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n\n")
                
                f.write("Confusion Matrix:\n")
                f.write(str(metrics['confusion_matrix']) + "\n")
        
        print(f"[OK] Da luu bao cao danh gia tai {report_path}")


def main():
    """Hàm chính để đánh giá models"""
    print("\n" + "="*80)
    print("BAT DAU DANH GIA MODELS")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    print(f"[OK] Loaded test data: {X_test.shape}")
    
    # Khởi tạo evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    evaluator.load_models()
    
    # Dự đoán
    evaluator.predict_all(X_test)
    
    # Tính metrics
    evaluator.calculate_metrics(y_test)
    
    # In bảng metrics
    evaluator.print_metrics_table()
    
    # Vẽ các biểu đồ
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves(y_test)
    evaluator.plot_metrics_comparison()
    
    # Lưu báo cáo
    evaluator.save_evaluation_report()
    
    print("\n" + "="*80)
    print("HOAN THANH DANH GIA!")
    print("="*80)


if __name__ == "__main__":
    main()
