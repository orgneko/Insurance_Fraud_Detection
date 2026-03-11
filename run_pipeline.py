"""
Script chạy toàn bộ pipeline: Data Preprocessing → Training → Evaluation
"""

import os
import sys
import time
from datetime import datetime

def print_header(text):
    """In header đẹp"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_step(step_name, script_path):
    """Chạy một bước trong pipeline"""
    print_header(f"BUOC: {step_name}")
    print(f"Bat dau: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    # Chạy script
    exit_code = os.system(f"{sys.executable} {script_path}")
    
    elapsed_time = time.time() - start_time
    
    if exit_code == 0:
        print(f"\n[OK] {step_name} hoan thanh trong {elapsed_time:.2f}s")
        return True
    else:
        print(f"\n[ERROR] {step_name} that bai!")
        return False

def main():
    """Hàm chính"""
    print_header("INSURANCE FRAUD DETECTION - FULL PIPELINE")
    print("Bai tap lon mon Tri tue nhan tao")
    print("De tai: Phat hien gian lan bao hiem voi XGBoost, Random Forest, ANN")
    print("\n" + "="*80)
    
    # Kiểm tra thư mục
    if not os.path.exists('src'):
        print("[ERROR] Loi: Khong tim thay thu muc 'src'")
        print("   Vui long chay script nay tu thu muc goc cua du an")
        sys.exit(1)
    
    # Pipeline steps
    steps = [
        ("1. Chuan bi Du lieu", "src/data_preprocessing.py"),
        ("2. Training Models", "src/train_models.py"),
        ("3. Danh gia Models", "src/evaluate_models.py")
    ]
    
    total_start = time.time()
    
    # Chạy từng bước
    for step_name, script_path in steps:
        success = run_step(step_name, script_path)
        if not success:
            print("\n[ERROR] Pipeline that bai!")
            sys.exit(1)
        time.sleep(1)  # Pause giữa các bước
    
    total_time = time.time() - total_start
    
    # Kết thúc
    print_header("HOAN THANH TOAN BO PIPELINE")
    print(f"Tong thoi gian: {total_time:.2f}s ({total_time/60:.2f} phut)")
    print("\nKet qua da duoc luu tai:")
    print("   - data/          : Du lieu da xu ly")
    print("   - models/        : Models da train")
    print("   - results/       : Visualizations va bao cao")
    print("\nDe chay web application:")
    print("   cd web")
    print("   python app.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
