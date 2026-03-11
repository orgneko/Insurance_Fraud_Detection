#!/bin/bash
# Script shell để chạy toàn bộ pipeline trên Linux/Mac

echo "============================================================"
echo "  INSURANCE FRAUD DETECTION - FULL PIPELINE"
echo "============================================================"
echo ""

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python chưa được cài đặt!"
    echo "Vui lòng cài đặt Python từ https://www.python.org/"
    exit 1
fi

echo "[1/5] Kiểm tra virtual environment..."
if [ ! -d "venv" ]; then
    echo "[INFO] Tạo virtual environment..."
    python3 -m venv venv
    echo "[OK] Đã tạo virtual environment"
else
    echo "[OK] Virtual environment đã tồn tại"
fi

echo ""
echo "[2/5] Kích hoạt virtual environment..."
source venv/bin/activate

echo ""
echo "[3/5] Cài đặt dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "[4/5] Chạy pipeline..."
python run_pipeline.py

echo ""
echo "[5/5] Hoàn thành!"
echo ""
echo "============================================================"
echo "  NEXT STEPS:"
echo "============================================================"
echo "  1. Xem kết quả tại thư mục 'results/'"
echo "  2. Chạy web app:"
echo "     cd web"
echo "     python app.py"
echo "  3. Xem hướng dẫn: QUICK_START.md"
echo "============================================================"
echo ""
