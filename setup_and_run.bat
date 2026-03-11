@echo off
REM Script batch để chạy toàn bộ pipeline trên Windows

echo ============================================================
echo  INSURANCE FRAUD DETECTION - FULL PIPELINE
echo ============================================================
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python chua duoc cai dat!
    echo Vui long cai dat Python tu https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Kiem tra virtual environment...
if not exist "venv\" (
    echo [INFO] Tao virtual environment...
    python -m venv venv
    echo [OK] Da tao virtual environment
) else (
    echo [OK] Virtual environment da ton tai
)

echo.
echo [2/5] Kich hoat virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/5] Cai dat dependencies...
pip install -r requirements.txt --quiet

echo.
echo [4/5] Chay pipeline...
python run_pipeline.py

echo.
echo [5/5] Hoan thanh!
echo.
echo ============================================================
echo  NEXT STEPS:
echo ============================================================
echo  1. Xem ket qua tai thu muc 'results/'
echo  2. Chay web app:
echo     cd web
echo     python app.py
echo ============================================================
echo.

pause
