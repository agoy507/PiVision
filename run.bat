@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   PiVision RPi5 - Industrial Machine Vision (Windows Debug)
echo ============================================================

if not exist "venv\Scripts\python.exe" (
    echo [INFO] Virtual environment belum ditemukan. Membuat venv...
    py -3 -m venv venv || python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Gagal membuat virtual environment. Pastikan Python 3 sudah terpasang.
        pause
        exit /b 1
    )
    echo [INFO] Menginstal dependensi dari requirements-windows.txt...
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements-windows.txt
) else (
    call venv\Scripts\activate.bat
)

echo [INFO] Menjalankan PiVision (main.py)...
python main.py
if errorlevel 1 (
    echo [ERROR] Aplikasi berhenti dengan error.
    pause
)
