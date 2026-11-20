#!/usr/bin/env bash
# ==============================================================================
# PiVision RPi5 — Skrip Instalasi & Setup Otomatis untuk Raspberry Pi 5
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo "  PiVision RPi5 — Setup Sistem Machine Vision (Linux ARM64)"
echo "=========================================================="

# 1. Update paket sistem & install dependensi runtime Qt6, OpenCV, dan lgpio
echo "[1/4] Memasang dependensi sistem (apt)..."
sudo apt update
sudo apt install -y \
  python3-venv python3-pip python3-lgpio liblgpio-dev \
  libgl1 libglib2.0-0 libxcb-cursor0 libegl1 libxkbcommon-x11-0 \
  v4l-utils

# 2. Pastikan user memiliki hak akses ke device video & gpio
echo "[2/4] Mengonfigurasi hak akses grup video & gpio..."
sudo usermod -aG video,gpio "$USER" || true

# 3. Buat direktori runtime jika belum ada
mkdir -p templates data ng_captures exports

# 4. Buat virtual environment dengan akses ke system-site-packages (wajib untuk lgpio)
echo "[3/4] Menyiapkan Virtual Environment Python..."
if [ ! -d "venv" ]; then
  python3 -m venv --system-site-packages venv
fi

# Aktifkan virtual environment
source venv/bin/activate

# 5. Instal paket Python untuk Raspberry Pi 5
echo "[4/4] Menginstal dependensi Python dari requirements-rpi.txt..."
pip install --upgrade pip
pip install -r requirements-rpi.txt

echo "=========================================================="
echo "  Instalasi selesai!"
echo "  Menjalankan aplikasi: python3 main.py"
echo "=========================================================="
python3 main.py
