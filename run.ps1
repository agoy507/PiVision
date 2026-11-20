# ============================================================
#   PiVision RPi5 - Industrial Machine Vision (Windows Debug)
# ============================================================
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  PiVision RPi5 - Industrial Machine Vision (Windows Debug)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "[INFO] Virtual environment belum ditemukan. Membuat venv..." -ForegroundColor Yellow
    try {
        & py -3 -m venv venv
    } catch {
        & python -m venv venv
    }
    Write-Host "[INFO] Menginstal dependensi dari requirements-windows.txt..." -ForegroundColor Yellow
    & .\venv\Scripts\python.exe -m pip install --upgrade pip
    & .\venv\Scripts\python.exe -m pip install -r requirements-windows.txt
}

Write-Host "[INFO] Menjalankan PiVision (main.py)..." -ForegroundColor Green
& .\venv\Scripts\python.exe main.py
