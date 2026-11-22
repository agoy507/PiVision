# PiVision RPi5 — Industrial Offline Machine Vision System

Sistem inspeksi visual industri (**Machine Vision**) berbasis Python murni yang dirancang untuk beroperasi secara **100% offline** pada **Raspberry Pi 5** (lingkungan produksi) dan **Windows 10/11** (lingkungan pengujian & simulasi tanpa modifikasi kode).

Dilengkapi dengan antarmuka operator HMI (*Human Machine Interface*) industrial berbasis **PySide6 (Qt6)**, pemrosesan citra berkecepatan tinggi dengan **OpenCV**, kontrol I/O perangkat keras melalui **lgpio / gpiozero**, serta pencatatan data inspeksi otomatis dengan **SQLite**.

---

## Fitur Utama

- **Antarmuka PySide6 Industrial (Dark Mode)**
  - Tampilan video kamera *real-time* dengan *framerate counter* (FPS).
  - Indikator status inspeksi visual besar (**PASS / OK** hijau, **FAIL / NG** merah).
  - Penghitung produksi (*Total*, *OK*, *NG*, *Yield %*) dengan tombol reset.
  - Gambar Region of Interest (ROI) secara interaktif langsung pada layar kamera.
  - Mode inspeksi **Kontinu** (otomatis per frame) atau **Trigger GPIO / Manual**.
  - Badge deteksi sistem: otomatis menampilkan `[SIMULASI GPIO - WINDOWS]` atau `[HARDWARE GPIO - RPI5]`.

- **3 Algoritma Vision Matching (0° – 360°)**
  1. **Template Matching (0° Tetap)**: Cepat dan efisien untuk objek dengan orientasi tetap menggunakan `cv2.TM_CCOEFF_NORMED`.
  2. **Multi-Angle Template Matching (0°–360°)**: Pencarian rotasi dua tahap (*coarse-to-fine*) dengan *circular mask* dan *in-memory cache* untuk orientasi acak pada konveyor.
  3. **Feature Matching (AKAZE + Affine 2D)**: Tahan terhadap rotasi ekstrem, variasi skala ringan, dan perubahan pencahayaan menggunakan ekstraksi *feature keypoint* AKAZE dan estimasi matriks affine dengan RANSAC.

- **Dukungan Kamera Fleksibel**
  - **Windows**: Menggunakan backend `cv2.CAP_DSHOW` untuk inisialisasi webcam USB yang cepat dan stabil.
  - **Linux / Raspberry Pi 5**: Menggunakan `cv2.CAP_V4L2` atau **PiCamera2** (CSI camera port).

- **Kontrol I/O Lintas Platform**
  - **Raspberry Pi 5**: Menggunakan `gpiozero` dengan backend `lgpio` resmi untuk chip I/O RP1.
  - **Windows / PC**: Otomatis dialihkan ke `MockGPIOController` yang mensimulasikan input tombol trigger dan output lampu OK, NG, serta Buzzer alarm di GUI.

- **Data Logging & Bukti Visual Cacat (NG)**
  - Pencatatan riwayat setiap siklus inspeksi ke database lokal SQLite (`data/inspection_logs.db`).
  - Penyimpanan otomatis foto bukti cacat (**NG**) dengan visualisasi *bounding box* / poligon deteksi ke folder `ng_captures/YYYY-MM-DD/`.
  - Ekspor laporan riwayat inspeksi ke file format **CSV (UTF-8)** di folder `exports/`.

---

## Struktur Direktori

```text
PiVision/
├── main.py                    # Aplikasi GUI PySide6 utama
├── camera_worker.py           # Abstraksi kamera (USB CAP_DSHOW/V4L2 & PiCamera2) + QThread
├── matcher.py                 # Engine pencocokan (0° Fixed, 0°–360° Multi-Angle, AKAZE)
├── template_manager.py        # Pengelola resep & template referensi (pathlib + JSON)
├── gpio_controller.py         # Kontrol GPIO (RPi 5 lgpio & MockGPIOController Windows)
├── data_logger.py             # SQLite worker, auto-save foto NG, ekspor CSV
├── requirements.txt           # Dependensi Python umum
├── requirements-windows.txt   # Dependensi Windows (tanpa gpiozero/lgpio)
├── requirements-rpi.txt       # Dependensi Raspberry Pi 5 (dengan gpiozero & lgpio)
├── run.bat                    # Skrip peluncur cepat Windows (CMD)
├── run.ps1                    # Skrip peluncur cepat Windows (PowerShell)
├── setup_rpi5.sh              # Skrip instalasi & peluncur otomatis di Raspberry Pi 5
├── templates/                 # Direktori penyimpanan template referensi inspeksi
├── data/                      # Direktori penyimpanan database SQLite (inspection_logs.db)
├── ng_captures/               # Direktori penyimpanan foto bukti cacat produk (NG)
└── exports/                   # Direktori penyimpanan file ekspor laporan CSV
```

---

## Panduan Instalasi & Menjalankan

### A. Windows 10/11 (Pengujian & Simulasi)

#### Menggunakan Skrip Cepat
Cukup klik ganda `run.bat` atau jalankan via PowerShell:
```powershell
.\run.ps1
```

#### Atau Langkah Manual via Terminal
```powershell
# 1. Buat dan aktifkan Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Perbarui pip dan pasang dependensi
python -m pip install --upgrade pip
pip install -r requirements-windows.txt

# 3. Jalankan aplikasi
python main.py
```

---

### B. Raspberry Pi 5 (Produksi Linux ARM64)

#### Cara Otomatis (1 Perintah)
```bash
chmod +x setup_rpi5.sh
./setup_rpi5.sh
```

#### Langkah Manual
```bash
# 1. Pasang dependensi sistem & driver lgpio
sudo apt update
sudo apt install -y \
  python3-venv python3-pip python3-lgpio liblgpio-dev \
  libgl1 libglib2.0-0 libxcb-cursor0 libegl1 libxkbcommon-x11-0 \
  v4l-utils

# 2. Beri hak akses hardware ke user
sudo usermod -aG video,gpio $USER

# 3. Buat virtual environment dengan akses system-site-packages (wajib untuk lgpio)
python3 -m venv --system-site-packages venv
source venv/bin/activate

# 4. Pasang paket Python produksi
pip install --upgrade pip
pip install -r requirements-rpi.txt

# 5. Jalankan aplikasi
python3 main.py
```

---

## Diagram Pengkabelan GPIO (Raspberry Pi 5)

Berikut adalah pemetaan pin header GPIO 40-pin Raspberry Pi 5 untuk integrasi sensor konveyor dan aktuator:

| Fungsi | Pin Fisik | BCM GPIO | Arah | Keterangan Koneksi |
| :--- | :---: | :---: | :---: | :--- |
| **Trigger Sensor / Button** | **Pin 16** | **BCM 23** | `INPUT` (*pull_up=True*) | Kaki 1 ke Pin 16 (BCM 23), Kaki 2 ke GND. Aktif saat terhubung GND (*falling edge*). |
| **Lampu Indikator Hijau (OK)** | **Pin 11** | **BCM 17** | `OUTPUT` | Ke anoda LED (+ resistor 330Ω) atau pin `IN1` modul Relay. |
| **Lampu Indikator Merah (NG)** | **Pin 13** | **BCM 27** | `OUTPUT` | Ke anoda LED (+ resistor 330Ω) atau pin `IN2` modul Relay. |
| **Buzzer Alarm (NG)** | **Pin 15** | **BCM 22** | `OUTPUT` | Ke Active Buzzer 3.3V atau pin `IN3` modul Relay. |
| **Ground Referensi** | **Pin 6 / 9 / 14** | `GND` | Ground | Ground bersama untuk sensor, LED, dan modul relay. |
| **Tegangan Suplai 5V** | **Pin 2 / 4** | `5V` | Power | VCC untuk modul relay optocoupler. |
| **Tegangan Suplai 3.3V** | **Pin 1 / 17** | `3.3V` | Power | Suplai referensi logika sensor (maks 50mA). |

> **PERINGATAN KESELAMATAN:** Pin GPIO Raspberry Pi 5 bekerja pada level tegangan **3.3V**. Jangan menghubungkan sinyal 5V atau 24V secara langsung ke pin GPIO Raspberry Pi tanpa modul pengaman seperti *optocoupler* atau *voltage divider*.

---

## Cara Penggunaan

1. **Pilih Sumber Kamera:** Pilih sumber kamera (`USB #0`, `USB #1`, `USB #2`, atau `PiCamera2 (RPi5)`) melalui dropdown pada panel info bawah layar video, lalu klik **Buka / Reconnect**.
2. **Buat Template Referensi (Teach-by-Example):**
   - Klik **MODE TEACH** di bagian kanan atas.
   - Letakkan sampel produk standar (OK) di depan kamera.
   - Klik **FREEZE / CAPTURE FRAME** untuk membekukan gambar pratinjau.
   - Tarik kotak ROI (*click-and-drag*) langsung di layar video untuk memilih area referensi.
   - Isi **Nama Template** dan atur **Threshold Kemiripan** (misal `0.80`).
   - Klik **SIMPAN TEMPLATE REFERENSI**.
3. **Mulai Inspeksi:**
   - Kembali ke **MODE RUN**, pilih **Metode** (*0° Tetap*, *Multi-Angle 0°–360°*, atau *AKAZE + Affine 2D*), lalu pilih template pada daftar (atau centang **Auto-Match**).
   - Threshold template terpilih juga dapat disesuaikan langsung dari panel RUN lewat input **Threshold Terpilih** + tombol **Update Threshold**.
   - Klik **START INSPEKSI**.
   - Pada *Trigger Mode*, tekan tombol **MANUAL TRIGGER [SPASI]** atau picu sensor fisik pada GPIO Pin 16 (BCM 23). Pada *Continuous Mode*, inspeksi berjalan otomatis pada setiap frame.
   - Status OK/NG beserta skor kecocokan dan sudut rotasi ditampilkan secara langsung di layar dan dicatat ke database SQLite.
