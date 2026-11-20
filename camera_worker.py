"""
camera_worker.py
----------------
Abstraksi sumber kamera (USB Webcam & PiCamera2) dan QThread pembacaan frame
secara non-blocking lintas OS:
- Windows 10/11 (64-bit): menggunakan cv2.CAP_DSHOW agar webcam terbuka cepat & stabil.
- Linux / Raspberry Pi 5 (64-bit): menggunakan cv2.CAP_V4L2 (fallback default) atau PiCamera2.
"""

from __future__ import annotations

import platform
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class CameraSource(ABC):
    """Antarmuka dasar (wrapper) untuk berbagai backend kamera."""

    @abstractmethod
    def open(self) -> Tuple[bool, str]:
        """Membuka koneksi kamera. Mengembalikan (sukses, pesan_error_atau_info)."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Membaca 1 frame BGR (OpenCV format)."""

    @abstractmethod
    def release(self) -> None:
        """Menutup koneksi kamera secara aman."""

    @abstractmethod
    def is_opened(self) -> bool:
        """Status apakah kamera sedang aktif."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nama deskriptif sumber kamera."""


class USBCameraSource(CameraSource):
    """
    Implementasi kamera USB Webcam lintas OS:
    - Windows: cv2.CAP_DSHOW (DirectShow) untuk inisialisasi cepat & stabil.
    - Linux / RPi 5: cv2.CAP_V4L2 dengan fallback backend default OpenCV.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._backend_label: str = "DEFAULT"

    def _try_open_backend(
        self, backend: Optional[int], label: str
    ) -> Optional[cv2.VideoCapture]:
        cap = (
            cv2.VideoCapture(self.device_index)
            if backend is None
            else cv2.VideoCapture(self.device_index, backend)
        )
        if not cap.isOpened():
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Uji baca 1 frame warm-up untuk mendeteksi webcam yang terkunci aplikasi lain
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                self._backend_label = label
                return cap
            time.sleep(0.05)

        cap.release()
        return None

    def open(self) -> Tuple[bool, str]:
        self.release()
        os_name = platform.system()

        if os_name == "Windows":
            cap = self._try_open_backend(cv2.CAP_DSHOW, "CAP_DSHOW")
            if cap is None:
                cap = self._try_open_backend(None, "DEFAULT")
            if cap is None:
                return (
                    False,
                    f"Webcam index #{self.device_index} tidak dapat dibuka di Windows (CAP_DSHOW). "
                    "Pastikan kamera terpasang, tidak sedang dipakai aplikasi lain "
                    "(Zoom/Teams/OBS/Camera), dan izin Kamera di Windows Settings aktif.",
                )
        elif os_name == "Linux":
            cap = self._try_open_backend(cv2.CAP_V4L2, "CAP_V4L2")
            if cap is None:
                cap = self._try_open_backend(None, "DEFAULT")
            if cap is None:
                return (
                    False,
                    f"Webcam index #{self.device_index} tidak dapat dibuka di Linux/RPi 5. "
                    "Periksa koneksi USB, pastikan tidak dipakai proses lain, dan cek izin /dev/video* "
                    "(sudo usermod -aG video $USER).",
                )
        else:
            cap = self._try_open_backend(None, "DEFAULT")
            if cap is None:
                return (
                    False,
                    f"Webcam index #{self.device_index} tidak terdeteksi atau sedang dipakai aplikasi lain.",
                )

        self._cap = cap
        return True, self.name

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def name(self) -> str:
        return (
            f"USB Camera #{self.device_index} "
            f"({self.width}x{self.height} · {self._backend_label})"
        )


class PiCamera2Source(CameraSource):
    """
    Wrapper siap pakai untuk modul kamera CSI Raspberry Pi (IMX219/IMX477/IMX708)
    menggunakan library resmi picamera2.
    Ganti camera_backend ke 'picamera2' di config.json untuk mengaktifkan di RPi 5.
    """

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self._picam2 = None
        self._opened = False

    def open(self) -> Tuple[bool, str]:
        self.release()
        if platform.system() != "Linux":
            return (
                False,
                "PiCamera2 hanya didukung di Linux/Raspberry Pi OS. "
                "Gunakan camera_backend='usb' saat debug di Windows.",
            )
        try:
            from picamera2 import Picamera2  # type: ignore

            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            self._picam2 = picam2
            self._opened = True
            return True, self.name
        except Exception as exc:
            self._picam2 = None
            self._opened = False
            return False, f"Gagal membuka PiCamera2 CSI: {exc}"

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened or self._picam2 is None:
            return False, None
        try:
            frame_rgb = self._picam2.capture_array()
            if frame_rgb is None:
                return False, None
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        except Exception:
            return False, None

    def release(self) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            finally:
                self._picam2 = None
        self._opened = False

    def is_opened(self) -> bool:
        return self._opened

    @property
    def name(self) -> str:
        return f"PiCamera2 CSI ({self.width}x{self.height})"


def create_camera_source(
    backend: str = "usb",
    device_index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> CameraSource:
    """Factory pembuat sumber kamera; otomatis fallback ke USB jika dijalankan di Windows."""
    if backend.strip().lower() == "picamera2" and platform.system() == "Linux":
        return PiCamera2Source(width=width, height=height)
    return USBCameraSource(
        device_index=device_index, width=width, height=height, fps=fps
    )


class CameraWorker(QThread):
    """
    Worker QThread untuk membaca frame kamera secara kontinu tanpa memblokir GUI.
    Menyediakan get_latest_frame() yang thread-safe untuk Trigger Mode.
    """

    frame_ready = Signal(np.ndarray, float)
    camera_status = Signal(bool, str)
    camera_error = Signal(str)

    def __init__(self, source: CameraSource, parent=None) -> None:
        super().__init__(parent)
        self._source = source
        self._running = False
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._fps_timestamps: deque[float] = deque(maxlen=30)

    def run(self) -> None:
        self._running = True
        opened, info_or_err = self._source.open()
        if not opened:
            self.camera_status.emit(False, "Kamera tidak tersedia")
            self.camera_error.emit(info_or_err)
            self._running = False
            return

        self.camera_status.emit(True, info_or_err)
        consecutive_failures = 0

        try:
            while self._running:
                ok, frame = self._source.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        self.camera_status.emit(
                            False, "Koneksi kamera terputus — mencoba ulang..."
                        )
                        self.camera_error.emit(
                            "Stream webcam terputus atau diambil alih oleh aplikasi lain. "
                            "Mencoba menyambungkan ulang..."
                        )
                        self._source.release()
                        time.sleep(1.0)
                        if not self._running:
                            break
                        reopened, reopen_msg = self._source.open()
                        if reopened:
                            consecutive_failures = 0
                            self.camera_status.emit(True, reopen_msg)
                        else:
                            self.camera_error.emit(reopen_msg)
                            time.sleep(1.5)
                    else:
                        time.sleep(0.015)
                    continue

                consecutive_failures = 0
                now = time.perf_counter()
                self._fps_timestamps.append(now)
                fps = 0.0
                if len(self._fps_timestamps) >= 2:
                    elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
                    if elapsed > 0:
                        fps = (len(self._fps_timestamps) - 1) / elapsed

                with self._frame_lock:
                    self._latest_frame = frame

                self.frame_ready.emit(frame.copy(), fps)
        finally:
            self._source.release()
            self.camera_status.emit(False, "Kamera berhenti")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Mengambil salinan frame terbaru secara aman untuk inspeksi Trigger."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def stop(self) -> None:
        """Menghentikan loop pembacaan kamera secara bersih."""
        self._running = False
        self.wait(3000)
