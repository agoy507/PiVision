"""
template_manager.py
-------------------
Manajemen Recipe / Template Referensi (Teach-by-Example) serta konfigurasi aplikasi
yang disimpan pada config.json dan folder templates/<nama_template>/ref.png.
100% kompatibel lintas OS (Windows 10/11 & Raspberry Pi OS 64-bit) menggunakan
pathlib.Path, encoding="utf-8", dan baca/tulis gambar tahan karakter Unicode.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from matcher import METHOD_TEMPLATE_FIXED, TemplateEntry, normalize_method_name


@dataclass
class GPIOConfig:
    trigger_pin: int = 23
    ok_led_pin: int = 17
    ng_led_pin: int = 27
    buzzer_pin: int = 22
    active_high: bool = True
    bounce_time_sec: float = 0.15


@dataclass
class AppConfig:
    camera_backend: str = "usb"  # "usb" atau "picamera2"
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    operation_mode: str = "trigger"  # "trigger" atau "continuous"
    matching_method: str = METHOD_TEMPLATE_FIXED
    match_all_templates: bool = False
    multi_angle_coarse_step: int = 10
    multi_angle_fine_step: int = 2
    trigger_delay_ms: int = 200
    output_hold_sec: float = 2.0
    continuous_cooldown_sec: float = 3.0
    ng_retention_days: int = 30
    mute_buzzer: bool = False
    gpio: GPIOConfig = field(default_factory=GPIOConfig)


@dataclass
class TemplateMetadata:
    name: str
    folder: str
    image_path: str  # Format POSIX relatif agar portabel Windows <-> Linux
    threshold: float
    width: int
    height: int
    created_at: str


class TemplateManager:
    """Mengelola file gambar template referensi dan konfigurasi lokal config.json."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir: Path = (
            Path(base_dir).resolve()
            if base_dir is not None
            else Path(__file__).resolve().parent
        )
        self.templates_dir: Path = self.base_dir / "templates"
        self.config_path: Path = self.base_dir / "config.json"

        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.config: AppConfig = AppConfig()
        self.templates: Dict[str, TemplateMetadata] = {}
        self._image_cache: Dict[str, np.ndarray] = {}

        self.load()

    @staticmethod
    def sanitize_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_\\-]+", "_", name.strip())
        cleaned = cleaned.strip("_")
        return cleaned or f"template_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def resolve_relative_path(self, rel_path: str) -> Path:
        normalized = rel_path.replace("\\\\", "/")
        parts = PurePosixPath(normalized).parts
        return self.base_dir.joinpath(*parts)

    @staticmethod
    def _write_image_unicode_safe(file_path: Path, image_bgr: np.ndarray) -> bool:
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            return False
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(encoded.tobytes())
        return True

    @staticmethod
    def _read_image_unicode_safe(file_path: Path) -> Optional[np.ndarray]:
        if not file_path.exists():
            return None
        try:
            raw = np.frombuffer(file_path.read_bytes(), dtype=np.uint8)
            return cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def load(self) -> None:
        if not self.config_path.exists():
            self.save_config()
            return

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            self.save_config()
            return

        cfg_raw = raw.get("app_config", {})
        gpio_raw = cfg_raw.get("gpio", {})

        gpio_cfg = GPIOConfig(
            trigger_pin=int(gpio_raw.get("trigger_pin", 23)),
            ok_led_pin=int(gpio_raw.get("ok_led_pin", 17)),
            ng_led_pin=int(gpio_raw.get("ng_led_pin", 27)),
            buzzer_pin=int(gpio_raw.get("buzzer_pin", 22)),
            active_high=bool(gpio_raw.get("active_high", True)),
            bounce_time_sec=float(gpio_raw.get("bounce_time_sec", 0.15)),
        )

        raw_method = str(cfg_raw.get("matching_method", METHOD_TEMPLATE_FIXED))
        self.config = AppConfig(
            camera_backend=str(cfg_raw.get("camera_backend", "usb")),
            camera_index=int(cfg_raw.get("camera_index", 0)),
            camera_width=int(cfg_raw.get("camera_width", 1280)),
            camera_height=int(cfg_raw.get("camera_height", 720)),
            camera_fps=int(cfg_raw.get("camera_fps", 30)),
            operation_mode=str(cfg_raw.get("operation_mode", "trigger")),
            matching_method=normalize_method_name(raw_method),
            match_all_templates=bool(cfg_raw.get("match_all_templates", False)),
            multi_angle_coarse_step=int(cfg_raw.get("multi_angle_coarse_step", 10)),
            multi_angle_fine_step=int(cfg_raw.get("multi_angle_fine_step", 2)),
            trigger_delay_ms=int(cfg_raw.get("trigger_delay_ms", 200)),
            output_hold_sec=float(cfg_raw.get("output_hold_sec", 2.0)),
            continuous_cooldown_sec=float(
                cfg_raw.get("continuous_cooldown_sec", 3.0)
            ),
            ng_retention_days=int(cfg_raw.get("ng_retention_days", 30)),
            mute_buzzer=bool(cfg_raw.get("mute_buzzer", False)),
            gpio=gpio_cfg,
        )

        self.templates.clear()
        self._image_cache.clear()
        for item in raw.get("templates", []):
            name = str(item.get("name", "")).strip()
            rel_path = str(item.get("image_path", "")).strip()
            if not name or not rel_path:
                continue
            abs_path = self.resolve_relative_path(rel_path)
            if abs_path.exists():
                meta = TemplateMetadata(
                    name=name,
                    folder=str(item.get("folder", self.sanitize_name(name))),
                    image_path=abs_path.relative_to(self.base_dir).as_posix(),
                    threshold=float(item.get("threshold", 0.80)),
                    width=int(item.get("width", 0)),
                    height=int(item.get("height", 0)),
                    created_at=str(item.get("created_at", "")),
                )
                self.templates[name] = meta

    def save_config(self) -> None:
        payload = {
            "app_config": asdict(self.config),
            "templates": [asdict(t) for t in self.templates.values()],
        }
        tmp_path = self.config_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.config_path)

    def save_template(
        self,
        name: str,
        frame_bgr: np.ndarray,
        roi: Tuple[int, int, int, int],
        threshold: float,
    ) -> TemplateMetadata:
        display_name = name.strip()
        if not display_name:
            raise ValueError("Nama template tidak boleh kosong.")

        x, y, w, h = roi
        fh, fw = frame_bgr.shape[:2]
        x1 = max(0, min(fw - 1, int(x)))
        y1 = max(0, min(fh - 1, int(y)))
        x2 = max(x1 + 1, min(fw, int(x + w)))
        y2 = max(y1 + 1, min(fh, int(y + h)))

        if (x2 - x1) < 16 or (y2 - y1) < 16:
            raise ValueError("Ukuran ROI terlalu kecil (minimal 16x16 piksel).")

        crop = frame_bgr[y1:y2, x1:x2].copy()
        safe_folder = self.sanitize_name(display_name)
        tpl_dir = self.templates_dir / safe_folder
        tpl_dir.mkdir(parents=True, exist_ok=True)
        img_file = tpl_dir / "ref.png"

        if not self._write_image_unicode_safe(img_file, crop):
            raise IOError(f"Gagal menyimpan file template ke {img_file}")

        rel_path = img_file.relative_to(self.base_dir).as_posix()
        meta = TemplateMetadata(
            name=display_name,
            folder=safe_folder,
            image_path=rel_path,
            threshold=round(float(threshold), 3),
            width=int(crop.shape[1]),
            height=int(crop.shape[0]),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.templates[display_name] = meta
        self._image_cache[display_name] = crop
        self.save_config()
        return meta

    def delete_template(self, name: str) -> bool:
        meta = self.templates.pop(name, None)
        self._image_cache.pop(name, None)
        if meta is None:
            return False

        tpl_dir = self.templates_dir / meta.folder
        if tpl_dir.exists() and tpl_dir.is_dir():
            shutil.rmtree(tpl_dir, ignore_errors=True)

        self.save_config()
        return True

    def list_templates(self) -> List[TemplateMetadata]:
        return list(self.templates.values())

    def get_template_image(self, name: str) -> Optional[np.ndarray]:
        if name in self._image_cache:
            return self._image_cache[name]

        meta = self.templates.get(name)
        if meta is None:
            return None

        img_path = self.resolve_relative_path(meta.image_path)
        img = self._read_image_unicode_safe(img_path)
        if img is not None:
            self._image_cache[name] = img
        return img

    def get_template_entries(
        self, selected_name: Optional[str] = None, match_all: bool = False
    ) -> List[TemplateEntry]:
        entries: List[TemplateEntry] = []
        target_names = (
            list(self.templates.keys())
            if match_all
            else ([selected_name] if selected_name in self.templates else [])
        )
        for name in target_names:
            meta = self.templates[name]
            img = self.get_template_image(name)
            if img is not None:
                entries.append(
                    TemplateEntry(
                        name=meta.name,
                        threshold=meta.threshold,
                        image_bgr=img,
                    )
                )
        return entries

    def update_template_threshold(self, name: str, threshold: float) -> None:
        if name in self.templates:
            self.templates[name].threshold = round(float(threshold), 3)
            self.save_config()
