"""
matcher.py
----------
Mesin pencocokan citra OpenCV planar 2D lintas OS (Raspberry Pi 5 ARM64 & Windows x64)
dengan dukungan rotasi bebas 0°–360° melalui 3 metode:
1. Template Matching (0° Tetap)            -> cv2.TM_CCOEFF_NORMED (tercepat, posisi tetap).
2. Multi-Angle Template Matching (0°–360°) -> Coarse-to-Fine (0.5x -> 1.0x), Circular Masking
                                              (TM_SQDIFF_NORMED), cache rotasi coarse di memori.
3. Feature Matching (AKAZE + Affine 2D)    -> cv2.AKAZE_create() + BFMatcher(NORM_HAMMING) +
                                              Lowe's Ratio (0.75) + cv2.estimateAffinePartial2D (RANSAC).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

METHOD_TEMPLATE_FIXED = "Template Matching (0° Tetap)"
METHOD_MULTI_ANGLE = "Multi-Angle Template Matching (0°–360°)"
METHOD_AKAZE_AFFINE = "Feature Matching (AKAZE + Affine 2D)"
AVAILABLE_METHODS = [
    METHOD_TEMPLATE_FIXED,
    METHOD_MULTI_ANGLE,
    METHOD_AKAZE_AFFINE,
]

# Pemetaan otomatis nama metode lama pada config.json agar kompatibel mundur
LEGACY_METHOD_MAP = {
    "Template Matching (TM_CCOEFF_NORMED)": METHOD_TEMPLATE_FIXED,
    "Feature Matching (ORB + Homography)": METHOD_AKAZE_AFFINE,
}


def normalize_method_name(method: str) -> str:
    if method in AVAILABLE_METHODS:
        return method
    return LEGACY_METHOD_MAP.get(method, METHOD_TEMPLATE_FIXED)


@dataclass
class TemplateEntry:
    """Data template di memori untuk pencocokan cepat."""

    name: str
    threshold: float
    image_bgr: np.ndarray


@dataclass
class MatchResult:
    """Hasil satu siklus pencocokan citra termasuk sudut rotasi (0°–360°) & kotak terputar."""

    matched: bool
    status: str  # "OK" atau "NG"
    template_name: str
    method: str
    score: float
    threshold: float
    bbox: Optional[Tuple[int, int, int, int]]  # Axis-aligned (x, y, w, h)
    polygon: Optional[List[Tuple[int, int]]]  # 4 titik sudut terputar [(x, y), ...]
    elapsed_ms: float
    angle_deg: float = 0.0  # Sudut rotasi terdeteksi dalam rentang 0.0 - 359.9 derajat
    keypoints_matched: int = 0


@dataclass
class _CoarseCacheEntry:
    """Cache template terputar tahap kasar (skala 0.5) & circular mask-nya."""

    mask: np.ndarray
    rotated_templates: List[Tuple[float, np.ndarray]]  # [(angle_deg, rotated_gray), ...]


class TemplateMatcherEngine:
    """
    Mesin pencocokan citra 2D dengan rotasi bebas 0°–360° yang dioptimalkan untuk
    CPU ARM64 Raspberry Pi 5 dan Windows 64-bit.
    """

    def __init__(
        self,
        coarse_scale: float = 0.5,
        coarse_step_deg: int = 10,
        fine_step_deg: int = 2,
    ) -> None:
        self.coarse_scale = float(coarse_scale)
        self.coarse_step_deg = max(2, int(coarse_step_deg))
        self.fine_step_deg = max(1, int(fine_step_deg))

        if hasattr(cv2, "AKAZE_create"):
            self._akaze = cv2.AKAZE_create()
        elif hasattr(cv2, "AKAZE") and hasattr(cv2.AKAZE, "create"):
            self._akaze = cv2.AKAZE.create()
        else:
            self._akaze = cv2.ORB_create()
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Cache per nama template: fitur AKAZE & rotasi kasar Multi-Angle
        self._akaze_cache: Dict[
            Tuple[str, Tuple[int, int]], Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]
        ] = {}
        self._coarse_cache: Dict[
            Tuple[str, Tuple[int, int], float, int], _CoarseCacheEntry
        ] = {}
        self._fine_mask_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._cache_lock = threading.Lock()

    def set_angle_steps(self, coarse_step_deg: int, fine_step_deg: int) -> None:
        new_coarse = max(2, int(coarse_step_deg))
        new_fine = max(1, min(new_coarse, int(fine_step_deg)))
        with self._cache_lock:
            if new_coarse != self.coarse_step_deg:
                self._coarse_cache.clear()
            self.coarse_step_deg = new_coarse
            self.fine_step_deg = new_fine

    def clear_cache(self, template_name: Optional[str] = None) -> None:
        """Menghapus cache seluruh template atau hanya satu template tertentu."""
        with self._cache_lock:
            if template_name is None:
                self._akaze_cache.clear()
                self._coarse_cache.clear()
                return

            self._akaze_cache = {
                k: v for k, v in self._akaze_cache.items() if k[0] != template_name
            }
            self._coarse_cache = {
                k: v for k, v in self._coarse_cache.items() if k[0] != template_name
            }

    @staticmethod
    def _create_circular_mask(height: int, width: int) -> np.ndarray:
        """
        Membuat circular mask (lingkaran penuh di tengah template) agar area sudut
        hitam hasil rotasi tidak merusak skor cv2.matchTemplate.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        cx, cy = width // 2, height // 2
        radius = max(4, min(width, height) // 2 - 1)
        cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
        return mask

    @staticmethod
    def _rotate_gray_keep_size(gray: np.ndarray, angle_deg: float) -> np.ndarray:
        """Memutar citra grayscale di sekitar pusatnya dengan mempertahankan dimensi (w, h)."""
        h, w = gray.shape[:2]
        if abs(angle_deg % 360.0) < 1e-6:
            return gray.copy()
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(
            gray,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    @staticmethod
    def _rotated_box_corners(
        top_left_x: int, top_left_y: int, width: int, height: int, angle_deg: float
    ) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
        """
        Menghitung 4 titik sudut bounding box terputar (mengikuti arah objek)
        serta bounding box axis-aligned pembungkusnya.
        """
        cx = width / 2.0
        cy = height / 2.0
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        corners = np.array(
            [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        transformed = cv2.transform(corners, rot_mat).reshape(-1, 2)
        transformed[:, 0] += float(top_left_x)
        transformed[:, 1] += float(top_left_y)

        poly = [(int(round(pt[0])), int(round(pt[1]))) for pt in transformed]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        return poly, bbox

    def _get_fine_mask(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        with self._cache_lock:
            mask = self._fine_mask_cache.get(key)
            if mask is None:
                mask = self._create_circular_mask(h, w)
                self._fine_mask_cache[key] = mask
            return mask

    def _get_coarse_rotated_cache(
        self, name: str, gray_tpl: np.ndarray
    ) -> _CoarseCacheEntry:
        th, tw = gray_tpl.shape[:2]
        scale = self.coarse_scale
        step = self.coarse_step_deg
        key = (name, (th, tw), scale, step)

        with self._cache_lock:
            cached = self._coarse_cache.get(key)
        if cached is not None:
            return cached

        cw = max(12, int(round(tw * scale)))
        ch = max(12, int(round(th * scale)))
        small_tpl = cv2.resize(gray_tpl, (cw, ch), interpolation=cv2.INTER_AREA)
        coarse_mask = self._create_circular_mask(ch, cw)

        rotated_list: List[Tuple[float, np.ndarray]] = []
        for angle in range(0, 360, step):
            rot = self._rotate_gray_keep_size(small_tpl, float(angle))
            rotated_list.append((float(angle), rot))

        entry = _CoarseCacheEntry(mask=coarse_mask, rotated_templates=rotated_list)
        with self._cache_lock:
            self._coarse_cache[key] = entry
        return entry

    def _get_template_akaze_features(
        self, name: str, gray_template: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        key = (name, gray_template.shape[:2])
        with self._cache_lock:
            cached = self._akaze_cache.get(key)
        if cached is not None:
            return cached

        keypoints, descriptors = self._akaze.detectAndCompute(gray_template, None)
        with self._cache_lock:
            self._akaze_cache[key] = (keypoints, descriptors)
        return keypoints, descriptors

    @staticmethod
    def _match_masked_sqdiff_score(
        gray_region: np.ndarray, rot_tpl: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Menjalankan cv2.matchTemplate dengan circular mask memakai TM_SQDIFF_NORMED
        lalu mengonversi hasilnya ke rentang [0.0, 1.0] (1.0 = paling cocok).
        """
        res = cv2.matchTemplate(
            gray_region, rot_tpl, cv2.TM_SQDIFF_NORMED, mask=mask
        )
        # Bersihkan kemungkinan NaN/Inf pada area polos serba nol
        res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
        score = float(np.clip(1.0 - float(min_val), 0.0, 1.0))
        return score, (int(min_loc[0]), int(min_loc[1]))

    def match_single(
        self, frame_bgr: np.ndarray, template: TemplateEntry, method: str
    ) -> MatchResult:
        """Mencocokkan satu template terhadap frame sesuai metode yang dipilih di GUI."""
        norm_method = normalize_method_name(method)
        t0 = time.perf_counter()

        if norm_method == METHOD_MULTI_ANGLE:
            result = self._match_multi_angle(frame_bgr, template)
        elif norm_method == METHOD_AKAZE_AFFINE:
            result = self._match_akaze_affine(frame_bgr, template)
        else:
            result = self._match_template_fixed(frame_bgr, template)

        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def match_many(
        self,
        frame_bgr: np.ndarray,
        templates: List[TemplateEntry],
        method: str,
    ) -> MatchResult:
        """Mode Auto-Match: membandingkan frame dengan seluruh template tersimpan."""
        norm_method = normalize_method_name(method)
        if not templates:
            return MatchResult(
                matched=False,
                status="NG",
                template_name="NO_TEMPLATE",
                method=norm_method,
                score=0.0,
                threshold=0.80,
                bbox=None,
                polygon=None,
                elapsed_ms=0.0,
                angle_deg=0.0,
                keypoints_matched=0,
            )

        t0 = time.perf_counter()
        results = [self.match_single(frame_bgr, tpl, norm_method) for tpl in templates]
        matched_results = [r for r in results if r.matched]
        if matched_results:
            best = max(matched_results, key=lambda r: r.score)
        else:
            best = max(results, key=lambda r: r.score)

        best.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return best

    # ------------------------------------------------------------------
    # Metode 1: Template Matching (0° Tetap) — cv2.TM_CCOEFF_NORMED
    # ------------------------------------------------------------------
    def _match_template_fixed(
        self, frame_bgr: np.ndarray, template: TemplateEntry
    ) -> MatchResult:
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_tpl = cv2.cvtColor(template.image_bgr, cv2.COLOR_BGR2GRAY)

        fh, fw = gray_frame.shape[:2]
        th, tw = gray_tpl.shape[:2]
        if th > fh or tw > fw or th < 4 or tw < 4:
            return MatchResult(
                matched=False,
                status="NG",
                template_name=template.name,
                method=METHOD_TEMPLATE_FIXED,
                score=0.0,
                threshold=template.threshold,
                bbox=None,
                polygon=None,
                elapsed_ms=0.0,
                angle_deg=0.0,
            )

        res = cv2.matchTemplate(gray_frame, gray_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        score = float(np.clip(max_val, 0.0, 1.0))
        x, y = int(max_loc[0]), int(max_loc[1])
        matched = score >= template.threshold

        return MatchResult(
            matched=matched,
            status="OK" if matched else "NG",
            template_name=template.name,
            method=METHOD_TEMPLATE_FIXED,
            score=score,
            threshold=template.threshold,
            bbox=(x, y, tw, th),
            polygon=None,
            elapsed_ms=0.0,
            angle_deg=0.0,
        )

    # ------------------------------------------------------------------
    # Metode 2: Multi-Angle Template Matching (0°–360°) — Coarse-to-Fine
    # ------------------------------------------------------------------
    def _match_multi_angle(
        self, frame_bgr: np.ndarray, template: TemplateEntry
    ) -> MatchResult:
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_tpl = cv2.cvtColor(template.image_bgr, cv2.COLOR_BGR2GRAY)

        fh, fw = gray_frame.shape[:2]
        th, tw = gray_tpl.shape[:2]
        if th > fh or tw > fw or th < 16 or tw < 16:
            return MatchResult(
                matched=False,
                status="NG",
                template_name=template.name,
                method=METHOD_MULTI_ANGLE,
                score=0.0,
                threshold=template.threshold,
                bbox=None,
                polygon=None,
                elapsed_ms=0.0,
                angle_deg=0.0,
            )

        # === TAHAP 1: COARSE (Skala 0.5x, 0°..350° dengan langkah kasar) ===
        scale = self.coarse_scale
        coarse_frame = cv2.resize(
            gray_frame,
            (max(16, int(round(fw * scale))), max(16, int(round(fh * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        coarse_cache = self._get_coarse_rotated_cache(template.name, gray_tpl)
        ch, cw = coarse_cache.mask.shape[:2]
        if ch > coarse_frame.shape[0] or cw > coarse_frame.shape[1]:
            return MatchResult(
                matched=False,
                status="NG",
                template_name=template.name,
                method=METHOD_MULTI_ANGLE,
                score=0.0,
                threshold=template.threshold,
                bbox=None,
                polygon=None,
                elapsed_ms=0.0,
                angle_deg=0.0,
            )

        coarse_candidates: List[Tuple[float, float, Tuple[int, int]]] = []
        for angle_deg, rot_small in coarse_cache.rotated_templates:
            sc, loc = self._match_masked_sqdiff_score(
                coarse_frame, rot_small, coarse_cache.mask
            )
            coarse_candidates.append((sc, angle_deg, loc))

        coarse_candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_coarse_angle, best_coarse_loc = coarse_candidates[0]

        # === TAHAP 2: FINE (Resolusi Asli 1.0x di sekitar sudut terbaik ± coarse_step) ===
        est_x = int(round(best_coarse_loc[0] / scale))
        est_y = int(round(best_coarse_loc[1] / scale))

        # Cari di jendela ROI lokal di sekitar posisi coarse agar sangat cepat di RPi 5
        margin = max(28, int(round(max(tw, th) * 0.35)))
        roi_x1 = max(0, est_x - margin)
        roi_y1 = max(0, est_y - margin)
        roi_x2 = min(fw, est_x + tw + margin)
        roi_y2 = min(fh, est_y + th + margin)

        if (roi_x2 - roi_x1) < tw:
            roi_x1 = max(0, min(fw - tw, est_x))
            roi_x2 = roi_x1 + tw
        if (roi_y2 - roi_y1) < th:
            roi_y1 = max(0, min(fh - th, est_y))
            roi_y2 = roi_y1 + th

        fine_region = gray_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        fine_mask = self._get_fine_mask(th, tw)

        span = self.coarse_step_deg
        fine_step = self.fine_step_deg
        fine_angles = [
            (best_coarse_angle + delta) % 360.0
            for delta in range(-span, span + 1, fine_step)
        ]

        best_score = -1.0
        best_angle = best_coarse_angle % 360.0
        best_loc = (est_x, est_y)

        for cand_angle in fine_angles:
            rot_full = self._rotate_gray_keep_size(gray_tpl, cand_angle)
            sc, loc = self._match_masked_sqdiff_score(fine_region, rot_full, fine_mask)
            if sc > best_score:
                best_score = sc
                best_angle = cand_angle % 360.0
                best_loc = (roi_x1 + loc[0], roi_y1 + loc[1])

        best_score = float(np.clip(best_score, 0.0, 1.0))
        polygon, bbox = self._rotated_box_corners(
            best_loc[0], best_loc[1], tw, th, best_angle
        )
        matched = best_score >= template.threshold

        return MatchResult(
            matched=matched,
            status="OK" if matched else "NG",
            template_name=template.name,
            method=METHOD_MULTI_ANGLE,
            score=best_score,
            threshold=template.threshold,
            bbox=bbox,
            polygon=polygon,
            elapsed_ms=0.0,
            angle_deg=round(best_angle, 1),
            keypoints_matched=0,
        )

    # ------------------------------------------------------------------
    # Metode 3: Feature Matching (AKAZE + Affine 2D RANSAC)
    # ------------------------------------------------------------------
    def _match_akaze_affine(
        self, frame_bgr: np.ndarray, template: TemplateEntry
    ) -> MatchResult:
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_tpl = cv2.cvtColor(template.image_bgr, cv2.COLOR_BGR2GRAY)
        th, tw = gray_tpl.shape[:2]
        fh, fw = gray_frame.shape[:2]

        kp_tpl, des_tpl = self._get_template_akaze_features(template.name, gray_tpl)
        kp_frm, des_frm = self._akaze.detectAndCompute(gray_frame, None)

        if (
            des_tpl is None
            or des_frm is None
            or len(kp_tpl) < 6
            or len(kp_frm) < 6
        ):
            return MatchResult(
                matched=False,
                status="NG",
                template_name=template.name,
                method=METHOD_AKAZE_AFFINE,
                score=0.0,
                threshold=template.threshold,
                bbox=None,
                polygon=None,
                elapsed_ms=0.0,
                angle_deg=0.0,
                keypoints_matched=0,
            )

        knn_matches = self._bf.knnMatch(des_tpl, des_frm, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 6:
            raw_score = min(0.40, len(good_matches) / max(float(len(kp_tpl)), 20.0))
            return MatchResult(
                matched=False,
                status="NG",
                template_name=template.name,
                method=METHOD_AKAZE_AFFINE,
                score=float(raw_score),
                threshold=template.threshold,
                bbox=None,
                polygon=None,
                elapsed_ms=0.0,
                angle_deg=0.0,
                keypoints_matched=len(good_matches),
            )

        src_pts = np.float32([kp_tpl[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp_frm[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # Affine Partial 2D (rotasi 0°-360° + translasi + skala seragam, tanpa distorsi perspektif)
        affine_mat, mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99,
        )
        if affine_mat is None or mask is None:
            return MatchResult(
                matched=False,
                status="NG",
                template_name=template.name,
                method=METHOD_AKAZE_AFFINE,
                score=0.0,
                threshold=template.threshold,
                bbox=None,
                polygon=None,
                elapsed_ms=0.0,
                angle_deg=0.0,
                keypoints_matched=len(good_matches),
            )

        inliers = int(mask.ravel().sum())
        if inliers < 5:
            score = min(0.38, inliers / 18.0)
            return MatchResult(
                matched=False,
                status="NG",
                template_name=template.name,
                method=METHOD_AKAZE_AFFINE,
                score=float(score),
                threshold=template.threshold,
                bbox=None,
                polygon=None,
                elapsed_ms=0.0,
                angle_deg=0.0,
                keypoints_matched=inliers,
            )

        # Ekstraksi skala seragam dan sudut rotasi 0°-360° dari matriks affine 2x3
        scale = float(np.hypot(affine_mat[0, 0], affine_mat[1, 0]))
        angle_deg = float(
            (-np.degrees(np.arctan2(affine_mat[1, 0], affine_mat[0, 0]))) % 360.0
        )

        # Validasi skala planar 2D (kamera jarak tetap: skala wajar 0.70x - 1.35x)
        valid_scale = 0.70 <= scale <= 1.35

        corners = np.float32(
            [[0.0, 0.0], [tw, 0.0], [tw, th], [0.0, th]]
        ).reshape(-1, 1, 2)
        projected = cv2.transform(corners, affine_mat).reshape(-1, 2)

        inlier_ratio = inliers / max(float(len(good_matches)), 1.0)
        feature_coverage = min(
            1.0, inliers / max(min(len(kp_tpl) * 0.30, 45.0), 12.0)
        )
        score = float(np.clip(0.55 * inlier_ratio + 0.45 * feature_coverage, 0.0, 1.0))

        if not valid_scale:
            score *= 0.35
            bbox = None
            polygon = None
        else:
            polygon = [(int(round(pt[0])), int(round(pt[1]))) for pt in projected]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1 = int(np.clip(min(xs), 0, fw - 1))
            y1 = int(np.clip(min(ys), 0, fh - 1))
            x2 = int(np.clip(max(xs), 0, fw - 1))
            y2 = int(np.clip(max(ys), 0, fh - 1))
            bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        matched = score >= template.threshold and bbox is not None
        return MatchResult(
            matched=matched,
            status="OK" if matched else "NG",
            template_name=template.name,
            method=METHOD_AKAZE_AFFINE,
            score=score,
            threshold=template.threshold,
            bbox=bbox,
            polygon=polygon,
            elapsed_ms=0.0,
            angle_deg=round(angle_deg, 1),
            keypoints_matched=inliers,
        )


@dataclass
class InspectionJob:
    """Paket pekerjaan inspeksi untuk dieksekusi di InspectionWorker thread."""

    frame_bgr: np.ndarray
    templates: List[TemplateEntry]
    method: str
    is_trigger_mode: bool


class InspectionWorker(QThread):
    """Worker QThread terpisah khusus pemrosesan OpenCV Matching."""

    inspection_finished = Signal(object, np.ndarray, bool)

    def __init__(
        self,
        coarse_step_deg: int = 10,
        fine_step_deg: int = 2,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine = TemplateMatcherEngine(
            coarse_scale=0.5,
            coarse_step_deg=coarse_step_deg,
            fine_step_deg=fine_step_deg,
        )
        self._queue: "queue.Queue[Optional[InspectionJob]]" = queue.Queue(maxsize=4)
        self._running = False

    def set_angle_steps(self, coarse_step_deg: int, fine_step_deg: int) -> None:
        self.engine.set_angle_steps(coarse_step_deg, fine_step_deg)

    def submit_job(self, job: InspectionJob) -> bool:
        if not self._running:
            return False

        if not job.is_trigger_mode:
            if self._queue.full():
                return False
            try:
                self._queue.put_nowait(job)
                return True
            except queue.Full:
                return False

        while self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(job)
        return True

    def clear_feature_cache(self, template_name: Optional[str] = None) -> None:
        self.engine.clear_cache(template_name)

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if job is None:
                break

            if len(job.templates) == 1:
                result = self.engine.match_single(
                    job.frame_bgr, job.templates[0], job.method
                )
            else:
                result = self.engine.match_many(
                    job.frame_bgr, job.templates, job.method
                )

            self.inspection_finished.emit(result, job.frame_bgr, job.is_trigger_mode)

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self.wait(3000)
