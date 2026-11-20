"""
data_logger.py
--------------
Modul pencatatan log inspeksi ke SQLite lokal (data/inspection_logs.db),
auto-save foto bukti NG dengan overlay teks & bounding box, ekspor CSV ke exports/,
serta auto-cleanup foto NG lama di background QThread.

100% kompatibel lintas OS (Windows 10/11 & Raspberry Pi OS 64-bit) menggunakan
pathlib.Path, encoding="utf-8", newline="" (CSV), dan penulisan JPEG tahan Unicode.
"""

from __future__ import annotations

import csv
import queue
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from matcher import MatchResult


@dataclass
class LogEntry:
    id: int
    timestamp: str
    template_name: str
    method: str
    score: float
    threshold: float
    status: str
    image_path: str


@dataclass
class LogWriteJob:
    result: MatchResult
    frame_bgr: Optional[np.ndarray]
    timestamp_dt: datetime


class DataLoggerWorker(QThread):
    """
    Worker QThread terpisah untuk menulis database SQLite dan menyimpan file JPG NG
    agar operasi I/O disk tidak pernah mengganggu FPS kamera atau respons GUI.
    """

    log_saved = Signal(object)  # LogEntry
    cleanup_completed = Signal(int)  # jumlah folder dihapus
    logger_error = Signal(str)

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        retention_days: int = 30,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.base_dir: Path = (
            Path(base_dir).resolve()
            if base_dir is not None
            else Path(__file__).resolve().parent
        )
        self.data_dir: Path = self.base_dir / "data"
        self.ng_dir: Path = self.base_dir / "ng_captures"
        self.exports_dir: Path = self.base_dir / "exports"
        self.db_path: Path = self.data_dir / "inspection_logs.db"
        self.retention_days = retention_days

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ng_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        self._queue: "queue.Queue[Optional[LogWriteJob]]" = queue.Queue()
        self._running = False
        self._init_database()

    def resolve_image_path(self, rel_path: str) -> Path:
        """Mengonversi path relatif dari SQLite menjadi pathlib.Path yang valid di Windows/Linux."""
        normalized = rel_path.replace("\\\\", "/")
        parts = PurePosixPath(normalized).parts
        return self.base_dir.joinpath(*parts)

    def _init_database(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inspection_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    template_name TEXT NOT NULL,
                    method TEXT NOT NULL,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    status TEXT NOT NULL,
                    image_path TEXT NOT NULL DEFAULT ''
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_inspection_timestamp
                ON inspection_logs(timestamp);
                """
            )
            conn.commit()
        finally:
            conn.close()

    def enqueue_inspection(
        self, result: MatchResult, frame_bgr: Optional[np.ndarray]
    ) -> None:
        frame_copy = (
            frame_bgr.copy()
            if (frame_bgr is not None and result.status == "NG")
            else None
        )
        self._queue.put(
            LogWriteJob(
                result=result,
                frame_bgr=frame_copy,
                timestamp_dt=datetime.now(),
            )
        )

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_\\-]+", "_", text.strip())
        return safe.strip("_") or "unknown"

    @staticmethod
    def _draw_ng_overlay(
        frame_bgr: np.ndarray,
        result: MatchResult,
        timestamp_str: str,
    ) -> np.ndarray:
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        if result.polygon and len(result.polygon) == 4:
            pts = np.array(result.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (40, 40, 240), 3, cv2.LINE_AA)
        elif result.bbox is not None:
            x, y, bw, bh = result.bbox
            cv2.rectangle(
                annotated, (x, y), (x + bw, y + bh), (40, 40, 240), 3, cv2.LINE_AA
            )

        banner_h = 68
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 16, 26), -1)
        cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)
        cv2.line(annotated, (0, banner_h), (w, banner_h), (48, 59, 239), 2)

        line1 = (
            f"STATUS: NG / FAIL  |  TEMPLATE: {result.template_name}  |  "
            f"SKOR: {result.score * 100:.1f}% (MIN {result.threshold * 100:.1f}%)"
        )
        line2 = (
            f"WAKTU: {timestamp_str}  |  SUDUT: {result.angle_deg:.1f} deg  |  "
            f"METODE: {result.method}"
        )

        cv2.putText(
            annotated,
            line1,
            (14, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (95, 110, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line2,
            (14, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (225, 232, 240),
            1,
            cv2.LINE_AA,
        )
        return annotated

    def _save_ng_photo(
        self,
        frame_bgr: np.ndarray,
        result: MatchResult,
        ts_dt: datetime,
    ) -> str:
        """
        Menyimpan foto NG ke:
        ng_captures/YYYY-MM-DD/NG_<timestamp>_<nama_template>_<skor>.jpg
        menggunakan pathlib.Path + cv2.imencode (aman untuk path Windows berkarakter Unicode).
        """
        date_folder = self.ng_dir / ts_dt.strftime("%Y-%m-%d")
        date_folder.mkdir(parents=True, exist_ok=True)

        ts_file = ts_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_tpl = self._sanitize_filename(result.template_name)
        score_str = f"{result.score * 100:.1f}pct"
        filename = f"NG_{ts_file}_{safe_tpl}_{score_str}.jpg"
        full_path = date_folder / filename

        annotated = self._draw_ng_overlay(
            frame_bgr, result, ts_dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        ok, encoded = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not ok:
            raise IOError(f"Gagal mengenkode gambar NG ke JPEG: {full_path}")

        full_path.write_bytes(encoded.tobytes())
        return full_path.relative_to(self.base_dir).as_posix()

    def run(self) -> None:
        self._running = True
        if self.retention_days > 0:
            removed = self.cleanup_old_ng_captures(self.retention_days)
            if removed > 0:
                self.cleanup_completed.emit(removed)

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            while self._running or not self._queue.empty():
                try:
                    job = self._queue.get(timeout=0.15)
                except queue.Empty:
                    continue

                if job is None:
                    break

                try:
                    ts_str = job.timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
                    image_rel_path = ""
                    if job.result.status == "NG" and job.frame_bgr is not None:
                        image_rel_path = self._save_ng_photo(
                            job.frame_bgr, job.result, job.timestamp_dt
                        )

                    cur = conn.execute(
                        """
                        INSERT INTO inspection_logs (
                            timestamp, template_name, method, score, threshold, status, image_path
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ts_str,
                            job.result.template_name,
                            job.result.method,
                            round(float(job.result.score), 4),
                            round(float(job.result.threshold), 4),
                            job.result.status,
                            image_rel_path,
                        ),
                    )
                    conn.commit()
                    entry = LogEntry(
                        id=int(cur.lastrowid),
                        timestamp=ts_str,
                        template_name=job.result.template_name,
                        method=job.result.method,
                        score=round(float(job.result.score), 4),
                        threshold=round(float(job.result.threshold), 4),
                        status=job.result.status,
                        image_path=image_rel_path,
                    )
                    self.log_saved.emit(entry)
                except Exception as exc:
                    self.logger_error.emit(f"Gagal menyimpan log/foto NG: {exc}")
        finally:
            conn.close()

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)
        self.wait(4000)

    def get_recent_logs(self, limit: int = 50) -> List[LogEntry]:
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT id, timestamp, template_name, method, score, threshold, status, image_path
                FROM inspection_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [LogEntry(*row) for row in rows]
        finally:
            conn.close()

    def get_today_stats(self) -> Tuple[int, int]:
        today_prefix = datetime.now().strftime("%Y-%m-%d") + "%"
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT status, COUNT(*)
                FROM inspection_logs
                WHERE timestamp LIKE ?
                GROUP BY status
                """,
                (today_prefix,),
            ).fetchall()
            counts = {status: int(cnt) for status, cnt in rows}
            return counts.get("OK", 0), counts.get("NG", 0)
        finally:
            conn.close()

    def export_to_csv(
        self, start_date: str, end_date: str
    ) -> Tuple[Path, int]:
        """
        Mengekspor log inspeksi dalam rentang tanggal (YYYY-MM-DD s/d YYYY-MM-DD)
        ke folder exports/ dengan encoding="utf-8" dan newline="".
        """
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        start_ts = f"{start_date} 00:00:00"
        end_ts = f"{end_date} 23:59:59"

        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT id, timestamp, template_name, method, score, threshold, status, image_path
                FROM inspection_logs
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY id ASC
                """,
                (start_ts, end_ts),
            ).fetchall()
        finally:
            conn.close()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = (
            self.exports_dir
            / f"inspection_logs_{start_date}_to_{end_date}_{stamp}.csv"
        )
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "timestamp",
                    "template_name",
                    "method",
                    "score",
                    "threshold",
                    "status",
                    "image_path",
                ]
            )
            writer.writerows(rows)

        return out_file, len(rows)

    def cleanup_old_ng_captures(self, retention_days: int) -> int:
        if retention_days <= 0 or not self.ng_dir.exists():
            return 0

        cutoff = datetime.now().date() - timedelta(days=retention_days)
        removed_count = 0
        for child in self.ng_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                folder_date = datetime.strptime(child.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if folder_date < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed_count += 1
        return removed_count
