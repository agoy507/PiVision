"""
main.py
-------
Aplikasi GUI Machine Vision Industri 100% Offline lintas OS:
- Siap debug langsung di Windows 10/11 64-bit (MockGPIOController + CAP_DSHOW).
- Siap deploy langsung di Raspberry Pi 5 64-bit (gpiozero/lgpio + V4L2/PiCamera2)
  tanpa perlu mengubah satu baris kode pun.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QDate, QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from camera_worker import CameraWorker, create_camera_source
from data_logger import DataLoggerWorker, LogEntry
from gpio_controller import create_gpio_controller
from matcher import (
    AVAILABLE_METHODS,
    InspectionJob,
    InspectionWorker,
    MatchResult,
)
from template_manager import TemplateManager


DARK_INDUSTRIAL_QSS = """
QMainWindow, QWidget {
    background-color: #0B0F17;
    color: #E6EDF5;
    font-family: "Segoe UI", "DejaVu Sans", sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #121824;
    border: 1px solid #243044;
    border-radius: 6px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    color: #B7C5D8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: #8BA3C2;
}
QPushButton {
    background-color: #1A2436;
    border: 1px solid #2D3E58;
    border-radius: 6px;
    color: #E6EDF5;
    padding: 10px 14px;
    min-height: 26px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #223047;
    border-color: #3D5478;
}
QPushButton:pressed {
    background-color: #141C2A;
}
QPushButton:disabled {
    background-color: #111622;
    color: #55647B;
    border-color: #1C2536;
}
QPushButton[role="primary"] {
    background-color: #0284C7;
    border-color: #0369A1;
    color: #FFFFFF;
}
QPushButton[role="primary"]:hover {
    background-color: #039BE5;
}
QPushButton[role="success"] {
    background-color: #059669;
    border-color: #047857;
    color: #FFFFFF;
}
QPushButton[role="success"]:hover {
    background-color: #10B981;
}
QPushButton[role="danger"] {
    background-color: #DC2626;
    border-color: #B91C1C;
    color: #FFFFFF;
}
QPushButton[role="danger"]:hover {
    background-color: #EF4444;
}
QPushButton[role="warning"] {
    background-color: #D97706;
    border-color: #B45309;
    color: #FFFFFF;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QPlainTextEdit {
    background-color: #0E1420;
    border: 1px solid #2A3850;
    border-radius: 5px;
    padding: 7px 10px;
    color: #F1F5F9;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {
    min-height: 24px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #06B6D4;
}
QListWidget, QTableWidget {
    background-color: #0E1420;
    border: 1px solid #243044;
    border-radius: 5px;
    gridline-color: #1D283A;
    selection-background-color: #1E3A5F;
    selection-color: #FFFFFF;
}
QHeaderView::section {
    background-color: #161F30;
    color: #9FB2C9;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #243044;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid #243044;
    background-color: #121824;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #151D2C;
    color: #8BA3C2;
    padding: 10px 16px;
    border: 1px solid #243044;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 3px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #1E2B40;
    color: #FFFFFF;
    border-color: #06B6D4;
}
QSlider::groove:horizontal {
    height: 8px;
    background: #1D283A;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #06B6D4;
    width: 22px;
    height: 22px;
    margin: -7px 0;
    border-radius: 11px;
}
"""


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Konversi aman array OpenCV BGR menjadi QImage RGB888."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


class VideoCanvasWidget(QWidget):
    """Panel video utama dengan seleksi ROI drag-and-drop & overlay deteksi."""

    roi_changed = Signal(tuple)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._qimage: Optional[QImage] = None
        self._frame_size: Tuple[int, int] = (0, 0)
        self._draw_rect = QRect()

        self._teach_mode: bool = False
        self._dragging: bool = False
        self._drag_start_widget = QPoint()
        self._drag_end_widget = QPoint()
        self._roi_frame: Optional[Tuple[int, int, int, int]] = None

        self._last_result: Optional[MatchResult] = None
        self._status_badge: str = "STANDBY"
        self._placeholder_text: str = "Menghubungkan ke kamera..."

    def set_teach_mode(self, enabled: bool) -> None:
        self._teach_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        if not enabled:
            self._dragging = False
        self.update()

    def set_placeholder_text(self, text: str) -> None:
        self._placeholder_text = text
        self.update()

    def clear_roi(self) -> None:
        self._roi_frame = None
        self._dragging = False
        self.roi_changed.emit((0, 0, 0, 0))
        self.update()

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._roi_frame

    def set_match_overlay(
        self, result: Optional[MatchResult], status_badge: str = "STANDBY"
    ) -> None:
        self._last_result = result
        self._status_badge = status_badge
        self.update()

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        self._frame_size = (w, h)
        self._qimage = bgr_to_qimage(frame_bgr)
        self.update()

    def _compute_draw_rect(self) -> QRect:
        fw, fh = self._frame_size
        if fw <= 0 or fh <= 0:
            return QRect()
        ww, wh = self.width(), self.height()
        scale = min(ww / fw, wh / fh)
        dw = int(fw * scale)
        dh = int(fh * scale)
        return QRect((ww - dw) // 2, (wh - dh) // 2, dw, dh)

    def _widget_to_frame(self, pt: QPoint) -> Tuple[int, int]:
        rect = self._draw_rect
        fw, fh = self._frame_size
        if rect.width() <= 0 or rect.height() <= 0 or fw <= 0 or fh <= 0:
            return 0, 0
        rx = max(rect.left(), min(rect.right(), pt.x()))
        ry = max(rect.top(), min(rect.bottom(), pt.y()))
        fx = int(((rx - rect.left()) / rect.width()) * fw)
        fy = int(((ry - rect.top()) / rect.height()) * fh)
        return max(0, min(fw - 1, fx)), max(0, min(fh - 1, fy))

    def _frame_to_widget(self, fx: float, fy: float) -> QPoint:
        rect = self._draw_rect
        fw, fh = self._frame_size
        if fw <= 0 or fh <= 0:
            return QPoint(0, 0)
        wx = rect.left() + int((fx / fw) * rect.width())
        wy = rect.top() + int((fy / fh) * rect.height())
        return QPoint(wx, wy)

    def mousePressEvent(self, event) -> None:
        if (
            self._teach_mode
            and event.button() == Qt.LeftButton
            and self._qimage is not None
        ):
            pos = event.position().toPoint()
            if self._draw_rect.contains(pos):
                self._dragging = True
                self._drag_start_widget = pos
                self._drag_end_widget = pos
                self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and self._teach_mode:
            self._drag_end_widget = event.position().toPoint()
            self._update_roi_from_drag()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging and event.button() == Qt.LeftButton:
            self._drag_end_widget = event.position().toPoint()
            self._dragging = False
            self._update_roi_from_drag()
            self.update()

    def _update_roi_from_drag(self) -> None:
        x1, y1 = self._widget_to_frame(self._drag_start_widget)
        x2, y2 = self._widget_to_frame(self._drag_end_widget)
        rx = min(x1, x2)
        ry = min(y1, y2)
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)
        if rw >= 16 and rh >= 16:
            self._roi_frame = (rx, ry, rw, rh)
            self.roi_changed.emit(self._roi_frame)
        else:
            self._roi_frame = None
            self.roi_changed.emit((0, 0, 0, 0))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#070A10"))

        if self._qimage is None:
            painter.setPen(QColor("#94A3B8"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(
                self.rect().adjusted(24, 24, -24, -24),
                Qt.AlignCenter | Qt.TextWordWrap,
                self._placeholder_text,
            )
            return

        self._draw_rect = self._compute_draw_rect()
        painter.drawImage(self._draw_rect, self._qimage)

        if self._teach_mode:
            if self._roi_frame is not None:
                rx, ry, rw, rh = self._roi_frame
                tl = self._frame_to_widget(rx, ry)
                br = self._frame_to_widget(rx + rw, ry + rh)
                roi_rect = QRect(tl, br)

                painter.setPen(QPen(QColor("#06B6D4"), 2, Qt.SolidLine))
                painter.setBrush(QColor(6, 182, 212, 35))
                painter.drawRect(roi_rect)

                label = f"ROI: {rw}x{rh} px @ ({rx}, {ry})"
                painter.setPen(QColor("#E0F2FE"))
                painter.setFont(QFont("Consolas", 10, QFont.Bold))
                tag_rect = QRect(roi_rect.left(), max(8, roi_rect.top() - 26), 240, 22)
                painter.fillRect(tag_rect, QColor(8, 22, 34, 210))
                painter.drawText(tag_rect.adjusted(6, 0, -4, 0), Qt.AlignVCenter, label)
            elif self._dragging:
                drag_rect = QRect(
                    self._drag_start_widget, self._drag_end_widget
                ).normalized()
                painter.setPen(QPen(QColor("#38BDF8"), 2, Qt.DashLine))
                painter.drawRect(drag_rect)
            return

        if self._last_result is not None:
            res = self._last_result
            color = QColor("#10B981") if res.matched else QColor("#EF4444")
            painter.setPen(QPen(color, 3))

            if res.polygon and len(res.polygon) == 4:
                pts = [self._frame_to_widget(px, py) for px, py in res.polygon]
                for i in range(4):
                    painter.drawLine(pts[i], pts[(i + 1) % 4])
                # Tandai sisi atas (sudut 0 -> 1) dengan warna cyan agar arah rotasi jelas
                painter.setPen(QPen(QColor("#38BDF8"), 3))
                painter.drawLine(pts[0], pts[1])
                painter.setBrush(QColor("#38BDF8"))
                painter.drawEllipse(pts[0], 4, 4)
                painter.setBrush(Qt.NoBrush)
                anchor = QPoint(min(p.x() for p in pts), min(p.y() for p in pts))
            elif res.bbox is not None:
                bx, by, bw, bh = res.bbox
                tl = self._frame_to_widget(bx, by)
                br = self._frame_to_widget(bx + bw, by + bh)
                painter.drawRect(QRect(tl, br))
                anchor = tl
            else:
                anchor = QPoint(self._draw_rect.left() + 14, self._draw_rect.top() + 56)

            info_text = (
                f"{res.status} | {res.template_name} | "
                f"{res.score * 100:.1f}% | {res.angle_deg:.1f} deg"
            )
            painter.setFont(QFont("Consolas", 10, QFont.Bold))
            box_w = min(self._draw_rect.width() - 20, max(280, len(info_text) * 8 + 20))
            lbl_rect = QRect(
                max(self._draw_rect.left() + 8, anchor.x()),
                max(self._draw_rect.top() + 8, anchor.y() - 30),
                box_w,
                26,
            )
            painter.fillRect(lbl_rect, QColor(10, 15, 24, 215))
            painter.setPen(color)
            painter.drawRect(lbl_rect)
            painter.setPen(QColor("#F8FAFC"))
            painter.drawText(lbl_rect.adjusted(8, 0, -6, 0), Qt.AlignVCenter, info_text)

        badge_colors = {
            "OK": QColor("#059669"),
            "NG": QColor("#DC2626"),
            "STANDBY": QColor("#334155"),
            "SETTLING": QColor("#D97706"),
        }
        badge_rect = QRect(
            self._draw_rect.left() + 12, self._draw_rect.top() + 12, 165, 32
        )
        painter.fillRect(
            badge_rect, badge_colors.get(self._status_badge, QColor("#334155"))
        )
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        painter.drawText(badge_rect, Qt.AlignCenter, f"STATUS: {self._status_badge}")


class ExportCSVDialog(QDialog):
    """Dialog pemilih rentang tanggal untuk mengekspor log inspeksi ke CSV."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Log Inspeksi ke CSV")
        self.setMinimumWidth(360)

        today = QDate.currentDate()
        self.start_date = QDateEdit(today)
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("yyyy-MM-dd")

        self.end_date = QDateEdit(today)
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd")

        form = QFormLayout()
        form.addRow("Tanggal Mulai:", self.start_date)
        form.addRow("Tanggal Akhir:", self.end_date)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export CSV")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def get_date_range(self) -> Tuple[str, str]:
        return (
            self.start_date.date().toString("yyyy-MM-dd"),
            self.end_date.date().toString("yyyy-MM-dd"),
        )


class NGImagePreviewDialog(QDialog):
    """Jendela penampil foto bukti NG saat baris log diklik."""

    def __init__(self, entry: LogEntry, image_abs_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Bukti Foto NG — {entry.template_name} ({entry.timestamp})")
        self.resize(900, 620)

        layout = QVBoxLayout(self)
        meta = QLabel(
            f"Waktu: {entry.timestamp}  |  Template: {entry.template_name}  |  "
            f"Metode: {entry.method}  |  Skor: {entry.score * 100:.1f}% "
            f"(Threshold: {entry.threshold * 100:.1f}%)"
        )
        meta.setStyleSheet("color: #FCA5A5; font-weight: 600; padding: 4px;")
        layout.addWidget(meta)

        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet(
            "background-color: #070A10; border: 1px solid #243044;"
        )
        if image_abs_path.exists():
            pixmap = QPixmap(str(image_abs_path))
            img_label.setPixmap(
                pixmap.scaled(860, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            img_label.setText(f"File foto tidak ditemukan:\\n{image_abs_path}")
        layout.addWidget(img_label, 1)

        close_btn = QPushButton("Tutup")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class VisionMainWindow(QMainWindow):
    """Jendela utama aplikasi Machine Vision Industri lintas OS (Windows & RPi 5)."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PiVision RPi5 — Industrial Machine Vision System")
        self.resize(1380, 860)

        self.base_dir = Path(__file__).resolve().parent
        self.template_manager = TemplateManager(self.base_dir)
        self.cfg = self.template_manager.config

        self._mode: str = "RUN"
        self._is_running: bool = False
        self._teach_frozen: bool = False
        self._frozen_frame: Optional[np.ndarray] = None
        self._trigger_pending: bool = False
        self._last_continuous_log_ts: float = 0.0
        self._last_fps: float = 0.0

        self._daily_ok: int = 0
        self._daily_ng: int = 0
        self._recent_logs: List[LogEntry] = []

        # Inisialisasi subsistem lintas OS
        self.logger_worker = DataLoggerWorker(
            self.base_dir, retention_days=self.cfg.ng_retention_days
        )
        self.inspection_worker = InspectionWorker(
            coarse_step_deg=self.cfg.multi_angle_coarse_step,
            fine_step_deg=self.cfg.multi_angle_fine_step,
        )
        self.gpio = create_gpio_controller(self.cfg.gpio)
        self.gpio.set_mute_buzzer(self.cfg.mute_buzzer)

        cam_source = create_camera_source(
            backend=self.cfg.camera_backend,
            device_index=self.cfg.camera_index,
            width=self.cfg.camera_width,
            height=self.cfg.camera_height,
            fps=self.cfg.camera_fps,
        )
        self.camera_worker = CameraWorker(cam_source)

        self._trigger_delay_timer = QTimer(self)
        self._trigger_delay_timer.setSingleShot(True)
        self._trigger_delay_timer.timeout.connect(self._execute_triggered_capture)

        self._build_ui()
        self._connect_signals()
        self._refresh_template_list()
        self._load_initial_stats_and_logs()

        self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._space_shortcut.activated.connect(self._on_manual_trigger_clicked)

        self.logger_worker.start()
        self.inspection_worker.start()
        self.camera_worker.start()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # === PANEL KIRI / UTAMA: Live Camera + Status Bar + Badge Simulasi GPIO ===
        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        self.video_canvas = VideoCanvasWidget()
        left_col.addWidget(self.video_canvas, 1)

        info_bar = QFrame()
        info_bar.setStyleSheet(
            "QFrame { background-color: #121824; border: 1px solid #243044; border-radius: 6px; }"
        )
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(12, 8, 12, 8)
        info_layout.setSpacing(12)

        self.lbl_camera_info = QLabel("Kamera: Memulai...")
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setStyleSheet(
            "font-family: 'Consolas', 'DejaVu Sans Mono'; font-weight: 700; color: #38BDF8;"
        )

        # Badge indikator mode GPIO ([SIMULASI GPIO - WINDOWS] atau [GPIO AKTIF - RPI5 LGPIO])
        self.lbl_gpio_badge = QLabel(self.gpio.mode_badge)
        if self.gpio.simulation_mode:
            self.lbl_gpio_badge.setStyleSheet(
                "background-color: #78350F; border: 1px solid #F59E0B; border-radius: 4px; "
                "padding: 3px 8px; font-family: 'Consolas', 'DejaVu Sans Mono'; "
                "font-size: 11px; font-weight: 700; color: #FEF3C7;"
            )
        else:
            self.lbl_gpio_badge.setStyleSheet(
                "background-color: #065F46; border: 1px solid #10B981; border-radius: 4px; "
                "padding: 3px 8px; font-family: 'Consolas', 'DejaVu Sans Mono'; "
                "font-size: 11px; font-weight: 700; color: #ECFDF5;"
            )

        info_layout.addWidget(self.lbl_camera_info, 1)
        info_layout.addWidget(self.lbl_fps)
        info_layout.addWidget(self.lbl_gpio_badge)
        left_col.addWidget(info_bar)

        root_layout.addLayout(left_col, 7)

        # === PANEL KANAN ===
        right_panel = QWidget()
        right_panel.setFixedWidth(480)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        top_ctrl = QHBoxLayout()
        top_ctrl.setSpacing(6)
        self.btn_mode_run = QPushButton("MODE RUN")
        self.btn_mode_teach = QPushButton("MODE TEACH")
        self.btn_start_stop = QPushButton("START INSPEKSI")
        self.btn_start_stop.setProperty("role", "success")

        top_ctrl.addWidget(self.btn_mode_run)
        top_ctrl.addWidget(self.btn_mode_teach)
        top_ctrl.addWidget(self.btn_start_stop, 1)
        right_layout.addLayout(top_ctrl)

        self.right_tabs = QTabWidget()
        right_layout.addWidget(self.right_tabs, 1)

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self._build_run_panel())
        self.mode_stack.addWidget(self._build_teach_panel())

        tab1_scroll = QScrollArea()
        tab1_scroll.setWidgetResizable(True)
        tab1_scroll.setFrameShape(QFrame.NoFrame)
        tab1_scroll.setWidget(self.mode_stack)
        self.right_tabs.addTab(tab1_scroll, "Inspeksi / Teach")

        tab2_scroll = QScrollArea()
        tab2_scroll.setWidgetResizable(True)
        tab2_scroll.setFrameShape(QFrame.NoFrame)
        tab2_scroll.setWidget(self._build_io_settings_panel())
        self.right_tabs.addTab(tab2_scroll, "I/O & Debug")

        self.right_tabs.addTab(self._build_log_panel(), "Riwayat Log (50)")

        root_layout.addWidget(right_panel, 0)
        self._apply_mode_ui()

    def _build_run_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        self.status_banner = QLabel("STANDBY")
        self.status_banner.setAlignment(Qt.AlignCenter)
        self.status_banner.setMinimumHeight(88)
        self.status_banner.setFont(QFont("Segoe UI", 26, QFont.Bold))
        layout.addWidget(self.status_banner)

        self.lbl_result_detail = QLabel("")
        self.lbl_result_detail.setAlignment(Qt.AlignCenter)
        self.lbl_result_detail.setStyleSheet(
            "font-family: 'Consolas', 'DejaVu Sans Mono'; font-size: 12px; color: #CBD5E1; padding: 2px;"
        )
        layout.addWidget(self.lbl_result_detail)
        self._set_status_banner("STANDBY", "Siaga — Tekan START lalu picu sensor / Spasi")

        trig_box = QGroupBox("Kontrol Trigger & Mode Operasi")
        trig_layout = QVBoxLayout(trig_box)
        trig_layout.setSpacing(8)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode Operasi:"))
        self.cmb_op_mode = QComboBox()
        self.cmb_op_mode.addItems(["Trigger Mode (Sensor/Button)", "Continuous Mode"])
        self.cmb_op_mode.setCurrentIndex(
            0 if self.cfg.operation_mode == "trigger" else 1
        )
        mode_row.addWidget(self.cmb_op_mode, 1)
        trig_layout.addLayout(mode_row)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Metode:"))
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(AVAILABLE_METHODS)
        idx = self.cmb_method.findText(self.cfg.matching_method)
        if idx >= 0:
            self.cmb_method.setCurrentIndex(idx)
        method_row.addWidget(self.cmb_method, 1)
        trig_layout.addLayout(method_row)

        self.btn_manual_trigger = QPushButton("MANUAL TRIGGER  [SPASI]")
        self.btn_manual_trigger.setProperty("role", "primary")
        self.btn_manual_trigger.setMinimumHeight(52)
        self.btn_manual_trigger.setFont(QFont("Segoe UI", 13, QFont.Bold))
        trig_layout.addWidget(self.btn_manual_trigger)

        self.lbl_sensor_state = QLabel(
            f"Sensor GPIO {self.cfg.gpio.trigger_pin}: IDLE (HIGH)"
        )
        self.lbl_sensor_state.setStyleSheet(
            "font-family: 'Consolas', 'DejaVu Sans Mono'; color: #94A3B8;"
        )
        trig_layout.addWidget(self.lbl_sensor_state)
        layout.addWidget(trig_box)

        stats_box = QGroupBox("Statistik Inspeksi Harian")
        stats_grid = QGridLayout(stats_box)
        self.lbl_count_ok = QLabel("0", alignment=Qt.AlignCenter)
        self.lbl_count_ok.setStyleSheet(
            "font-family: 'Consolas', 'DejaVu Sans Mono'; font-size: 24px; font-weight: 700; color: #10B981;"
        )
        self.lbl_count_ng = QLabel("0", alignment=Qt.AlignCenter)
        self.lbl_count_ng.setStyleSheet(
            "font-family: 'Consolas', 'DejaVu Sans Mono'; font-size: 24px; font-weight: 700; color: #EF4444;"
        )
        self.lbl_yield = QLabel("0.0%", alignment=Qt.AlignCenter)
        self.lbl_yield.setStyleSheet(
            "font-family: 'Consolas', 'DejaVu Sans Mono'; font-size: 24px; font-weight: 700; color: #38BDF8;"
        )

        stats_grid.addWidget(QLabel("Total OK", alignment=Qt.AlignCenter), 0, 0)
        stats_grid.addWidget(QLabel("Total NG", alignment=Qt.AlignCenter), 0, 1)
        stats_grid.addWidget(QLabel("Yield / Pass", alignment=Qt.AlignCenter), 0, 2)
        stats_grid.addWidget(self.lbl_count_ok, 1, 0)
        stats_grid.addWidget(self.lbl_count_ng, 1, 1)
        stats_grid.addWidget(self.lbl_yield, 1, 2)

        self.btn_reset_counter = QPushButton("Reset Counter")
        stats_grid.addWidget(self.btn_reset_counter, 2, 0, 1, 3)
        layout.addWidget(stats_box)

        tpl_box = QGroupBox("Recipe / Template Referensi")
        tpl_layout = QVBoxLayout(tpl_box)
        tpl_layout.setSpacing(8)

        self.chk_auto_match = QCheckBox("Auto-Match (Cocokkan semua template)")
        self.chk_auto_match.setChecked(self.cfg.match_all_templates)
        tpl_layout.addWidget(self.chk_auto_match)

        list_and_thumb = QHBoxLayout()
        self.list_templates = QListWidget()
        self.list_templates.setMaximumHeight(140)
        list_and_thumb.addWidget(self.list_templates, 1)

        thumb_col = QVBoxLayout()
        self.lbl_tpl_thumb = QLabel("Pratinjau", alignment=Qt.AlignCenter)
        self.lbl_tpl_thumb.setFixedSize(120, 100)
        self.lbl_tpl_thumb.setStyleSheet(
            "background-color: #070A10; border: 1px solid #243044; border-radius: 4px; color: #64748B;"
        )
        self.lbl_tpl_meta = QLabel("-", alignment=Qt.AlignCenter)
        self.lbl_tpl_meta.setStyleSheet("font-size: 11px; color: #94A3B8;")
        thumb_col.addWidget(self.lbl_tpl_thumb)
        thumb_col.addWidget(self.lbl_tpl_meta)
        list_and_thumb.addLayout(thumb_col)
        tpl_layout.addLayout(list_and_thumb)

        self.btn_delete_template = QPushButton("Hapus Template Terpilih")
        self.btn_delete_template.setProperty("role", "danger")
        tpl_layout.addWidget(self.btn_delete_template)
        layout.addWidget(tpl_box)
        layout.addStretch(1)
        return panel

    def _build_teach_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        guide = QLabel(
            "Langkah Rekam Template (Teach-by-Example):\\n"
            "1. Posisikan objek referensi di depan kamera.\\n"
            "2. Tekan 'Freeze / Capture' untuk membekukan frame.\\n"
            "3. Click-and-drag pada layar video untuk memilih kotak ROI.\\n"
            "4. Isi Nama Template, atur Threshold, lalu klik 'Simpan Template'."
        )
        guide.setWordWrap(True)
        guide.setStyleSheet(
            "background-color: #162032; border: 1px solid #293B57; border-radius: 6px; padding: 10px; color: #CBD5E1;"
        )
        layout.addWidget(guide)

        capture_row = QHBoxLayout()
        self.btn_freeze = QPushButton("FREEZE / CAPTURE FRAME")
        self.btn_freeze.setProperty("role", "warning")
        self.btn_freeze.setMinimumHeight(48)
        self.btn_clear_roi = QPushButton("Hapus ROI")
        self.btn_clear_roi.setMinimumHeight(48)
        capture_row.addWidget(self.btn_freeze, 2)
        capture_row.addWidget(self.btn_clear_roi, 1)
        layout.addLayout(capture_row)

        form_box = QGroupBox("Detail Template Baru")
        form = QVBoxLayout(form_box)
        form.setSpacing(10)

        self.lbl_roi_info = QLabel("ROI: Belum dipilih (drag mouse di layar video)")
        self.lbl_roi_info.setStyleSheet(
            "font-family: 'Consolas', 'DejaVu Sans Mono'; color: #38BDF8;"
        )
        form.addWidget(self.lbl_roi_info)

        form.addWidget(QLabel("Nama Template:"))
        self.txt_tpl_name = QLineEdit()
        self.txt_tpl_name.setPlaceholderText("Contoh: PCB_Connector_J4")
        form.addWidget(self.txt_tpl_name)

        form.addWidget(QLabel("Threshold Kemiripan (0.50 – 0.99):"))
        thresh_row = QHBoxLayout()
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(50, 99)
        self.slider_threshold.setValue(80)
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.50, 0.99)
        self.spin_threshold.setSingleStep(0.01)
        self.spin_threshold.setDecimals(2)
        self.spin_threshold.setValue(0.80)
        thresh_row.addWidget(self.slider_threshold, 1)
        thresh_row.addWidget(self.spin_threshold)
        form.addLayout(thresh_row)

        self.btn_save_template = QPushButton("SIMPAN TEMPLATE REFERENSI")
        self.btn_save_template.setProperty("role", "success")
        self.btn_save_template.setMinimumHeight(52)
        self.btn_save_template.setFont(QFont("Segoe UI", 13, QFont.Bold))
        form.addWidget(self.btn_save_template)

        layout.addWidget(form_box)
        layout.addStretch(1)
        return panel

    def _build_io_settings_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        timing_box = QGroupBox("Timing Inspeksi, Multi-Angle & Penyimpanan")
        form = QFormLayout(timing_box)
        form.setSpacing(10)

        self.spin_trigger_delay = QSpinBox()
        self.spin_trigger_delay.setRange(0, 5000)
        self.spin_trigger_delay.setSingleStep(50)
        self.spin_trigger_delay.setSuffix(" ms")
        self.spin_trigger_delay.setValue(self.cfg.trigger_delay_ms)
        form.addRow("Trigger Delay (Settling):", self.spin_trigger_delay)

        self.spin_hold_time = QDoubleSpinBox()
        self.spin_hold_time.setRange(0.2, 30.0)
        self.spin_hold_time.setSingleStep(0.5)
        self.spin_hold_time.setDecimals(1)
        self.spin_hold_time.setSuffix(" detik")
        self.spin_hold_time.setValue(self.cfg.output_hold_sec)
        form.addRow("Output Hold Time:", self.spin_hold_time)

        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(0.5, 60.0)
        self.spin_cooldown.setSingleStep(0.5)
        self.spin_cooldown.setDecimals(1)
        self.spin_cooldown.setSuffix(" detik")
        self.spin_cooldown.setValue(self.cfg.continuous_cooldown_sec)
        form.addRow("Continuous Cooldown:", self.spin_cooldown)

        self.spin_coarse_step = QSpinBox()
        self.spin_coarse_step.setRange(5, 30)
        self.spin_coarse_step.setSingleStep(5)
        self.spin_coarse_step.setSuffix("°")
        self.spin_coarse_step.setValue(self.cfg.multi_angle_coarse_step)
        form.addRow("Multi-Angle Langkah Kasar:", self.spin_coarse_step)

        self.spin_fine_step = QSpinBox()
        self.spin_fine_step.setRange(1, 10)
        self.spin_fine_step.setSingleStep(1)
        self.spin_fine_step.setSuffix("°")
        self.spin_fine_step.setValue(self.cfg.multi_angle_fine_step)
        form.addRow("Multi-Angle Langkah Halus:", self.spin_fine_step)

        self.spin_retention = QSpinBox()
        self.spin_retention.setRange(0, 365)
        self.spin_retention.setSuffix(" hari (0=Nonaktif)")
        self.spin_retention.setValue(self.cfg.ng_retention_days)
        form.addRow("Auto-Cleanup Foto NG:", self.spin_retention)

        self.cmb_relay_logic = QComboBox()
        self.cmb_relay_logic.addItems(
            ["Active-High (LED / Relay High)", "Active-Low (Modul Relay 5V/24V Low)"]
        )
        self.cmb_relay_logic.setCurrentIndex(0 if self.cfg.gpio.active_high else 1)
        form.addRow("Polaritas Output GPIO:", self.cmb_relay_logic)
        layout.addWidget(timing_box)

        gpio_box = QGroupBox("Kontrol I/O GPIO & Status Pin")
        gpio_layout = QVBoxLayout(gpio_box)
        gpio_layout.setSpacing(8)

        lamp_row = QHBoxLayout()
        self.ind_ok_lamp = QLabel(f"GPIO {self.cfg.gpio.ok_led_pin}\\nOK HIJAU\\nOFF")
        self.ind_ng_lamp = QLabel(f"GPIO {self.cfg.gpio.ng_led_pin}\\nNG MERAH\\nOFF")
        self.ind_buzzer = QLabel(f"GPIO {self.cfg.gpio.buzzer_pin}\\nBUZZER\\nOFF")
        for ind in (self.ind_ok_lamp, self.ind_ng_lamp, self.ind_buzzer):
            ind.setAlignment(Qt.AlignCenter)
            ind.setMinimumHeight(64)
            ind.setStyleSheet(
                "background-color: #0E1420; border: 1px solid #2A3850; border-radius: 6px; "
                "font-family: 'Consolas', 'DejaVu Sans Mono'; font-size: 11px; color: #64748B;"
            )
            lamp_row.addWidget(ind)
        gpio_layout.addLayout(lamp_row)

        btn_row = QHBoxLayout()
        self.btn_mute_buzzer = QPushButton("Mute Buzzer: OFF")
        self.btn_mute_buzzer.setCheckable(True)
        self.btn_mute_buzzer.setChecked(self.cfg.mute_buzzer)
        self._update_mute_button_text()

        self.btn_test_gpio = QPushButton("Test GPIO")
        self.btn_test_gpio.setProperty("role", "primary")
        btn_row.addWidget(self.btn_mute_buzzer)
        btn_row.addWidget(self.btn_test_gpio)
        gpio_layout.addLayout(btn_row)

        self.btn_cleanup_now = QPushButton("Bersihkan Foto NG Lama Sekarang")
        gpio_layout.addWidget(self.btn_cleanup_now)
        layout.addWidget(gpio_box)

        # Panel Konsol Debug Perubahan Status Pin GPIO (khusus memudahkan debug di Windows)
        debug_box = QGroupBox("Log Perubahan Pin GPIO (Debug Monitor)")
        debug_layout = QVBoxLayout(debug_box)
        self.txt_gpio_console = QPlainTextEdit()
        self.txt_gpio_console.setReadOnly(True)
        self.txt_gpio_console.setMaximumBlockCount(150)
        self.txt_gpio_console.setMinimumHeight(130)
        self.txt_gpio_console.setStyleSheet(
            "font-family: 'Consolas', 'DejaVu Sans Mono'; font-size: 11px; color: #CBD5E1;"
        )
        self.txt_gpio_console.appendPlainText(self.gpio.status_summary)
        debug_layout.addWidget(self.txt_gpio_console)

        btn_clear_log = QPushButton("Bersihkan Log GPIO")
        btn_clear_log.clicked.connect(self.txt_gpio_console.clear)
        debug_layout.addWidget(btn_clear_log)
        layout.addWidget(debug_box)

        exp_box = QGroupBox("Laporan & Ekspor Data")
        exp_layout = QVBoxLayout(exp_box)
        self.btn_export_csv = QPushButton("Export Log Inspeksi ke CSV")
        self.btn_export_csv.setProperty("role", "success")
        self.btn_export_csv.setMinimumHeight(46)
        exp_layout.addWidget(self.btn_export_csv)
        layout.addWidget(exp_box)

        layout.addStretch(1)
        return panel

    def _build_log_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        hint = QLabel("50 inspeksi terakhir (klik baris NG untuk melihat foto bukti):")
        hint.setStyleSheet("color: #94A3B8; font-size: 12px;")
        layout.addWidget(hint)

        self.tbl_logs = QTableWidget(0, 5)
        self.tbl_logs.setHorizontalHeaderLabels(
            ["Waktu", "Template", "Skor", "Status", "Foto"]
        )
        self.tbl_logs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_logs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_logs.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_logs.verticalHeader().setVisible(False)
        header = self.tbl_logs.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        layout.addWidget(self.tbl_logs, 1)

        bottom_row = QHBoxLayout()
        self.btn_refresh_logs = QPushButton("Muat Ulang")
        self.btn_export_csv_tab = QPushButton("Export CSV")
        self.btn_export_csv_tab.setProperty("role", "success")
        bottom_row.addWidget(self.btn_refresh_logs)
        bottom_row.addWidget(self.btn_export_csv_tab)
        layout.addLayout(bottom_row)
        return panel

    def _connect_signals(self) -> None:
        self.btn_mode_run.clicked.connect(lambda: self._switch_mode("RUN"))
        self.btn_mode_teach.clicked.connect(lambda: self._switch_mode("TEACH"))
        self.btn_start_stop.clicked.connect(self._toggle_start_stop)

        self.cmb_op_mode.currentIndexChanged.connect(self._on_config_changed)
        self.cmb_method.currentIndexChanged.connect(self._on_config_changed)
        self.chk_auto_match.toggled.connect(self._on_config_changed)
        self.btn_manual_trigger.clicked.connect(self._on_manual_trigger_clicked)
        self.btn_reset_counter.clicked.connect(self._reset_daily_counter)
        self.list_templates.currentItemChanged.connect(self._on_template_selected)
        self.btn_delete_template.clicked.connect(self._on_delete_template)

        self.btn_freeze.clicked.connect(self._toggle_freeze_frame)
        self.btn_clear_roi.clicked.connect(self.video_canvas.clear_roi)
        self.video_canvas.roi_changed.connect(self._on_roi_changed)
        self.slider_threshold.valueChanged.connect(
            lambda v: self.spin_threshold.setValue(v / 100.0)
        )
        self.spin_threshold.valueChanged.connect(
            lambda v: self.slider_threshold.setValue(int(round(v * 100)))
        )
        self.btn_save_template.clicked.connect(self._on_save_template)

        self.spin_trigger_delay.valueChanged.connect(self._on_config_changed)
        self.spin_hold_time.valueChanged.connect(self._on_config_changed)
        self.spin_cooldown.valueChanged.connect(self._on_config_changed)
        self.spin_coarse_step.valueChanged.connect(self._on_config_changed)
        self.spin_fine_step.valueChanged.connect(self._on_config_changed)
        self.spin_retention.valueChanged.connect(self._on_config_changed)
        self.cmb_relay_logic.currentIndexChanged.connect(self._on_relay_logic_changed)
        self.btn_mute_buzzer.toggled.connect(self._on_mute_toggled)
        self.btn_test_gpio.clicked.connect(self.gpio.run_test_sequence)
        self.btn_cleanup_now.clicked.connect(self._on_cleanup_now)
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_export_csv_tab.clicked.connect(self._on_export_csv)

        self.btn_refresh_logs.clicked.connect(self._load_initial_stats_and_logs)
        self.tbl_logs.cellClicked.connect(self._on_log_row_clicked)

        self.camera_worker.frame_ready.connect(self._on_camera_frame)
        self.camera_worker.camera_status.connect(self._on_camera_status)
        self.camera_worker.camera_error.connect(self._on_camera_error)
        self.inspection_worker.inspection_finished.connect(
            self._on_inspection_finished
        )
        self.logger_worker.log_saved.connect(self._on_log_saved)
        self.logger_worker.logger_error.connect(
            lambda msg: self.statusBar().showMessage(msg, 6000)
        )

        self.gpio.trigger_received.connect(self._on_trigger_signal)
        self.gpio.sensor_state_changed.connect(self._on_sensor_state_changed)
        self.gpio.output_state_changed.connect(self._on_gpio_outputs_changed)
        self.gpio.standby_restored.connect(self._on_standby_restored)
        self.gpio.gpio_info.connect(self._on_gpio_info_updated)
        self.gpio.gpio_debug_log.connect(self.txt_gpio_console.appendPlainText)

    def _on_gpio_info_updated(self, text: str) -> None:
        self.lbl_gpio_badge.setText(self.gpio.mode_badge)
        self.statusBar().showMessage(text, 5000)

    def _switch_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        if mode == "TEACH":
            self._set_running(False)
            self._teach_frozen = False
            self._frozen_frame = None
            self.btn_freeze.setText("FREEZE / CAPTURE FRAME")
            self.video_canvas.set_teach_mode(True)
            self.video_canvas.set_match_overlay(None, "STANDBY")
        else:
            self._teach_frozen = False
            self._frozen_frame = None
            self.video_canvas.set_teach_mode(False)
            self.video_canvas.clear_roi()
        self._apply_mode_ui()

    def _apply_mode_ui(self) -> None:
        is_run = self._mode == "RUN"
        self.mode_stack.setCurrentIndex(0 if is_run else 1)
        self.right_tabs.setCurrentIndex(0)
        self.btn_mode_run.setProperty("role", "primary" if is_run else "")
        self.btn_mode_teach.setProperty("role", "warning" if not is_run else "")
        self.btn_start_stop.setEnabled(is_run)
        for btn in (self.btn_mode_run, self.btn_mode_teach):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _toggle_start_stop(self) -> None:
        if self._mode != "RUN":
            return
        if not self._is_running:
            if not self.template_manager.list_templates():
                QMessageBox.warning(
                    self,
                    "Template Kosong",
                    "Belum ada template referensi. Masuk ke MODE TEACH untuk merekam template terlebih dahulu.",
                )
                return
            self._set_running(True)
        else:
            self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        self._trigger_pending = False
        self._trigger_delay_timer.stop()

        is_trigger_mode = self.cmb_op_mode.currentIndex() == 0
        self.gpio.set_trigger_enabled(
            running and self._mode == "RUN" and is_trigger_mode
        )

        if running:
            self.btn_start_stop.setText("STOP INSPEKSI")
            self.btn_start_stop.setProperty("role", "danger")
            if is_trigger_mode:
                self._set_status_banner(
                    "STANDBY", "Trigger Mode Aktif — Menunggu Sensor / Spasi"
                )
                self.video_canvas.set_match_overlay(None, "STANDBY")
            else:
                self._set_status_banner(
                    "STANDBY", "Continuous Mode Aktif — Inspeksi berjalan"
                )
        else:
            self.btn_start_stop.setText("START INSPEKSI")
            self.btn_start_stop.setProperty("role", "success")
            self.gpio.all_off()
            self._set_status_banner(
                "STANDBY", "Inspeksi dihentikan — Semua output GPIO OFF"
            )
            self.video_canvas.set_match_overlay(None, "STANDBY")

        self.btn_start_stop.style().unpolish(self.btn_start_stop)
        self.btn_start_stop.style().polish(self.btn_start_stop)

    def _on_camera_frame(self, frame_bgr: np.ndarray, fps: float) -> None:
        self._last_fps = fps
        self.lbl_fps.setText(f"FPS: {fps:4.1f}")
        if self._mode == "TEACH" and self._teach_frozen:
            return
        self.video_canvas.update_frame(frame_bgr)

        if (
            self._mode == "RUN"
            and self._is_running
            and self.cmb_op_mode.currentIndex() == 1
        ):
            entries = self._get_active_templates()
            if entries:
                self.inspection_worker.submit_job(
                    InspectionJob(
                        frame_bgr=frame_bgr,
                        templates=entries,
                        method=self.cmb_method.currentText(),
                        is_trigger_mode=False,
                    )
                )

    def _on_camera_status(self, active: bool, info: str) -> None:
        self.lbl_camera_info.setText(f"Kamera: {info}")
        if not active:
            self.video_canvas.set_placeholder_text(info)

    def _on_camera_error(self, message: str) -> None:
        self.video_canvas.set_placeholder_text(message)
        self.statusBar().showMessage(message, 10000)

    def _on_manual_trigger_clicked(self) -> None:
        if self._mode != "RUN":
            return
        if not self._is_running:
            self.statusBar().showMessage(
                "Tekan START INSPEKSI terlebih dahulu sebelum memicu Trigger.", 4000
            )
            return
        if self.cmb_op_mode.currentIndex() != 0:
            self.statusBar().showMessage(
                "Manual Trigger hanya berlaku pada Trigger Mode.", 3500
            )
            return
        self.gpio.simulate_trigger_pulse()

    def _on_trigger_signal(self) -> None:
        if (
            self._mode != "RUN"
            or not self._is_running
            or self.cmb_op_mode.currentIndex() != 0
            or self._trigger_pending
        ):
            return
        delay_ms = self.spin_trigger_delay.value()
        self._trigger_pending = True
        if delay_ms > 0:
            self._set_status_banner("STANDBY", f"Settling delay {delay_ms} ms...")
            self.video_canvas.set_match_overlay(None, "SETTLING")
            self._trigger_delay_timer.start(delay_ms)
        else:
            self._execute_triggered_capture()

    def _execute_triggered_capture(self) -> None:
        self._trigger_pending = False
        if not self._is_running or self._mode != "RUN":
            return
        frame = self.camera_worker.get_latest_frame()
        if frame is None:
            self._set_status_banner("STANDBY", "Frame kamera belum tersedia")
            return
        entries = self._get_active_templates()
        if not entries:
            self._set_status_banner("STANDBY", "Pilih minimal satu template referensi")
            return
        self.inspection_worker.submit_job(
            InspectionJob(
                frame_bgr=frame,
                templates=entries,
                method=self.cmb_method.currentText(),
                is_trigger_mode=True,
            )
        )

    def _on_inspection_finished(
        self, result: MatchResult, frame_bgr: np.ndarray, is_trigger_mode: bool
    ) -> None:
        if self._mode != "RUN" or not self._is_running:
            return
        detail = (
            f"Template: {result.template_name} | Skor: {result.score * 100:.1f}% "
            f"(Min {result.threshold * 100:.0f}%) | Sudut: {result.angle_deg:.1f}° | "
            f"{result.elapsed_ms:.1f} ms"
        )
        self._set_status_banner(result.status, detail)
        self.video_canvas.set_match_overlay(result, result.status)

        if is_trigger_mode:
            self._increment_counter(result.matched)
            self.logger_worker.enqueue_inspection(result, frame_bgr)
            self.gpio.set_result_output(
                result.matched, hold_time_sec=self.spin_hold_time.value()
            )
        else:
            self.gpio.set_result_output(result.matched, hold_time_sec=None)
            now = time.monotonic()
            if (now - self._last_continuous_log_ts) >= self.spin_cooldown.value():
                self._last_continuous_log_ts = now
                self._increment_counter(result.matched)
                self.logger_worker.enqueue_inspection(result, frame_bgr)

    def _on_standby_restored(self) -> None:
        if (
            self._mode == "RUN"
            and self._is_running
            and self.cmb_op_mode.currentIndex() == 0
        ):
            self._set_status_banner(
                "STANDBY", "Siklus selesai — Siaga menunggu sinyal trigger berikutnya"
            )
            self.video_canvas.set_match_overlay(None, "STANDBY")

    def _set_status_banner(self, status: str, detail: str) -> None:
        styles = {
            "OK": "background-color: #065F46; border: 2px solid #10B981; border-radius: 8px; color: #ECFDF5;",
            "NG": "background-color: #7F1D1D; border: 2px solid #EF4444; border-radius: 8px; color: #FEF2F2;",
            "STANDBY": "background-color: #1E293B; border: 2px solid #475569; border-radius: 8px; color: #CBD5E1;",
        }
        labels = {"OK": "OK / PASS", "NG": "NG / FAIL", "STANDBY": "STANDBY"}
        self.status_banner.setText(labels.get(status, status))
        self.status_banner.setStyleSheet(styles.get(status, styles["STANDBY"]))
        self.lbl_result_detail.setText(detail)

    def _toggle_freeze_frame(self) -> None:
        if not self._teach_frozen:
            frame = self.camera_worker.get_latest_frame()
            if frame is None:
                QMessageBox.warning(
                    self,
                    "Kamera Belum Siap",
                    "Belum ada frame kamera untuk dibekukan.",
                )
                return
            self._frozen_frame = frame
            self._teach_frozen = True
            self.video_canvas.update_frame(frame)
            self.btn_freeze.setText("UNFREEZE (LIVE VIDEO)")
        else:
            self._teach_frozen = False
            self._frozen_frame = None
            self.btn_freeze.setText("FREEZE / CAPTURE FRAME")

    def _on_roi_changed(self, roi: Tuple[int, int, int, int]) -> None:
        x, y, w, h = roi
        if w > 0 and h > 0:
            self.lbl_roi_info.setText(f"ROI Aktif: x={x}, y={y}, w={w} px, h={h} px")
        else:
            self.lbl_roi_info.setText("ROI: Belum dipilih (drag mouse di layar video)")

    def _on_save_template(self) -> None:
        name = self.txt_tpl_name.text().strip()
        roi = self.video_canvas.get_roi()
        frame = (
            self._frozen_frame
            if self._teach_frozen and self._frozen_frame is not None
            else self.camera_worker.get_latest_frame()
        )
        if frame is None or not name or roi is None:
            QMessageBox.warning(
                self,
                "Data Belum Lengkap",
                "Pastikan kamera aktif, Nama Template terisi, dan kotak ROI telah dipilih.",
            )
            return

        try:
            meta = self.template_manager.save_template(
                name=name,
                frame_bgr=frame,
                roi=roi,
                threshold=self.spin_threshold.value(),
            )
            self.inspection_worker.clear_feature_cache(meta.name)
            self._refresh_template_list(select_name=meta.name)
            self.txt_tpl_name.clear()
            self.video_canvas.clear_roi()
            QMessageBox.information(
                self,
                "Template Tersimpan",
                f"Template '{meta.name}' berhasil disimpan ke:\\n{meta.image_path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Gagal Menyimpan Template", str(exc))

    def _refresh_template_list(self, select_name: Optional[str] = None) -> None:
        current = select_name or (
            self.list_templates.currentItem().data(Qt.UserRole)
            if self.list_templates.currentItem()
            else None
        )
        self.list_templates.clear()
        templates = self.template_manager.list_templates()
        target_row = 0
        for idx, meta in enumerate(templates):
            item = QListWidgetItem(
                f"{meta.name}  (Min {meta.threshold * 100:.0f}% | {meta.width}x{meta.height})"
            )
            item.setData(Qt.UserRole, meta.name)
            self.list_templates.addItem(item)
            if meta.name == current:
                target_row = idx

        if templates:
            self.list_templates.setCurrentRow(target_row)
        else:
            self.lbl_tpl_thumb.clear()
            self.lbl_tpl_thumb.setText("Kosong")
            self.lbl_tpl_meta.setText("-")

    def _on_template_selected(
        self, current: Optional[QListWidgetItem], _prev: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            self.lbl_tpl_thumb.clear()
            self.lbl_tpl_thumb.setText("Pratinjau")
            self.lbl_tpl_meta.setText("-")
            return
        name = current.data(Qt.UserRole)
        img = self.template_manager.get_template_image(name)
        meta = self.template_manager.templates.get(name)
        if img is not None and meta is not None:
            pix = QPixmap.fromImage(bgr_to_qimage(img))
            self.lbl_tpl_thumb.setPixmap(
                pix.scaled(116, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.lbl_tpl_meta.setText(
                f"{meta.width}x{meta.height} | {meta.threshold:.2f}"
            )

    def _on_delete_template(self) -> None:
        item = self.list_templates.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole)
        if (
            QMessageBox.question(
                self,
                "Hapus Template",
                f"Yakin ingin menghapus template '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
            )
            == QMessageBox.Yes
        ):
            self.template_manager.delete_template(name)
            self.inspection_worker.clear_feature_cache(name)
            self._refresh_template_list()

    def _get_active_templates(self):
        if self.chk_auto_match.isChecked():
            return self.template_manager.get_template_entries(match_all=True)
        item = self.list_templates.currentItem()
        if item is None:
            return []
        return self.template_manager.get_template_entries(
            selected_name=item.data(Qt.UserRole), match_all=False
        )

    def _increment_counter(self, is_ok: bool) -> None:
        if is_ok:
            self._daily_ok += 1
        else:
            self._daily_ng += 1
        self._update_counter_labels()

    def _reset_daily_counter(self) -> None:
        self._daily_ok = 0
        self._daily_ng = 0
        self._update_counter_labels()

    def _update_counter_labels(self) -> None:
        total = self._daily_ok + self._daily_ng
        yield_pct = (self._daily_ok / total * 100.0) if total > 0 else 0.0
        self.lbl_count_ok.setText(str(self._daily_ok))
        self.lbl_count_ng.setText(str(self._daily_ng))
        self.lbl_yield.setText(f"{yield_pct:.1f}%")

    def _on_config_changed(self) -> None:
        coarse_step = self.spin_coarse_step.value()
        fine_step = self.spin_fine_step.value()
        if fine_step > coarse_step:
            fine_step = coarse_step
            self.spin_fine_step.blockSignals(True)
            self.spin_fine_step.setValue(fine_step)
            self.spin_fine_step.blockSignals(False)

        self.cfg.operation_mode = (
            "trigger" if self.cmb_op_mode.currentIndex() == 0 else "continuous"
        )
        self.cfg.matching_method = self.cmb_method.currentText()
        self.cfg.match_all_templates = self.chk_auto_match.isChecked()
        self.cfg.trigger_delay_ms = self.spin_trigger_delay.value()
        self.cfg.output_hold_sec = self.spin_hold_time.value()
        self.cfg.continuous_cooldown_sec = self.spin_cooldown.value()
        self.cfg.multi_angle_coarse_step = coarse_step
        self.cfg.multi_angle_fine_step = fine_step
        self.cfg.ng_retention_days = self.spin_retention.value()
        self.inspection_worker.set_angle_steps(coarse_step, fine_step)
        self.template_manager.save_config()

        if self._is_running and self._mode == "RUN":
            is_trigger = self.cfg.operation_mode == "trigger"
            self.gpio.set_trigger_enabled(is_trigger)
            if is_trigger:
                self.gpio.all_off()
                self._set_status_banner(
                    "STANDBY", "Trigger Mode Aktif — Menunggu Sensor / Spasi"
                )

    def _on_relay_logic_changed(self, idx: int) -> None:
        active_high = idx == 0
        self.cfg.gpio.active_high = active_high
        self.gpio.reconfigure_active_high(active_high)
        self.template_manager.save_config()

    def _on_mute_toggled(self, muted: bool) -> None:
        self.cfg.mute_buzzer = muted
        self.gpio.set_mute_buzzer(muted)
        self.template_manager.save_config()
        self._update_mute_button_text()

    def _update_mute_button_text(self) -> None:
        muted = self.btn_mute_buzzer.isChecked()
        self.btn_mute_buzzer.setText("Mute Buzzer: ON" if muted else "Mute Buzzer: OFF")
        self.btn_mute_buzzer.setProperty("role", "warning" if muted else "")
        self.btn_mute_buzzer.style().unpolish(self.btn_mute_buzzer)
        self.btn_mute_buzzer.style().polish(self.btn_mute_buzzer)

    def _on_sensor_state_changed(self, active: bool) -> None:
        pin = self.cfg.gpio.trigger_pin
        if active:
            self.lbl_sensor_state.setText(f"Sensor GPIO {pin}: TRIGGERED (LOW)")
            self.lbl_sensor_state.setStyleSheet(
                "font-family: 'Consolas', 'DejaVu Sans Mono'; font-weight: 700; color: #F59E0B;"
            )
        else:
            self.lbl_sensor_state.setText(f"Sensor GPIO {pin}: IDLE (HIGH)")
            self.lbl_sensor_state.setStyleSheet(
                "font-family: 'Consolas', 'DejaVu Sans Mono'; color: #94A3B8;"
            )

    def _on_gpio_outputs_changed(
        self, ok_on: bool, ng_on: bool, buzzer_on: bool
    ) -> None:
        off_style = (
            "background-color: #0E1420; border: 1px solid #2A3850; border-radius: 6px; "
            "font-family: 'Consolas', 'DejaVu Sans Mono'; font-size: 11px; color: #64748B;"
        )
        self.ind_ok_lamp.setText(
            f"GPIO {self.cfg.gpio.ok_led_pin}\\nOK HIJAU\\n{'ON' if ok_on else 'OFF'}"
        )
        self.ind_ok_lamp.setStyleSheet(
            "background-color: #065F46; border: 1px solid #10B981; border-radius: 6px; "
            "font-family: 'Consolas', 'DejaVu Sans Mono'; font-size: 11px; font-weight: 700; color: #ECFDF5;"
            if ok_on
            else off_style
        )
        self.ind_ng_lamp.setText(
            f"GPIO {self.cfg.gpio.ng_led_pin}\\nNG MERAH\\n{'ON' if ng_on else 'OFF'}"
        )
        self.ind_ng_lamp.setStyleSheet(
            "background-color: #7F1D1D; border: 1px solid #EF4444; border-radius: 6px; "
            "font-family: 'Consolas', 'DejaVu Sans Mono'; font-size: 11px; font-weight: 700; color: #FEF2F2;"
            if ng_on
            else off_style
        )
        self.ind_buzzer.setText(
            f"GPIO {self.cfg.gpio.buzzer_pin}\\nBUZZER\\n{'ON' if buzzer_on else 'OFF'}"
        )
        self.ind_buzzer.setStyleSheet(
            "background-color: #78350F; border: 1px solid #F59E0B; border-radius: 6px; "
            "font-family: 'Consolas', 'DejaVu Sans Mono'; font-size: 11px; font-weight: 700; color: #FEF3C7;"
            if buzzer_on
            else off_style
        )

    def _load_initial_stats_and_logs(self) -> None:
        self._daily_ok, self._daily_ng = self.logger_worker.get_today_stats()
        self._update_counter_labels()
        self._recent_logs = self.logger_worker.get_recent_logs(limit=50)
        self._render_log_table()

    def _on_log_saved(self, entry: LogEntry) -> None:
        self._recent_logs.insert(0, entry)
        self._recent_logs = self._recent_logs[:50]
        self._render_log_table()

    def _render_log_table(self) -> None:
        self.tbl_logs.setRowCount(len(self._recent_logs))
        for row, entry in enumerate(self._recent_logs):
            time_short = (
                entry.timestamp.split(" ")[-1]
                if " " in entry.timestamp
                else entry.timestamp
            )
            items = [
                QTableWidgetItem(time_short),
                QTableWidgetItem(entry.template_name),
                QTableWidgetItem(f"{entry.score * 100:.1f}%"),
                QTableWidgetItem(entry.status),
                QTableWidgetItem("Lihat" if entry.image_path else "-"),
            ]
            for col, item in enumerate(items):
                item.setTextAlignment(
                    Qt.AlignCenter if col != 1 else Qt.AlignVCenter | Qt.AlignLeft
                )
                if col == 3:
                    item.setForeground(
                        QColor("#10B981") if entry.status == "OK" else QColor("#EF4444")
                    )
                if col == 4 and entry.image_path:
                    item.setForeground(QColor("#38BDF8"))
                self.tbl_logs.setItem(row, col, item)

    def _on_log_row_clicked(self, row: int, _col: int) -> None:
        if row < 0 or row >= len(self._recent_logs):
            return
        entry = self._recent_logs[row]
        if not entry.image_path:
            return
        img_abs = self.logger_worker.resolve_image_path(entry.image_path)
        NGImagePreviewDialog(entry, img_abs, self).exec()

    def _on_export_csv(self) -> None:
        dlg = ExportCSVDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        start_d, end_d = dlg.get_date_range()
        try:
            out_path, count = self.logger_worker.export_to_csv(start_d, end_d)
            QMessageBox.information(
                self,
                "Export CSV Berhasil",
                f"Berhasil mengekspor {count} baris log inspeksi ke:\\n{out_path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Gagal Export CSV", str(exc))

    def _on_cleanup_now(self) -> None:
        days = self.spin_retention.value()
        if days <= 0:
            QMessageBox.information(
                self,
                "Auto-Cleanup Nonaktif",
                "Atur retensi > 0 hari terlebih dahulu untuk membersihkan foto NG lama.",
            )
            return
        removed = self.logger_worker.cleanup_old_ng_captures(days)
        QMessageBox.information(
            self,
            "Pembersihan Selesai",
            f"{removed} folder foto NG lebih lama dari {days} hari telah dihapus.",
        )

    def closeEvent(self, event) -> None:
        """Pembersihan aman semua thread dan pin GPIO saat aplikasi ditutup."""
        self._is_running = False
        self._trigger_delay_timer.stop()
        self.template_manager.save_config()

        self.gpio.cleanup()
        self.camera_worker.stop()
        self.inspection_worker.stop()
        self.logger_worker.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_INDUSTRIAL_QSS)
    window = VisionMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
