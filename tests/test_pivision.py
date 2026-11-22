"""
Unit test komprehensif untuk seluruh subsistem PiVision RPi5.
Jalankan secara offline dengan:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

import gpio_controller  # noqa: E402
from camera_worker import USBCameraSource, create_camera_source  # noqa: E402
from data_logger import DataLoggerWorker  # noqa: E402
from gpio_controller import (  # noqa: E402
    HardwareGPIOController,
    MockGPIOController,
    create_gpio_controller,
)
from main import VideoCanvasWidget, VisionMainWindow  # noqa: E402
from matcher import (  # noqa: E402
    METHOD_AKAZE_AFFINE,
    METHOD_MULTI_ANGLE,
    METHOD_TEMPLATE_FIXED,
    InspectionJob,
    InspectionWorker,
    MatchResult,
    TemplateEntry,
    TemplateMatcherEngine,
)
from template_manager import GPIOConfig, TemplateManager  # noqa: E402


_QAPP = QApplication.instance() or QApplication(sys.argv)


class TestTemplateManager(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)
        self.tm = TemplateManager(self.base_dir)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_sanitize_name_strips_backslashes_and_traversal(self) -> None:
        cleaned = self.tm.sanitize_name(r"..\..\evil\Part #1!")
        self.assertNotIn("\\", cleaned)
        self.assertNotIn("/", cleaned)
        self.assertNotIn("..", cleaned)
        self.assertEqual(cleaned, "evil_Part_1")

    def test_resolve_relative_path_normalizes_and_blocks_traversal(self) -> None:
        resolved = self.tm.resolve_relative_path(r"templates\PartA\ref.png")
        expected = (self.base_dir / "templates" / "PartA" / "ref.png").resolve()
        self.assertEqual(resolved, expected)

        with self.assertRaises(ValueError):
            self.tm.resolve_relative_path("../../etc/passwd")

        with self.assertRaises(ValueError):
            self.tm.resolve_relative_path(r"C:\Windows\System32\cmd.exe")

    def test_folder_collision_part_space_vs_underscore(self) -> None:
        img1 = np.full((80, 80, 3), 50, dtype=np.uint8)
        img2 = np.full((80, 80, 3), 200, dtype=np.uint8)

        meta1 = self.tm.save_template("Part A", img1, (10, 10, 40, 40), 0.80)
        meta2 = self.tm.save_template("Part_A", img2, (10, 10, 40, 40), 0.85)

        self.assertNotEqual(meta1.folder, meta2.folder)
        path1 = self.tm.resolve_relative_path(meta1.image_path)
        path2 = self.tm.resolve_relative_path(meta2.image_path)
        self.assertTrue(path1.exists())
        self.assertTrue(path2.exists())

        self.assertTrue(self.tm.delete_template("Part A"))
        self.assertFalse(path1.exists())
        self.assertTrue(path2.exists(), "Deleting 'Part A' must not delete 'Part_A'")

    def test_update_template_threshold_persists_to_config(self) -> None:
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        self.tm.save_template("PCB_1", img, (10, 10, 40, 40), 0.80)
        self.tm.update_template_threshold("PCB_1", 0.93)

        tm_reloaded = TemplateManager(self.base_dir)
        self.assertIn("PCB_1", tm_reloaded.templates)
        self.assertAlmostEqual(tm_reloaded.templates["PCB_1"].threshold, 0.93, places=3)


class TestMatcherAndWorker(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = TemplateMatcherEngine()

    def test_fixed_matching_and_oversized_template(self) -> None:
        frame = np.full((300, 400, 3), 30, dtype=np.uint8)
        cv2.rectangle(frame, (120, 90), (200, 170), (220, 220, 220), -1)
        cv2.circle(frame, (160, 130), 20, (0, 0, 255), -1)
        tpl = frame[90:170, 120:200].copy()

        res = self.engine.match_single(
            frame, TemplateEntry("IC", 0.80, tpl), METHOD_TEMPLATE_FIXED
        )
        self.assertTrue(res.matched)
        self.assertEqual(res.status, "OK")
        self.assertGreaterEqual(res.score, 0.95)
        self.assertIsNotNone(res.bbox)
        self.assertLessEqual(abs(res.bbox[0] - 120), 2)
        self.assertLessEqual(abs(res.bbox[1] - 90), 2)

        oversized = TemplateEntry("Big", 0.8, np.zeros((500, 500, 3), dtype=np.uint8))
        res_big = self.engine.match_single(frame, oversized, METHOD_TEMPLATE_FIXED)
        self.assertFalse(res_big.matched)
        self.assertEqual(res_big.status, "NG")

    def test_multi_angle_45_deg_asymmetric_l_shape(self) -> None:
        tpl = np.full((60, 100, 3), 30, dtype=np.uint8)
        cv2.rectangle(tpl, (10, 10), (90, 25), (255, 255, 255), -1)
        cv2.rectangle(tpl, (10, 10), (25, 50), (0, 255, 0), -1)

        frame = np.full((400, 400, 3), 30, dtype=np.uint8)
        frame[170:230, 150:250] = tpl
        rot_mat = cv2.getRotationMatrix2D((200, 200), 45.0, 1.0)
        rot_frame = cv2.warpAffine(frame, rot_mat, (400, 400), borderValue=(30, 30, 30))

        res = self.engine.match_single(
            rot_frame, TemplateEntry("L_Shape", 0.70, tpl), METHOD_MULTI_ANGLE
        )
        self.assertTrue(res.matched)
        self.assertGreater(res.score, 0.85)
        self.assertIsNotNone(res.polygon)

        poly_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [np.array(res.polygon, dtype=np.int32)], 255)
        obj_mask = (rot_frame[:, :, 0] > 150) | (rot_frame[:, :, 1] > 150)
        total_obj = int(obj_mask.sum())
        inside = int(np.logical_and(obj_mask, poly_mask > 0).sum())
        self.assertGreater(total_obj, 0)
        self.assertGreaterEqual(inside / total_obj, 0.95)

    def test_akaze_blank_and_out_of_frame_rejection(self) -> None:
        blank_frame = np.zeros((240, 240, 3), dtype=np.uint8)
        blank_tpl = TemplateEntry("Blank", 0.70, np.zeros((50, 50, 3), dtype=np.uint8))
        res_blank = self.engine.match_single(blank_frame, blank_tpl, METHOD_AKAZE_AFFINE)
        self.assertFalse(res_blank.matched)
        self.assertEqual(res_blank.status, "NG")

        rng = np.random.default_rng(42)
        textured = rng.integers(30, 230, size=(300, 300, 3), dtype=np.uint8)
        for i in range(12):
            cv2.circle(textured, (40 + i * 20, 40 + (i % 5) * 45), 14, (255, 255, 0), 2)
        tpl = textured[60:180, 60:180].copy()

        off_frame_mat = np.array([[1.0, 0.0, 5000.0], [0.0, 1.0, 5000.0]], dtype=np.float64)
        fake_mask = np.ones((20, 1), dtype=np.uint8)
        fake_matches = [[cv2.DMatch(i, i, 5.0), cv2.DMatch(i, i, 50.0)] for i in range(20)]
        fake_kps = [cv2.KeyPoint(float(10 + i * 4), float(10 + i * 4), 5.0) for i in range(25)]
        fake_des = np.ones((25, 32), dtype=np.uint8)

        fake_akaze = MagicMock()
        fake_akaze.detectAndCompute.return_value = (fake_kps, fake_des)
        fake_bf = MagicMock()
        fake_bf.knnMatch.return_value = fake_matches

        with (
            patch.object(self.engine, "_akaze", fake_akaze),
            patch.object(self.engine, "_bf", fake_bf),
            patch("matcher.cv2.estimateAffinePartial2D", return_value=(off_frame_mat, fake_mask)),
        ):
            self.engine.clear_cache()
            res_off = self.engine.match_single(
                textured, TemplateEntry("OffFrame", 0.50, tpl), METHOD_AKAZE_AFFINE
            )
        self.assertFalse(res_off.matched)
        self.assertIsNone(res_off.bbox)
        self.assertIsNone(res_off.polygon)

    def test_inspection_worker_submit_and_stop(self) -> None:
        worker = InspectionWorker()
        results = []
        worker.inspection_finished.connect(lambda r, _f, _t: results.append(r))
        worker.start()

        frame = np.full((120, 120, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (30, 30), (70, 70), (240, 240, 240), -1)
        tpl = TemplateEntry("Box", 0.80, frame[30:70, 30:70].copy())

        # Tunggu sejenak agar thread masuk run()
        for _ in range(20):
            if worker._running:
                break
            _QAPP.processEvents()
            import time
            time.sleep(0.02)

        self.assertTrue(
            worker.submit_job(InspectionJob(frame, [tpl], METHOD_TEMPLATE_FIXED, False))
        )
        self.assertTrue(
            worker.submit_job(InspectionJob(frame, [tpl], METHOD_TEMPLATE_FIXED, True))
        )

        import time
        deadline = time.perf_counter() + 3.0
        while len(results) < 2 and time.perf_counter() < deadline:
            _QAPP.processEvents()
            time.sleep(0.02)

        worker.stop()
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.matched for r in results))


class TestDataLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)
        self.logger = DataLoggerWorker(base_dir=self.base_dir, retention_days=30)

    def tearDown(self) -> None:
        if self.logger.isRunning():
            self.logger.stop()
        self.tmp_dir.cleanup()

    def test_logging_stats_csv_and_path_resolution(self) -> None:
        self.logger.start()
        frame = np.full((120, 160, 3), 80, dtype=np.uint8)

        ok_res = MatchResult(
            True, "OK", "PartA", METHOD_TEMPLATE_FIXED, 0.97, 0.80, (10, 10, 30, 30), None, 5.0
        )
        ng_res = MatchResult(
            False, "NG", "PartA", METHOD_TEMPLATE_FIXED, 0.42, 0.80, (10, 10, 30, 30), None, 6.0
        )
        self.logger.enqueue_inspection(ok_res, frame)
        self.logger.enqueue_inspection(ng_res, frame)
        self.logger.stop()

        ok_cnt, ng_cnt = self.logger.get_today_stats()
        self.assertEqual(ok_cnt, 1)
        self.assertEqual(ng_cnt, 1)

        recent = self.logger.get_recent_logs(10)
        self.assertEqual(len(recent), 2)
        ng_entries = [e for e in recent if e.status == "NG"]
        self.assertEqual(len(ng_entries), 1)
        self.assertTrue(self.logger.resolve_image_path(ng_entries[0].image_path).exists())

        resolved_win = self.logger.resolve_image_path(r"ng_captures\2026-11-22\test.jpg")
        self.assertEqual(
            resolved_win,
            (self.base_dir / "ng_captures" / "2026-11-22" / "test.jpg").resolve(),
        )
        with self.assertRaises(ValueError):
            self.logger.resolve_image_path("../../secret.jpg")

        today = datetime.now().strftime("%Y-%m-%d")
        csv_path, count = self.logger.export_to_csv(today, today)
        self.assertTrue(csv_path.exists())
        self.assertEqual(count, 2)
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 3)

        with self.assertRaises(ValueError):
            self.logger.export_to_csv("2026-12-31", "2026-01-01")

    def test_cleanup_preserves_gitkeep(self) -> None:
        gitkeep = self.logger.ng_dir / ".gitkeep"
        gitkeep.touch()
        old_date = (datetime.now().date() - timedelta(days=45)).strftime("%Y-%m-%d")
        old_dir = self.logger.ng_dir / old_date
        old_dir.mkdir(parents=True, exist_ok=True)
        (old_dir / "old.jpg").write_bytes(b"123")

        removed = self.logger.cleanup_old_ng_captures(30)
        self.assertEqual(removed, 1)
        self.assertFalse(old_dir.exists())
        self.assertTrue(gitkeep.exists())


class TestGPIOAndCamera(unittest.TestCase):
    def test_mock_gpio_lifecycle_and_cleanup(self) -> None:
        cfg = GPIOConfig(trigger_pin=23, ok_led_pin=17, ng_led_pin=27, buzzer_pin=22)
        ctrl = create_gpio_controller(cfg)
        self.assertIsInstance(ctrl, MockGPIOController)

        received = []
        ctrl.trigger_received.connect(lambda: received.append(True))
        ctrl.set_trigger_enabled(True)
        ctrl.simulate_trigger_pulse()
        self.assertEqual(len(received), 1)
        self.assertTrue(ctrl._sim_release_timer.isActive())

        ctrl.set_result_output(False, hold_time_sec=0.5)
        self.assertTrue(ctrl._ng_state)
        self.assertTrue(ctrl._buzzer_state)

        ctrl.set_mute_buzzer(True)
        self.assertFalse(ctrl._buzzer_state)

        ctrl.cleanup()
        self.assertFalse(ctrl._sim_release_timer.isActive())
        self.assertFalse(ctrl._ok_state)
        self.assertFalse(ctrl._ng_state)

    def test_hardware_gpio_reconfigure_does_not_recreate_lgpio_factory(self) -> None:
        fake_device = MagicMock()
        fake_device.pin_factory = None
        fake_factory_cls = MagicMock(return_value=MagicMock())
        fake_button_cls = MagicMock(return_value=MagicMock())
        fake_output_cls = MagicMock(return_value=MagicMock())

        with (
            patch.object(gpio_controller, "_GPIOZERO_AVAILABLE", True),
            patch.object(gpio_controller, "Device", fake_device),
            patch.object(gpio_controller, "LGPIOFactory", fake_factory_cls),
            patch.object(gpio_controller, "Button", fake_button_cls),
            patch.object(gpio_controller, "OutputDevice", fake_output_cls),
        ):
            hw = HardwareGPIOController(GPIOConfig())
            hw.reconfigure_active_high(False)
            hw.cleanup()

        fake_factory_cls.assert_called_once()

    def test_camera_source_factory(self) -> None:
        src = create_camera_source("usb", device_index=99)
        self.assertIsInstance(src, USBCameraSource)


class TestMainWindowGUI(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg_path = ROOT_DIR / "config.json"
        self.db_path = ROOT_DIR / "data" / "inspection_logs.db"
        self.cfg_existed = self.cfg_path.exists()
        self.db_existed = self.db_path.exists()

        # Hindari blokir open webcam fisik selama unit test headless
        self.cam_open_patch = patch.object(
            USBCameraSource, "open", return_value=(False, "Simulasi Test Tanpa Kamera")
        )
        self.cam_open_patch.start()
        self.win = VisionMainWindow()

    def tearDown(self) -> None:
        self.win.close()
        _QAPP.processEvents()
        self.cam_open_patch.stop()
        if not self.cfg_existed and self.cfg_path.exists():
            self.cfg_path.unlink(missing_ok=True)
        if not self.db_existed and self.db_path.exists():
            self.db_path.unlink(missing_ok=True)

    def test_no_double_escaped_newlines_in_ui(self) -> None:
        for lbl in self.win.findChildren(QLabel):
            self.assertNotIn("\\n", lbl.text())

        self.win._on_gpio_outputs_changed(True, False, True)
        self.assertNotIn("\\n", self.win.ind_ok_lamp.text())
        self.assertNotIn("\\n", self.win.ind_ng_lamp.text())
        self.assertNotIn("\\n", self.win.ind_buzzer.text())
        self.assertIn("\n", self.win.ind_ok_lamp.text())

    def test_space_shortcut_disabled_in_teach_mode(self) -> None:
        self.assertTrue(self.win._space_shortcut.isEnabled())
        self.win._switch_mode("TEACH")
        self.assertFalse(self.win._space_shortcut.isEnabled())
        self.win._switch_mode("RUN")
        self.assertTrue(self.win._space_shortcut.isEnabled())

    def test_reset_counter_not_overwritten_by_refresh_logs(self) -> None:
        self.win._increment_counter(True)
        self.win._increment_counter(False)
        self.assertEqual(self.win._daily_ok, 1)
        self.assertEqual(self.win._daily_ng, 1)

        self.win._reset_daily_counter()
        self.assertEqual(self.win._daily_ok, 0)
        self.assertEqual(self.win._daily_ng, 0)

        self.win._refresh_log_table()
        self.assertEqual(self.win._daily_ok, 0)
        self.assertEqual(self.win._daily_ng, 0)

    def test_video_canvas_roi_and_overlays(self) -> None:
        canvas = VideoCanvasWidget()
        canvas.resize(640, 480)
        frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
        canvas.update_frame(frame)
        canvas.set_teach_mode(True)
        canvas._drag_start_widget = QPoint(80, 80)
        canvas._drag_end_widget = QPoint(320, 320)
        canvas._update_roi_from_drag()
        roi = canvas.get_roi()
        self.assertIsNotNone(roi)
        self.assertGreaterEqual(roi[2], 16)
        self.assertGreaterEqual(roi[3], 16)

        canvas.set_teach_mode(False)
        res_poly = MatchResult(
            True,
            "OK",
            "PartA",
            METHOD_MULTI_ANGLE,
            0.91,
            0.80,
            (50, 50, 100, 100),
            [(50, 50), (150, 60), (140, 160), (40, 150)],
            12.0,
            15.0,
        )
        canvas.set_match_overlay(res_poly, "OK")
        canvas.repaint()

        res_bbox = MatchResult(
            False,
            "NG",
            "PartB",
            METHOD_TEMPLATE_FIXED,
            0.55,
            0.80,
            (60, 60, 80, 80),
            None,
            8.0,
            0.0,
        )
        canvas.set_match_overlay(res_bbox, "NG")
        canvas.repaint()

    def test_reconnect_camera_switches_source_safely(self) -> None:
        idx = self.win.cmb_camera_source.findData("usb:1")
        self.assertGreaterEqual(idx, 0)
        self.win.cmb_camera_source.setCurrentIndex(idx)
        self.win._on_reconnect_camera()
        _QAPP.processEvents()
        self.assertEqual(self.win.cfg.camera_backend, "usb")
        self.assertEqual(self.win.cfg.camera_index, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
