"""
gpio_controller.py
------------------
Kontrol GPIO Input & Output lintas OS:
- Raspberry Pi 5 (Linux ARM64): menggunakan gpiozero + LGPIOFactory (chip RP1).
- Windows 10/11 atau Linux non-RPi: otomatis masuk ke MockGPIOController tanpa crash,
  menampilkan badge [SIMULASI GPIO - WINDOWS] di GUI, dan mencetak setiap perubahan
  status pin (Trigger, Lampu Hijau, Lampu Merah, Buzzer) ke konsol & panel log GUI.
"""

from __future__ import annotations

import platform
from datetime import datetime
from typing import List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from template_manager import GPIOConfig

try:
    from gpiozero import Button, Device, OutputDevice  # type: ignore
    from gpiozero.pins.lgpio import LGPIOFactory  # type: ignore

    _GPIOZERO_AVAILABLE = True
except Exception:
    Button = None  # type: ignore
    Device = None  # type: ignore
    OutputDevice = None  # type: ignore
    LGPIOFactory = None  # type: ignore
    _GPIOZERO_AVAILABLE = False


class BaseGPIOController(QObject):
    """Kelas dasar bersama untuk HardwareGPIOController (RPi 5) dan MockGPIOController (Windows/PC)."""

    trigger_received = Signal()
    sensor_state_changed = Signal(bool)
    output_state_changed = Signal(bool, bool, bool)  # (ok_on, ng_on, buzzer_on)
    standby_restored = Signal()
    gpio_info = Signal(str)
    gpio_debug_log = Signal(str)

    def __init__(self, config: GPIOConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.simulation_mode: bool = True
        self.mode_badge: str = f"[SIMULASI GPIO - {platform.system().upper()}]"

        self._trigger_enabled: bool = False
        self._mute_buzzer: bool = False
        self._ok_state: bool = False
        self._ng_state: bool = False
        self._buzzer_state: bool = False
        self._sensor_active: bool = False

        # Timer non-blocking untuk Output Hold Time pada Trigger Mode
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._on_hold_timeout)

        # Timer non-blocking untuk sekuens Test GPIO
        self._test_timers: List[QTimer] = []

    def _log_pin_event(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{ts}] {self.mode_badge} {message}"
        print(formatted, flush=True)
        self.gpio_debug_log.emit(formatted)

    def _apply_hardware_outputs(
        self, ok_on: bool, ng_on: bool, buzzer_on: bool
    ) -> None:
        """Di-override oleh subclass untuk mengendalikan pin fisik atau mencetak log simulasi."""
        raise NotImplementedError

    def reconfigure_active_high(self, active_high: bool) -> None:
        self.config.active_high = active_high

    def set_trigger_enabled(self, enabled: bool) -> None:
        self._trigger_enabled = enabled
        self._log_pin_event(
            f"Trigger Input (BCM {self.config.trigger_pin}): {'ENABLED' if enabled else 'DISABLED'}"
        )
        if not enabled:
            self.all_off()

    def set_mute_buzzer(self, muted: bool) -> None:
        self._mute_buzzer = muted
        self._log_pin_event(
            f"Mute Buzzer (BCM {self.config.buzzer_pin}): {'ON (Muted)' if muted else 'OFF'}"
        )
        if muted and self._buzzer_state:
            self._set_output_states(self._ok_state, self._ng_state, False)

    def simulate_trigger_pulse(self) -> None:
        """Memicu sinyal trigger dari tombol software/Spasi untuk pengujian & debug."""
        self._sensor_active = True
        self.sensor_state_changed.emit(True)
        self._log_pin_event(
            f"PIN TRIGGER (BCM {self.config.trigger_pin}): PRESSED -> LOW "
            f"({'DITERIMA' if self._trigger_enabled else 'DIABAIKAN - TIDAK RUNNING'})"
        )
        if self._trigger_enabled:
            self.trigger_received.emit()

        def _release() -> None:
            self._sensor_active = False
            self.sensor_state_changed.emit(False)
            self._log_pin_event(
                f"PIN TRIGGER (BCM {self.config.trigger_pin}): RELEASED -> HIGH (IDLE)"
            )

        QTimer.singleShot(180, _release)

    def _set_output_states(
        self, ok_on: bool, ng_on: bool, buzzer_on: bool
    ) -> None:
        changed = (
            ok_on != self._ok_state
            or ng_on != self._ng_state
            or buzzer_on != self._buzzer_state
        )
        self._ok_state = ok_on
        self._ng_state = ng_on
        self._buzzer_state = buzzer_on

        self._apply_hardware_outputs(ok_on, ng_on, buzzer_on)
        self.output_state_changed.emit(ok_on, ng_on, buzzer_on)

        if changed:
            polarity = "ACTIVE-HIGH" if self.config.active_high else "ACTIVE-LOW"
            self._log_pin_event(
                f"OUTPUT ({polarity}) | Hijau(BCM {self.config.ok_led_pin})={'ON' if ok_on else 'OFF'} | "
                f"Merah(BCM {self.config.ng_led_pin})={'ON' if ng_on else 'OFF'} | "
                f"Buzzer(BCM {self.config.buzzer_pin})={'ON' if buzzer_on else 'OFF'}"
            )

    def set_result_output(
        self, is_ok: bool, hold_time_sec: Optional[float] = None
    ) -> None:
        self._cancel_test_timers()
        self._hold_timer.stop()

        if is_ok:
            self._set_output_states(True, False, False)
        else:
            self._set_output_states(False, True, not self._mute_buzzer)

        if hold_time_sec is not None and hold_time_sec > 0:
            self._hold_timer.start(int(hold_time_sec * 1000))

    def _on_hold_timeout(self) -> None:
        self._log_pin_event("Output Hold Time selesai -> kembali ke STANDBY (semua pin OFF)")
        self.all_off()
        self.standby_restored.emit()

    def all_off(self) -> None:
        self._cancel_test_timers()
        self._hold_timer.stop()
        self._set_output_states(False, False, False)

    def run_test_sequence(self) -> None:
        """Pengujian non-blocking berurutan: Hijau (0.7s) -> Merah (0.7s) -> Buzzer (0.5s) -> OFF."""
        self.all_off()
        self._log_pin_event("Memulai sekuens Test GPIO (Hijau -> Merah -> Buzzer -> OFF)...")

        schedule = [
            (0, lambda: self._set_output_states(True, False, False)),
            (700, lambda: self._set_output_states(False, True, False)),
            (1400, lambda: self._set_output_states(False, False, not self._mute_buzzer)),
            (1900, self.all_off),
        ]
        for delay_ms, fn in schedule:
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(fn)
            t.start(delay_ms)
            self._test_timers.append(t)

    def _cancel_test_timers(self) -> None:
        for timer in self._test_timers:
            timer.stop()
            timer.deleteLater()
        self._test_timers.clear()

    def cleanup(self) -> None:
        self._trigger_enabled = False
        self.all_off()
        self._log_pin_event("Cleanup GPIO selesai — semua pin OFF dan dilepas aman.")


class MockGPIOController(BaseGPIOController):
    """
    Pengendali GPIO simulasi aman untuk debugging di Windows 10/11 atau PC Linux/macOS
    tanpa memerlukan library gpiozero maupun lgpio.
    """

    def __init__(
        self,
        config: GPIOConfig,
        reason: str = "",
        parent=None,
    ) -> None:
        super().__init__(config, parent)
        self.simulation_mode = True
        os_tag = platform.system().upper() or "PC"
        self.mode_badge = f"[SIMULASI GPIO - {os_tag}]"

        relay_mode = "Active-High" if self.config.active_high else "Active-Low"
        info = f"{self.mode_badge} Polaritas: {relay_mode}"
        if reason:
            info = f"{info} ({reason})"
        self._initial_info = info
        self._log_pin_event(
            f"MockGPIOController aktif. Trigger=BCM {self.config.trigger_pin}, "
            f"Hijau=BCM {self.config.ok_led_pin}, Merah=BCM {self.config.ng_led_pin}, "
            f"Buzzer=BCM {self.config.buzzer_pin}, {relay_mode}."
        )

    @property
    def status_summary(self) -> str:
        return self._initial_info

    def _apply_hardware_outputs(
        self, ok_on: bool, ng_on: bool, buzzer_on: bool
    ) -> None:
        # Tidak ada pin fisik pada mode simulasi; log sudah dicetak oleh BaseGPIOController.
        return

    def reconfigure_active_high(self, active_high: bool) -> None:
        if self.config.active_high == active_high:
            return
        self.all_off()
        self.config.active_high = active_high
        relay_mode = "Active-High" if active_high else "Active-Low"
        self._initial_info = f"{self.mode_badge} Polaritas: {relay_mode}"
        self.gpio_info.emit(self._initial_info)
        self._log_pin_event(f"Polaritas output simulasi diubah ke {relay_mode}")


class HardwareGPIOController(BaseGPIOController):
    """Pengendali GPIO fisik khusus Raspberry Pi 5 (chip RP1) memakai gpiozero + LGPIOFactory."""

    def __init__(self, config: GPIOConfig, parent=None) -> None:
        super().__init__(config, parent)
        self.simulation_mode = False
        self.mode_badge = "[GPIO AKTIF - RPI5 LGPIO]"
        self._button = None
        self._ok_out = None
        self._ng_out = None
        self._buzzer_out = None
        self._initial_info = ""
        self._initialize_hardware()

    @property
    def status_summary(self) -> str:
        return self._initial_info

    def _initialize_hardware(self) -> None:
        if not _GPIOZERO_AVAILABLE or LGPIOFactory is None:
            raise RuntimeError("gpiozero/lgpio tidak terpasang pada sistem ini.")

        Device.pin_factory = LGPIOFactory()

        self._button = Button(
            self.config.trigger_pin,
            pull_up=True,
            bounce_time=self.config.bounce_time_sec,
        )
        self._button.when_pressed = self._handle_physical_trigger_pressed
        self._button.when_released = self._handle_physical_trigger_released

        self._ok_out = OutputDevice(
            self.config.ok_led_pin,
            active_high=self.config.active_high,
            initial_value=False,
        )
        self._ng_out = OutputDevice(
            self.config.ng_led_pin,
            active_high=self.config.active_high,
            initial_value=False,
        )
        self._buzzer_out = OutputDevice(
            self.config.buzzer_pin,
            active_high=self.config.active_high,
            initial_value=False,
        )

        relay_mode = "Active-High" if self.config.active_high else "Active-Low"
        self._initial_info = (
            f"{self.mode_badge} Trigger BCM {self.config.trigger_pin} | {relay_mode}"
        )
        self.gpio_info.emit(self._initial_info)
        self._log_pin_event(f"Hardware RP1 siap ({relay_mode}).")

    def _handle_physical_trigger_pressed(self) -> None:
        self._sensor_active = True
        self.sensor_state_changed.emit(True)
        self._log_pin_event(
            f"SENSOR FISIK (BCM {self.config.trigger_pin}): TRIGGERED (LOW)"
        )
        if self._trigger_enabled:
            self.trigger_received.emit()

    def _handle_physical_trigger_released(self) -> None:
        self._sensor_active = False
        self.sensor_state_changed.emit(False)

    def _apply_hardware_outputs(
        self, ok_on: bool, ng_on: bool, buzzer_on: bool
    ) -> None:
        for dev, state in (
            (self._ok_out, ok_on),
            (self._ng_out, ng_on),
            (self._buzzer_out, buzzer_on),
        ):
            if dev is not None:
                if state:
                    dev.on()
                else:
                    dev.off()

    def reconfigure_active_high(self, active_high: bool) -> None:
        if self.config.active_high == active_high:
            return
        self.all_off()
        self._release_devices()
        self.config.active_high = active_high
        self._initialize_hardware()

    def _release_devices(self) -> None:
        for dev in (self._ok_out, self._ng_out, self._buzzer_out):
            if dev is not None:
                try:
                    dev.off()
                    dev.close()
                except Exception:
                    pass
        if self._button is not None:
            try:
                self._button.when_pressed = None
                self._button.when_released = None
                self._button.close()
            except Exception:
                pass
        self._ok_out = None
        self._ng_out = None
        self._buzzer_out = None
        self._button = None

    def cleanup(self) -> None:
        super().cleanup()
        self._release_devices()


def create_gpio_controller(
    config: GPIOConfig, parent=None
) -> BaseGPIOController:
    """
    Factory lintas OS otomatis:
    - Windows / macOS: langsung memakai MockGPIOController (tidak crash meski tanpa gpiozero/lgpio).
    - Linux: mencoba HardwareGPIOController (RPi 5 lgpio), fallback aman ke MockGPIOController jika gagal.
    """
    os_name = platform.system()
    if os_name != "Linux":
        return MockGPIOController(
            config, reason=f"Debug OS {os_name}", parent=parent
        )

    if not _GPIOZERO_AVAILABLE:
        return MockGPIOController(
            config, reason="gpiozero/lgpio tidak tersedia", parent=parent
        )

    try:
        return HardwareGPIOController(config, parent=parent)
    except Exception as exc:
        return MockGPIOController(
            config, reason=f"Fallback ({type(exc).__name__})", parent=parent
        )
