from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional during bootstrap
    cv2 = None


@dataclass(slots=True)
class DeviceConfig:
    serial: str | None = None
    adb_path: str = "adb"
    package_name: str = ""
    activity_name: str = ""


class AdbController:
    """ADB wrapper for Vita Mahjong automation."""

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config

    def _base_cmd(self) -> list[str]:
        cmd = [self._config.adb_path]
        if self._config.serial:
            cmd.extend(["-s", self._config.serial])
        return cmd

    def _run(self, extra: list[str], timeout_sec: float = 10.0) -> subprocess.CompletedProcess[str]:
        cmd = self._base_cmd() + extra
        return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_sec)

    def connect(self) -> bool:
        result = self._run(["devices"], timeout_sec=8.0)
        return result.returncode == 0 and "device" in result.stdout

    def is_device_online(self) -> bool:
        result = self._run(["get-state"], timeout_sec=5.0)
        return result.returncode == 0 and "device" in result.stdout

    def start_app(self) -> None:
        if not self._config.package_name:
            raise ValueError("package_name is required")
        if self._config.activity_name:
            target = f"{self._config.package_name}/{self._config.activity_name}"
            self._run(["shell", "am", "start", "-n", target], timeout_sec=12.0)
            return
        self._run(
            ["shell", "monkey", "-p", self._config.package_name, "-c", "android.intent.category.LAUNCHER", "1"],
            timeout_sec=12.0,
        )

    def stop_app(self) -> None:
        if not self._config.package_name:
            raise ValueError("package_name is required")
        self._run(["shell", "am", "force-stop", self._config.package_name], timeout_sec=8.0)

    def keyevent(self, keycode: int) -> None:
        self._run(["shell", "input", "keyevent", str(keycode)], timeout_sec=5.0)

    def tap(self, x: int, y: int) -> None:
        self._run(["shell", "input", "tap", str(x), str(y)], timeout_sec=4.0)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 200) -> None:
        self._run(
            ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration)],
            timeout_sec=6.0,
        )

    def screencap(self) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for screencap decoding")

        cmd = self._base_cmd() + ["exec-out", "screencap", "-p"]
        raw = subprocess.check_output(cmd, timeout=8.0)
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("failed to decode adb screencap")
        return image
