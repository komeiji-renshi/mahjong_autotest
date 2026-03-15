from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

import numpy as np

try:
    import pyautogui  # type: ignore
except ImportError:  # pragma: no cover - optional during bootstrap
    pyautogui = None

try:
    import pygetwindow as gw  # type: ignore
except ImportError:  # pragma: no cover - optional during bootstrap
    gw = None


@dataclass(slots=True)
class PcConfig:
    window_title_keyword: str = "Mahjong"
    app_aumid: str = ""
    launch_cmd: str = ""
    process_name: str = ""


class PcController:
    """
    Windows controller for Microsoft Store Mahjong apps.
    Coordinates used by tap() are relative to game window.
    """

    def __init__(self, config: PcConfig) -> None:
        self._config = config

    def connect(self) -> bool:
        return self._get_window() is not None

    def is_device_online(self) -> bool:
        return self.connect()

    def start_app(self) -> None:
        if self._config.launch_cmd:
            subprocess.Popen(self._config.launch_cmd, shell=True)
            time.sleep(2.0)
            return
        if self._config.app_aumid:
            cmd = f'start shell:AppsFolder\\{self._config.app_aumid}'
            subprocess.Popen(["cmd", "/c", cmd], shell=False)
            time.sleep(2.0)

    def stop_app(self) -> None:
        if not self._config.process_name:
            return
        subprocess.run(
            ["taskkill", "/IM", self._config.process_name, "/F"],
            check=False,
            capture_output=True,
            text=True,
        )

    def screencap(self) -> np.ndarray:
        if pyautogui is None:
            raise RuntimeError("pyautogui is required for PC screencap")
        win = self._require_window()
        screenshot = pyautogui.screenshot(region=(win.left, win.top, win.width, win.height))
        rgb = np.array(screenshot)
        # convert RGB to BGR for OpenCV-compatible downstream pipeline
        return rgb[:, :, ::-1].copy()

    def tap(self, x: int, y: int) -> None:
        if pyautogui is None:
            raise RuntimeError("pyautogui is required for PC tap")
        win = self._require_window()
        px = win.left + x
        py = win.top + y
        pyautogui.click(px, py)

    def keyevent(self, keycode: int) -> None:
        if pyautogui is None:
            raise RuntimeError("pyautogui is required for PC keyevent")
        key_map = {4: "esc", 66: "enter"}
        key = key_map.get(keycode)
        if key:
            pyautogui.press(key)

    def _get_window(self):
        if gw is None:
            raise RuntimeError("pygetwindow is required for PC window discovery")
        keyword = self._config.window_title_keyword.lower()
        windows = gw.getAllWindows()
        candidates = [w for w in windows if w.title and keyword in w.title.lower()]
        if not candidates:
            return None
        win = candidates[0]
        try:
            win.activate()
        except Exception:
            pass
        return win

    def _require_window(self):
        win = self._get_window()
        if win is None:
            raise RuntimeError(
                f"Could not find game window with keyword: {self._config.window_title_keyword!r}"
            )
        return win
