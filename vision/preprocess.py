from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional during bootstrap
    cv2 = None


def crop_play_area(img: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = region
    return img[y1:y2, x1:x2].copy()


def preprocess_screen(img: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def normalize_tile(tile_img: np.ndarray, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    if cv2 is None:
        return tile_img
    return cv2.resize(tile_img, size, interpolation=cv2.INTER_AREA)


def crop_tile_core(tile_img: np.ndarray, ratio: float = 0.7) -> np.ndarray:
    h, w = tile_img.shape[:2]
    core_w = int(w * ratio)
    core_h = int(h * ratio)
    x1 = (w - core_w) // 2
    y1 = (h - core_h) // 2
    return tile_img[y1 : y1 + core_h, x1 : x1 + core_w].copy()
