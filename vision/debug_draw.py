from __future__ import annotations

from pathlib import Path

import numpy as np

from model.tile import Tile

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional during bootstrap
    cv2 = None


def draw_tiles_debug(image: np.ndarray, tiles: list[Tile], output_path: str) -> None:
    if cv2 is None:
        return
    canvas = image.copy()
    for tile in tiles:
        x, y, w, h = tile.bbox
        color = (0, 200, 0) if tile.clickable else (0, 0, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        text = f"id={tile.id} c={tile.class_id} free={int(tile.clickable)}"
        cv2.putText(canvas, text, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), canvas)
