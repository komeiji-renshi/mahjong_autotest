import numpy as np

from vision.tile_detector import TileBox, TileDetector

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def _draw_tile(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    # Tile-like rectangle with border and inner content.
    cv2.rectangle(img, (x, y), (x + w, y + h), (245, 245, 245), thickness=-1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), thickness=2)
    cv2.circle(img, (x + w // 2, y + h // 2), max(3, w // 6), (50, 50, 180), thickness=-1)


def test_tile_detector_ignores_non_playfield_header_shapes():
    if cv2 is None:
        return

    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    _draw_tile(img, 20, 20, 50, 65)  # header area (should be ignored)
    _draw_tile(img, 300, 180, 50, 65)  # playfield
    _draw_tile(img, 360, 180, 50, 65)  # playfield

    detector = TileDetector(
        template_path=None,
        min_tile_w=38,
        max_tile_w=95,
        min_tile_h=48,
        max_tile_h=130,
        playfield_left_ratio=0.12,
        playfield_right_ratio=0.88,
        playfield_top_ratio=0.10,
        playfield_bottom_ratio=0.92,
    )
    boxes = detector.detect(img)
    assert len(boxes) >= 2
    assert all(box.y >= 60 for box in boxes)


def test_tile_detector_merges_template_and_contour_results():
    class _FakeDetector(TileDetector):
        def _detect_by_template(self, img):  # type: ignore[override]
            return [TileBox(x=100, y=100, w=50, h=65, confidence=0.8)]

        def _detect_by_contours(self, img):  # type: ignore[override]
            return [TileBox(x=220, y=120, w=49, h=64, confidence=0.6)]

    detector = _FakeDetector(template_path=None)
    boxes = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(boxes) == 2


def test_tile_detector_allows_top_row_when_top_ratio_relaxed():
    if cv2 is None:
        return

    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    _draw_tile(img, 300, 36, 50, 65)  # top playable row

    detector = TileDetector(
        template_path=None,
        min_tile_w=38,
        max_tile_w=95,
        min_tile_h=40,
        max_tile_h=130,
        playfield_left_ratio=0.12,
        playfield_right_ratio=0.88,
        playfield_top_ratio=0.05,
        playfield_bottom_ratio=0.92,
    )
    boxes = detector.detect(img)
    assert len(boxes) >= 1
