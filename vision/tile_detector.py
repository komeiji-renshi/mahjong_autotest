from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional during bootstrap
    cv2 = None


@dataclass(slots=True)
class TileBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float


class TileDetector:
    """Hybrid detector for Vita Mahjong tile candidates."""

    def __init__(
        self,
        template_path: str | None = None,
        template_threshold: float = 0.84,
        min_tile_w: int = 50,
        max_tile_w: int = 180,
        min_tile_h: int = 60,
        max_tile_h: int = 220,
        size_templates: tuple[tuple[int, int], ...] | None = None,
        size_match_tolerance_ratio: float = 0.30,
        center_dedup_distance: int = 16,
        min_white_ratio: float = 0.30,
        white_sat_max: int = 90,
        white_val_min: int = 130,
        playfield_left_ratio: float = 0.12,
        playfield_right_ratio: float = 0.88,
        playfield_top_ratio: float = 0.05,
        playfield_bottom_ratio: float = 0.92,
    ) -> None:
        self.template_path = template_path
        self.template_threshold = template_threshold
        self.min_tile_w = min_tile_w
        self.max_tile_w = max_tile_w
        self.min_tile_h = min_tile_h
        self.max_tile_h = max_tile_h
        self.size_templates = size_templates or ((49, 65), (53, 69), (47, 64), (45, 60), (57, 74))
        self.size_match_tolerance_ratio = size_match_tolerance_ratio
        self.center_dedup_distance = center_dedup_distance
        self.min_white_ratio = min_white_ratio
        self.white_sat_max = white_sat_max
        self.white_val_min = white_val_min
        self.playfield_left_ratio = playfield_left_ratio
        self.playfield_right_ratio = playfield_right_ratio
        self.playfield_top_ratio = playfield_top_ratio
        self.playfield_bottom_ratio = playfield_bottom_ratio
        self._template_gray: np.ndarray | None = None

    def detect(self, img: np.ndarray) -> list[TileBox]:
        if cv2 is None:
            return []
        template_boxes = self._detect_by_template(img)
        contour_boxes = self._detect_by_contours(img)
        if len(contour_boxes) < 12:
            edge_boxes = self._detect_by_edges(img)
            contour_boxes = contour_boxes + edge_boxes
        merged = self._deduplicate(template_boxes + contour_boxes)
        return self._deduplicate_by_center(merged)

    def _load_template(self) -> np.ndarray | None:
        if self._template_gray is not None:
            return self._template_gray
        if not self.template_path:
            return None
        path = Path(self.template_path)
        if not path.exists() or cv2 is None:
            return None
        self._template_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return self._template_gray

    def _detect_by_template(self, img: np.ndarray) -> list[TileBox]:
        template = self._load_template()
        if template is None or cv2 is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= self.template_threshold)
        h, w = template.shape[:2]
        img_h, img_w = img.shape[:2]
        boxes: list[TileBox] = []
        for x, y in zip(xs, ys):
            if not self._in_playfield(int(x), int(y), img_w, img_h):
                continue
            size_score = self._size_template_score(w, h)
            if size_score <= 0:
                continue
            boxes.append(TileBox(int(x), int(y), int(w), int(h), float(result[y, x]) * size_score))
        return boxes

    def _detect_by_contours(self, img: np.ndarray) -> list[TileBox]:
        if cv2 is None:
            return []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, self.white_val_min), (179, self.white_sat_max, 255))
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = img.shape[:2]
        boxes: list[TileBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not self._in_playfield(x, y, img_w, img_h):
                continue
            if not (self.min_tile_w <= w <= self.max_tile_w):
                continue
            if not (self.min_tile_h <= h <= self.max_tile_h):
                continue
            aspect = h / max(w, 1)
            if not (1.1 <= aspect <= 2.3):
                continue
            area = w * h
            if area <= 0:
                continue
            patch = white_mask[y : y + h, x : x + w]
            white_ratio = float(np.count_nonzero(patch)) / float(area)
            if white_ratio < self.min_white_ratio:
                continue
            size_score = self._size_template_score(w, h)
            if size_score <= 0:
                continue
            confidence = min(1.0, 0.35 + (0.40 * size_score) + (0.25 * min(1.0, white_ratio)))
            boxes.append(TileBox(x=x, y=y, w=w, h=h, confidence=confidence))
        return self._deduplicate_by_center(self._deduplicate(boxes))

    def _detect_by_edges(self, img: np.ndarray) -> list[TileBox]:
        if cv2 is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = img.shape[:2]
        boxes: list[TileBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not self._in_playfield(x, y, img_w, img_h):
                continue
            if not (self.min_tile_w <= w <= self.max_tile_w):
                continue
            if not (self.min_tile_h <= h <= self.max_tile_h):
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if not (4 <= len(approx) <= 12):
                continue
            size_score = self._size_template_score(w, h)
            if size_score <= 0:
                continue
            confidence = min(1.0, 0.25 + (0.45 * size_score) + (0.30 * (len(approx) / 12.0)))
            boxes.append(TileBox(x=x, y=y, w=w, h=h, confidence=confidence))
        return self._deduplicate_by_center(self._deduplicate(boxes))

    def _deduplicate(self, boxes: list[TileBox], iou_threshold: float = 0.4) -> list[TileBox]:
        if not boxes:
            return []
        boxes_sorted = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        kept: list[TileBox] = []
        for box in boxes_sorted:
            if all(self._iou(box, existing) < iou_threshold for existing in kept):
                kept.append(box)
        return kept

    def _deduplicate_by_center(self, boxes: list[TileBox]) -> list[TileBox]:
        if not boxes:
            return []
        boxes_sorted = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        kept: list[TileBox] = []
        for box in boxes_sorted:
            cx = box.x + (box.w // 2)
            cy = box.y + (box.h // 2)
            duplicate = False
            for existing in kept:
                ex = existing.x + (existing.w // 2)
                ey = existing.y + (existing.h // 2)
                if abs(cx - ex) <= self.center_dedup_distance and abs(cy - ey) <= self.center_dedup_distance:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(box)
        return kept

    def _size_template_score(self, w: int, h: int) -> float:
        best_err = 9e9
        for tw, th in self.size_templates:
            if tw <= 0 or th <= 0:
                continue
            err_w = abs(w - tw) / float(tw)
            err_h = abs(h - th) / float(th)
            best_err = min(best_err, max(err_w, err_h))
        if best_err == 9e9:
            return 1.0
        if best_err > self.size_match_tolerance_ratio:
            return 0.0
        return max(0.0, 1.0 - (best_err / self.size_match_tolerance_ratio))

    def _in_playfield(self, x: int, y: int, img_w: int, img_h: int) -> bool:
        min_x = int(img_w * self.playfield_left_ratio)
        max_x = int(img_w * self.playfield_right_ratio)
        min_y = int(img_h * self.playfield_top_ratio)
        max_y = int(img_h * self.playfield_bottom_ratio)
        return min_x <= x <= max_x and min_y <= y <= max_y

    @staticmethod
    def _iou(a: TileBox, b: TileBox) -> float:
        ax2, ay2 = a.x + a.w, a.y + a.h
        bx2, by2 = b.x + b.w, b.y + b.h
        inter_x1 = max(a.x, b.x)
        inter_y1 = max(a.y, b.y)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        union_area = (a.w * a.h) + (b.w * b.h) - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area
