from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from vision.preprocess import crop_tile_core, normalize_tile


@dataclass(slots=True)
class _TileFeature:
    gray_core: np.ndarray
    color_hist: np.ndarray
    mean_color: np.ndarray


class TileClassifier:
    """
    Online clustering classifier.
    First version groups tiles by normalized core-image distance.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        min_structure_similarity: float = 0.86,
        min_color_similarity: float = 0.72,
        structure_weight: float = 0.7,
        color_weight: float = 0.3,
        color_hist_bins: int = 16,
        max_mean_color_distance: float = 26.0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_structure_similarity = min_structure_similarity
        self.min_color_similarity = min_color_similarity
        self.structure_weight = structure_weight
        self.color_weight = color_weight
        self.color_hist_bins = color_hist_bins
        self.max_mean_color_distance = max_mean_color_distance
        self._prototypes: list[_TileFeature] = []

    def reset(self) -> None:
        self._prototypes.clear()

    def classify(self, tile_images: list[np.ndarray]) -> list[int]:
        class_ids: list[int] = []
        for tile_img in tile_images:
            class_ids.append(self.assign_class(tile_img))
        return class_ids

    def assign_class(self, tile_img: np.ndarray) -> int:
        feature = self._extract_feature(tile_img)
        for idx, proto in enumerate(self._prototypes):
            combined, structure, color, mean_color_dist = self._feature_similarity(feature, proto)
            if (
                combined >= self.similarity_threshold
                and structure >= self.min_structure_similarity
                and color >= self.min_color_similarity
                and mean_color_dist <= self.max_mean_color_distance
            ):
                return idx
        self._prototypes.append(feature)
        return len(self._prototypes) - 1

    def pair_similarity(self, tile_a: np.ndarray, tile_b: np.ndarray) -> tuple[float, float, float, float]:
        feature_a = self._extract_feature(tile_a)
        feature_b = self._extract_feature(tile_b)
        return self._feature_similarity(feature_a, feature_b)

    def _extract_feature(self, tile_img: np.ndarray) -> _TileFeature:
        core = crop_tile_core(tile_img, ratio=0.7)
        norm = normalize_tile(core, size=(64, 64)).astype(np.float32)
        if norm.ndim == 2:
            gray = norm
            color = np.repeat(norm[:, :, None], 3, axis=2)
        else:
            gray = norm.mean(axis=2)
            color = norm

        hist_parts: list[np.ndarray] = []
        for ch in range(3):
            hist, _ = np.histogram(color[:, :, ch], bins=self.color_hist_bins, range=(0, 255))
            hist_parts.append(hist.astype(np.float32))
        channel_means = color.reshape(-1, 3).mean(axis=0).astype(np.float32)
        hsv_hist = self._hsv_color_hist(color, bins=self.color_hist_bins)
        color_hist = np.concatenate(hist_parts + [channel_means, hsv_hist])
        color_hist = self._l2_normalize(color_hist)

        return _TileFeature(gray_core=gray, color_hist=color_hist, mean_color=channel_means)

    def _feature_similarity(self, a: _TileFeature, b: _TileFeature) -> tuple[float, float, float, float]:
        structure = self._cosine_similarity(a.gray_core, b.gray_core)
        color = self._cosine_similarity(a.color_hist, b.color_hist)
        mean_color_dist = float(np.linalg.norm(a.mean_color - b.mean_color))
        combined = (self.structure_weight * structure) + (self.color_weight * color)
        return float(combined), float(structure), float(color), mean_color_dist

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        va = a.flatten()
        vb = b.flatten()
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(v))
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def _hsv_color_hist(color_bgr: np.ndarray, bins: int) -> np.ndarray:
        b = color_bgr[:, :, 0]
        g = color_bgr[:, :, 1]
        r = color_bgr[:, :, 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc

        sat = np.zeros_like(maxc, dtype=np.float32)
        nonzero = maxc > 0
        sat[nonzero] = delta[nonzero] / maxc[nonzero]
        value = maxc / 255.0

        hue = np.zeros_like(maxc, dtype=np.float32)
        mask = delta > 0
        r_is_max = (maxc == r) & mask
        g_is_max = (maxc == g) & mask
        b_is_max = (maxc == b) & mask
        hue[r_is_max] = ((g[r_is_max] - b[r_is_max]) / delta[r_is_max]) % 6.0
        hue[g_is_max] = ((b[g_is_max] - r[g_is_max]) / delta[g_is_max]) + 2.0
        hue[b_is_max] = ((r[b_is_max] - g[b_is_max]) / delta[b_is_max]) + 4.0
        hue = hue / 6.0

        colorful = sat > 0.15
        if not np.any(colorful):
            return np.zeros((bins * 2,), dtype=np.float32)
        hue_hist, _ = np.histogram(hue[colorful], bins=bins, range=(0.0, 1.0))
        sat_hist, _ = np.histogram(sat[colorful], bins=bins, range=(0.0, 1.0))
        hv = np.concatenate([hue_hist.astype(np.float32), sat_hist.astype(np.float32)])
        return hv
