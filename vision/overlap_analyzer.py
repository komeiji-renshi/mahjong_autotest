from __future__ import annotations

from model.board_state import BoardState
from model.tile import Tile


class BoardAnalyzer:
    """Computes covered_by/left_right_blocked/clickable states."""

    def __init__(
        self,
        overlap_ratio_threshold: float = 0.18,
        side_margin_ratio: float = 0.35,
        same_layer_y_ratio: float = 0.35,
        min_side_y_overlap_ratio: float = 0.45,
    ) -> None:
        self.overlap_ratio_threshold = overlap_ratio_threshold
        self.side_margin_ratio = side_margin_ratio
        self.same_layer_y_ratio = same_layer_y_ratio
        self.min_side_y_overlap_ratio = min_side_y_overlap_ratio

    def build_board_state(self, tiles: list[Tile], timestamp: float) -> BoardState:
        self._compute_covered(tiles)
        self._compute_side_blocking(tiles)
        for tile in tiles:
            tile.clickable = (len(tile.covered_by) == 0) and (not (tile.left_blocked and tile.right_blocked))
        return BoardState(tiles=tiles, timestamp=timestamp)

    def _compute_covered(self, tiles: list[Tile]) -> None:
        for tile in tiles:
            tile.covered_by.clear()
        for tile in tiles:
            tx, ty, tw, th = tile.bbox
            for other in tiles:
                if tile.id == other.id:
                    continue
                ox, oy, ow, oh = other.bbox
                if oy >= ty:
                    continue
                overlap_w = max(0, min(tx + tw, ox + ow) - max(tx, ox))
                overlap_h = max(0, min(ty + th, oy + oh) - max(ty, oy))
                overlap_area = overlap_w * overlap_h
                tile_area = tw * th
                ratio = overlap_area / max(tile_area, 1)
                if ratio >= self.overlap_ratio_threshold:
                    tile.covered_by.append(other.id)

    def _compute_side_blocking(self, tiles: list[Tile]) -> None:
        if not tiles:
            return
        tile_heights = sorted(tile.bbox[3] for tile in tiles)
        median_h = tile_heights[len(tile_heights) // 2]
        same_layer_tol = max(6, int(median_h * self.same_layer_y_ratio))
        for tile in tiles:
            tile.left_blocked = False
            tile.right_blocked = False
        for tile in tiles:
            tx, ty, tw, th = tile.bbox
            x_margin = int(tw * self.side_margin_ratio)
            tcy = ty + (th / 2.0)
            for other in tiles:
                if tile.id == other.id:
                    continue
                ox, oy, ow, oh = other.bbox
                ocy = oy + (oh / 2.0)
                if abs(tcy - ocy) > same_layer_tol:
                    continue
                y_overlap = max(0, min(ty + th, oy + oh) - max(ty, oy))
                if y_overlap <= 0:
                    continue
                overlap_ratio = y_overlap / max(1, min(th, oh))
                if overlap_ratio < self.min_side_y_overlap_ratio:
                    continue
                if ox + ow <= tx and (tx - (ox + ow)) <= x_margin:
                    tile.left_blocked = True
                if ox >= tx + tw and (ox - (tx + tw)) <= x_margin:
                    tile.right_blocked = True
