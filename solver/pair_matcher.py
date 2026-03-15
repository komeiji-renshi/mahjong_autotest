from model.board_state import BoardState
from model.tile import Tile


class PairMatcher:
    """Find clickable same-class pairs for Mahjong solitaire rules."""

    def __init__(self, max_pairs_per_class: int = 24) -> None:
        self.max_pairs_per_class = max_pairs_per_class

    def find_pairs(self, board: BoardState) -> list[tuple[Tile, Tile]]:
        grouped = board.group_by_class(clickable_only=True)
        pairs: list[tuple[Tile, Tile]] = []
        for tiles in grouped.values():
            if len(tiles) < 2:
                continue
            ranked_pairs: list[tuple[int, int, tuple[Tile, Tile]]] = []
            for i in range(len(tiles)):
                for j in range(i + 1, len(tiles)):
                    a, b = tiles[i], tiles[j]
                    manhattan = abs(a.center[0] - b.center[0]) + abs(a.center[1] - b.center[1])
                    row_gap = abs(a.center[1] - b.center[1])
                    ranked_pairs.append((row_gap, manhattan, (a, b)))
            ranked_pairs.sort(key=lambda item: (item[0], item[1]))
            for _, _, pair in ranked_pairs[: self.max_pairs_per_class]:
                pairs.append(pair)
        return pairs
