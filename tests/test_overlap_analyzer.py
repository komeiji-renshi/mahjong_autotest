from model.tile import Tile
from vision.overlap_analyzer import BoardAnalyzer


def _tile(tile_id: int, x: int, y: int, w: int = 50, h: int = 65) -> Tile:
    return Tile(id=tile_id, bbox=(x, y, w, h), center=(x + (w // 2), y + (h // 2)), class_id=1)


def test_side_blocking_uses_same_layer_constraint():
    target = _tile(1, 120, 100)
    diagonal_left = _tile(2, 60, 138)  # large center-y gap, should not block
    same_row_left = _tile(3, 65, 102)  # close center-y, should block

    board = BoardAnalyzer(
        overlap_ratio_threshold=0.12,
        side_margin_ratio=0.45,
        same_layer_y_ratio=0.35,
        min_side_y_overlap_ratio=0.45,
    ).build_board_state([target, diagonal_left, same_row_left], timestamp=1.0)

    t = next(tile for tile in board.tiles if tile.id == 1)
    assert t.left_blocked is True


def test_diagonal_neighbor_alone_does_not_block():
    target = _tile(1, 120, 100)
    diagonal_left = _tile(2, 60, 140)

    board = BoardAnalyzer(
        overlap_ratio_threshold=0.12,
        side_margin_ratio=0.45,
        same_layer_y_ratio=0.35,
        min_side_y_overlap_ratio=0.45,
    ).build_board_state([target, diagonal_left], timestamp=1.0)

    t = next(tile for tile in board.tiles if tile.id == 1)
    assert t.left_blocked is False
