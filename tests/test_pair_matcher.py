from model.board_state import BoardState
from model.tile import Tile
from solver.pair_matcher import PairMatcher


def t(tile_id: int, class_id: int, clickable: bool) -> Tile:
    return Tile(
        id=tile_id,
        bbox=(0, 0, 40, 60),
        center=(20, 30),
        class_id=class_id,
        clickable=clickable,
    )


def test_find_pairs_uses_clickable_tiles_only():
    board = BoardState(
        tiles=[
            t(1, 7, True),
            t(2, 7, True),
            t(3, 7, False),
            t(4, 9, True),
            t(5, 9, True),
        ],
        timestamp=1.0,
    )

    pairs = PairMatcher().find_pairs(board)

    assert [(a.id, b.id) for a, b in pairs] == [(1, 2), (4, 5)]


def test_find_pairs_returns_all_combinations_per_class():
    board = BoardState(
        tiles=[
            t(1, 7, True),
            t(2, 7, True),
            t(3, 7, True),
        ],
        timestamp=1.0,
    )

    pairs = PairMatcher().find_pairs(board)

    assert {(a.id, b.id) for a, b in pairs} == {(1, 2), (1, 3), (2, 3)}


def test_find_pairs_limits_pairs_per_class_and_prefers_nearby():
    board = BoardState(
        tiles=[
            Tile(id=1, bbox=(0, 0, 40, 60), center=(20, 30), class_id=7, clickable=True),
            Tile(id=2, bbox=(50, 0, 40, 60), center=(70, 30), class_id=7, clickable=True),
            Tile(id=3, bbox=(110, 0, 40, 60), center=(130, 30), class_id=7, clickable=True),
            Tile(id=4, bbox=(700, 0, 40, 60), center=(720, 30), class_id=7, clickable=True),
        ],
        timestamp=1.0,
    )

    pairs = PairMatcher(max_pairs_per_class=2).find_pairs(board)

    assert len(pairs) == 2
    pair_ids = {(a.id, b.id) for a, b in pairs}
    assert (1, 2) in pair_ids
