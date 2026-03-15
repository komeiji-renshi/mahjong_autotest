from model.board_state import BoardState
from model.tile import Tile


def make_tile(
    tile_id: int,
    class_id: int | None,
    clickable: bool,
) -> Tile:
    return Tile(
        id=tile_id,
        bbox=(10 * tile_id, 10, 40, 60),
        center=(10 * tile_id + 20, 40),
        class_id=class_id,
        clickable=clickable,
    )


def test_clickable_tiles_and_grouping():
    board = BoardState(
        tiles=[
            make_tile(1, class_id=101, clickable=True),
            make_tile(2, class_id=101, clickable=False),
            make_tile(3, class_id=202, clickable=True),
            make_tile(4, class_id=None, clickable=True),
        ],
        timestamp=123.0,
    )

    clickable = board.clickable_tiles()
    grouped = board.group_by_class(clickable_only=False)

    assert [tile.id for tile in clickable] == [1, 3, 4]
    assert set(grouped.keys()) == {101, 202}
    assert [tile.id for tile in grouped[101]] == [1, 2]
    assert [tile.id for tile in grouped[202]] == [3]
