from model.tile import Tile


def is_clickable(tile: Tile) -> bool:
    return (len(tile.covered_by) == 0) and (not (tile.left_blocked and tile.right_blocked))
