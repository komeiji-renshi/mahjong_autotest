from dataclasses import dataclass

from model.tile import Tile


@dataclass(slots=True)
class BoardState:
    """Computed board snapshot used by solver/runner layers."""

    tiles: list[Tile]
    timestamp: float
    screenshot_path: str | None = None

    def clickable_tiles(self) -> list[Tile]:
        return [tile for tile in self.tiles if tile.clickable]

    def group_by_class(self, clickable_only: bool = True) -> dict[int, list[Tile]]:
        source = self.clickable_tiles() if clickable_only else self.tiles
        grouped: dict[int, list[Tile]] = {}
        for tile in source:
            if tile.class_id is None:
                continue
            grouped.setdefault(tile.class_id, []).append(tile)
        return grouped
