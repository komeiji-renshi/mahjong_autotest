from dataclasses import dataclass, field


@dataclass(slots=True)
class Tile:
    """Single detected tile on Vita Mahjong board."""

    id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    class_id: int | None = None
    confidence: float = 1.0
    visible_ratio: float = 1.0
    covered_by: list[int] = field(default_factory=list)
    left_blocked: bool = False
    right_blocked: bool = False
    clickable: bool = False
