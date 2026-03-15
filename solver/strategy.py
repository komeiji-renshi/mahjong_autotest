from model.board_state import BoardState
from model.tile import Tile


class Strategy:
    """
    First usable strategy:
    prefer pairs whose two tiles currently expose more side freedom.
    """

    def choose_action(
        self,
        pairs: list[tuple[Tile, Tile]],
        board: BoardState,
    ) -> tuple[Tile, Tile] | None:
        if not pairs:
            return None

        def score(pair: tuple[Tile, Tile]) -> float:
            a, b = pair
            score_a = int(not a.left_blocked) + int(not a.right_blocked)
            score_b = int(not b.left_blocked) + int(not b.right_blocked)
            manhattan = abs(a.center[0] - b.center[0]) + abs(a.center[1] - b.center[1])
            # Prefer freer and spatially closer pairs to reduce random cross-board trials.
            return (score_a + score_b) - (manhattan / 240.0)

        return max(pairs, key=score)
