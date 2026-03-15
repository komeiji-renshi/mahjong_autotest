from model.board_state import BoardState
from model.tile import Tile
from solver.pair_matcher import PairMatcher
from solver.strategy import Strategy


class ActionPlanner:
    """Glue PairMatcher + Strategy together for readability."""

    def __init__(self, matcher: PairMatcher, strategy: Strategy) -> None:
        self.matcher = matcher
        self.strategy = strategy

    def next_action(self, board: BoardState) -> tuple[Tile, Tile] | None:
        pairs = self.matcher.find_pairs(board)
        return self.strategy.choose_action(pairs, board)
