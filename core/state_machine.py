from enum import Enum, auto


class BotState(Enum):
    INIT = auto()
    IDLE = auto()
    IN_LEVEL = auto()
    ANALYZING = auto()
    ACTING = auto()
    WAIT_ANIMATION = auto()
    LEVEL_WIN = auto()
    LEVEL_FAIL = auto()
    POPUP = auto()
    RECOVERING = auto()
    STOPPED = auto()


_ALLOWED_TRANSITIONS: dict[BotState, set[BotState]] = {
    BotState.INIT: {BotState.IDLE, BotState.STOPPED},
    BotState.IDLE: {BotState.IN_LEVEL, BotState.STOPPED},
    BotState.IN_LEVEL: {BotState.ANALYZING, BotState.POPUP, BotState.RECOVERING, BotState.STOPPED},
    BotState.ANALYZING: {
        BotState.ACTING,
        BotState.LEVEL_WIN,
        BotState.LEVEL_FAIL,
        BotState.POPUP,
        BotState.RECOVERING,
        BotState.STOPPED,
    },
    BotState.ACTING: {BotState.WAIT_ANIMATION, BotState.RECOVERING, BotState.STOPPED},
    BotState.WAIT_ANIMATION: {
        BotState.ANALYZING,
        BotState.LEVEL_WIN,
        BotState.LEVEL_FAIL,
        BotState.RECOVERING,
        BotState.STOPPED,
    },
    BotState.LEVEL_WIN: {BotState.IDLE, BotState.IN_LEVEL, BotState.STOPPED},
    BotState.LEVEL_FAIL: {BotState.IDLE, BotState.RECOVERING, BotState.STOPPED},
    BotState.POPUP: {BotState.ANALYZING, BotState.RECOVERING, BotState.STOPPED},
    BotState.RECOVERING: {BotState.IDLE, BotState.IN_LEVEL, BotState.STOPPED},
    BotState.STOPPED: set(),
}


class StateMachine:
    def __init__(self, initial: BotState = BotState.INIT) -> None:
        self.current = initial

    def transition(self, next_state: BotState) -> None:
        if next_state == self.current:
            return
        allowed = _ALLOWED_TRANSITIONS.get(self.current, set())
        if next_state not in allowed:
            raise ValueError(f"Invalid transition: {self.current.name} -> {next_state.name}")
        self.current = next_state
