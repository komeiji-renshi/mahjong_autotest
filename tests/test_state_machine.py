import pytest

from core.state_machine import BotState, StateMachine


def test_state_machine_allows_happy_path_transitions():
    sm = StateMachine()

    sm.transition(BotState.IDLE)
    sm.transition(BotState.IN_LEVEL)
    sm.transition(BotState.ANALYZING)
    sm.transition(BotState.ACTING)
    sm.transition(BotState.WAIT_ANIMATION)
    sm.transition(BotState.ANALYZING)
    sm.transition(BotState.LEVEL_WIN)
    sm.transition(BotState.IDLE)

    assert sm.current == BotState.IDLE


def test_state_machine_rejects_invalid_transition():
    sm = StateMachine()

    with pytest.raises(ValueError):
        sm.transition(BotState.ACTING)


def test_state_machine_allows_idempotent_transition():
    sm = StateMachine()
    sm.transition(BotState.IDLE)
    sm.transition(BotState.IN_LEVEL)
    sm.transition(BotState.ANALYZING)

    sm.transition(BotState.ANALYZING)

    assert sm.current == BotState.ANALYZING
