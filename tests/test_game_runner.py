from core.state_machine import BotState, StateMachine
from runner.game_runner import GameRunner
from runner.level_runner import LevelResult


class _FakeController:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.keyevents: list[int] = []

    def connect(self) -> bool:
        return True

    def is_device_online(self) -> bool:
        return True

    def start_app(self) -> None:
        self.start_calls += 1

    def stop_app(self) -> None:
        self.stop_calls += 1

    def screencap(self):
        raise NotImplementedError

    def tap(self, x: int, y: int) -> None:
        raise NotImplementedError

    def keyevent(self, keycode: int) -> None:
        self.keyevents.append(keycode)


class _FakeLevelRunner:
    def __init__(self, sm: StateMachine, results: list[LevelResult]) -> None:
        self.sm = sm
        self.results = results[:]

    def run_one_level(self) -> LevelResult:
        result = self.results.pop(0)
        self.sm.transition(BotState.IN_LEVEL)
        if result.success:
            self.sm.transition(BotState.ANALYZING)
            self.sm.transition(BotState.LEVEL_WIN)
        else:
            self.sm.transition(BotState.RECOVERING)
        return result


def test_game_runner_skips_recovery_restart_on_final_level():
    sm = StateMachine()
    controller = _FakeController()
    level_runner = _FakeLevelRunner(sm, [LevelResult(success=False, reason="no_pairs_stalled", steps=10)])
    runner = GameRunner(controller=controller, level_runner=level_runner, state_machine=sm)

    summary = runner.run(max_levels=1)

    assert summary.fail_levels == 1
    assert controller.start_calls == 1
    assert controller.stop_calls == 0


def test_game_runner_restarts_recovery_when_more_levels_remaining():
    sm = StateMachine()
    controller = _FakeController()
    level_runner = _FakeLevelRunner(
        sm,
        [
            LevelResult(success=False, reason="no_pairs_stalled", steps=10),
            LevelResult(success=True, reason="win", steps=5),
        ],
    )
    runner = GameRunner(controller=controller, level_runner=level_runner, state_machine=sm)

    summary = runner.run(max_levels=2)

    assert summary.total_levels == 2
    assert controller.stop_calls == 1
    assert controller.start_calls == 2
