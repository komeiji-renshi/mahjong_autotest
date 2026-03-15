from __future__ import annotations

import logging
from dataclasses import dataclass

from core.controller_protocol import Controller
from core.state_machine import BotState, StateMachine
from runner.level_runner import LevelResult, LevelRunner


@dataclass(slots=True)
class RunSummary:
    total_levels: int
    success_levels: int
    fail_levels: int
    reasons: list[str]


class GameRunner:
    def __init__(self, controller: Controller, level_runner: LevelRunner, state_machine: StateMachine) -> None:
        self.controller = controller
        self.level_runner = level_runner
        self.sm = state_machine
        self.log = logging.getLogger("vita_mahjong_bot")

    def run(self, max_levels: int) -> RunSummary:
        if self.sm.current == BotState.INIT:
            self.sm.transition(BotState.IDLE)

        success = 0
        fail = 0
        reasons: list[str] = []

        self.controller.start_app()
        for idx in range(max_levels):
            if self.sm.current not in {BotState.IDLE, BotState.IN_LEVEL}:
                self.sm.transition(BotState.IDLE)

            result: LevelResult = self.level_runner.run_one_level()
            reasons.append(result.reason)
            if result.success:
                success += 1
            else:
                fail += 1

            # Minimal transition to next level. Replace with real "next" template action.
            self.controller.keyevent(66)  # enter
            if self.sm.current == BotState.LEVEL_WIN:
                self.sm.transition(BotState.IDLE)
            elif self.sm.current == BotState.LEVEL_FAIL:
                self.sm.transition(BotState.IDLE)
            elif self.sm.current == BotState.RECOVERING:
                is_last_level = idx == (max_levels - 1)
                if is_last_level:
                    self.log.info("Skip recovery app restart on final level.")
                else:
                    self.controller.stop_app()
                    self.controller.start_app()
                    self.sm.transition(BotState.IDLE)

            self.log.info("Level finished: success=%s reason=%s", result.success, result.reason)

        self.sm.transition(BotState.STOPPED)
        return RunSummary(
            total_levels=max_levels,
            success_levels=success,
            fail_levels=fail,
            reasons=reasons,
        )
