import time


class ProgressWatchdog:
    """Tracks whether board is progressing to detect stalls."""

    def __init__(self, timeout_sec: float, max_no_progress_actions: int) -> None:
        self._timeout_sec = timeout_sec
        self._max_no_progress_actions = max_no_progress_actions
        self._last_progress_ts = time.time()
        self._no_progress_actions = 0

    def mark_progress(self) -> None:
        self._last_progress_ts = time.time()
        self._no_progress_actions = 0

    def mark_no_progress_action(self) -> None:
        self._no_progress_actions += 1

    def is_stalled(self) -> bool:
        timeout_reached = (time.time() - self._last_progress_ts) >= self._timeout_sec
        action_limit_reached = self._no_progress_actions >= self._max_no_progress_actions
        return timeout_reached or action_limit_reached
