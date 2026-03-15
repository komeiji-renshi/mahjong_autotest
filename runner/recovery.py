from core.controller_protocol import Controller


class RecoveryHandler:
    """Central place for recovery actions in later phases."""

    def __init__(self, controller: Controller) -> None:
        self.controller = controller

    def recover_from_stall(self) -> None:
        self.controller.keyevent(4)  # back / esc
        self.controller.stop_app()
        self.controller.start_app()
