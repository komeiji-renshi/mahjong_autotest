from dataclasses import dataclass, field

from runner.level_runner import LevelResult


@dataclass(slots=True)
class ResultCollector:
    results: list[LevelResult] = field(default_factory=list)

    def record(self, result: LevelResult) -> None:
        self.results.append(result)

    def summary(self) -> dict[str, int]:
        success = sum(1 for item in self.results if item.success)
        fail = len(self.results) - success
        return {"total": len(self.results), "success": success, "fail": fail}
