from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional during bootstrap
    cv2 = None


class UiRecognizer:
    """
    Lightweight template-based UI state recognizer.
    Keys supported by default: win, fail, popup.
    """

    def __init__(self, templates: dict[str, str], threshold: float = 0.82) -> None:
        self.templates = templates
        self.threshold = threshold
        self._cache: dict[str, np.ndarray] = {}

    def recognize(self, image: np.ndarray) -> str:
        if self._match(image, "win"):
            return "win"
        if self._match(image, "fail"):
            return "fail"
        if self._match(image, "popup"):
            return "popup"
        return "in_level"

    def _match(self, image: np.ndarray, key: str) -> bool:
        if cv2 is None:
            return False
        template = self._load_template(key)
        if template is None:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, _ = cv2.minMaxLoc(result)
        return max_score >= self.threshold

    def _load_template(self, key: str) -> np.ndarray | None:
        if key in self._cache:
            return self._cache[key]
        path = self.templates.get(key)
        if not path:
            return None
        file_path = Path(path)
        if not file_path.exists() or cv2 is None:
            return None
        template = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            return None
        self._cache[key] = template
        return template
