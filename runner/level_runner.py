from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional during bootstrap
    cv2 = None

from core.controller_protocol import Controller
from core.state_machine import BotState, StateMachine
from core.watchdog import ProgressWatchdog
from model.tile import Tile
from solver.pair_matcher import PairMatcher
from solver.strategy import Strategy
from vision.debug_draw import draw_tiles_debug
from vision.overlap_analyzer import BoardAnalyzer
from vision.tile_classifier import TileClassifier
from vision.tile_detector import TileDetector
from vision.ui_recognizer import UiRecognizer


@dataclass(slots=True)
class LevelResult:
    success: bool
    reason: str
    steps: int


class LevelRunner:
    def __init__(
        self,
        controller: Controller,
        detector: TileDetector,
        classifier: TileClassifier,
        analyzer: BoardAnalyzer,
        matcher: PairMatcher,
        strategy: Strategy,
        ui: UiRecognizer,
        state_machine: StateMachine,
        debug_dir: str = "logs/debug",
        animation_wait_sec: float = 0.6,
        level_timeout_sec: float = 90.0,
        max_same_board_cycles: int = 40,
        min_pair_similarity: float = 0.90,
        min_pair_structure_similarity: float = 0.92,
        min_pair_color_similarity: float = 0.86,
        max_pair_mean_color_distance: float = 26.0,
        min_tiles_for_action: int = 20,
        watchdog_timeout_sec: float = 30.0,
        watchdog_no_progress_actions: int = 30,
        hint_trigger_no_progress_actions: int = 10,
        hint_wait_sec: float = 0.35,
    ) -> None:
        self.controller = controller
        self.detector = detector
        self.classifier = classifier
        self.analyzer = analyzer
        self.matcher = matcher
        self.strategy = strategy
        self.ui = ui
        self.sm = state_machine
        self.debug_dir = debug_dir
        self.animation_wait_sec = animation_wait_sec
        self.level_timeout_sec = level_timeout_sec
        self.max_same_board_cycles = max_same_board_cycles
        self.min_pair_similarity = min_pair_similarity
        self.min_pair_structure_similarity = min_pair_structure_similarity
        self.min_pair_color_similarity = min_pair_color_similarity
        self.max_pair_mean_color_distance = max_pair_mean_color_distance
        self.min_tiles_for_action = min_tiles_for_action
        self.watchdog_timeout_sec = watchdog_timeout_sec
        self.watchdog_no_progress_actions = watchdog_no_progress_actions
        self.hint_trigger_no_progress_actions = hint_trigger_no_progress_actions
        self.hint_wait_sec = hint_wait_sec
        self.log = logging.getLogger("vita_mahjong_bot.level")

    def run_one_level(self, max_steps: int = 400) -> LevelResult:
        watchdog = ProgressWatchdog(
            timeout_sec=self.watchdog_timeout_sec,
            max_no_progress_actions=self.watchdog_no_progress_actions,
        )
        failed_pair_keys: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        failed_pair_centers: list[tuple[tuple[int, int], tuple[int, int]]] = []
        last_board_signature: tuple[tuple[int, int, int, int], ...] | None = None
        same_board_cycles = 0
        no_progress_streak = 0
        level_start_ts = time.time()
        self.sm.transition(BotState.IN_LEVEL)
        steps = 0

        while steps < max_steps:
            elapsed = time.time() - level_start_ts
            if elapsed >= self.level_timeout_sec:
                self.log.warning("Level timeout reached: %.1fs", elapsed)
                self.sm.transition(BotState.RECOVERING)
                return LevelResult(success=False, reason="level_timeout", steps=steps)
            self.sm.transition(BotState.ANALYZING)
            image = self.controller.screencap()
            state = self.ui.recognize(image)

            if state == "win":
                self.sm.transition(BotState.LEVEL_WIN)
                return LevelResult(success=True, reason="win", steps=steps)
            if state == "fail":
                self.sm.transition(BotState.LEVEL_FAIL)
                return LevelResult(success=False, reason="fail", steps=steps)
            if state == "popup":
                self.sm.transition(BotState.POPUP)
                self.controller.keyevent(4)  # back/esc
                self.sm.transition(BotState.ANALYZING)
                continue

            boxes = self.detector.detect(image)
            if len(boxes) < self.min_tiles_for_action:
                initial_count = len(boxes)
                retry_image = self.controller.screencap()
                retry_boxes = self.detector.detect(retry_image)
                if len(retry_boxes) > len(boxes):
                    image = retry_image
                    boxes = retry_boxes
                self.log.info(
                    "Low tile count frame=%s retry=%s chosen=%s",
                    initial_count,
                    len(retry_boxes),
                    len(boxes),
                )
            if not boxes:
                self.log.info("No tiles detected on current frame.")
                watchdog.mark_no_progress_action()
                no_progress_streak += 1
                if watchdog.is_stalled():
                    self.sm.transition(BotState.RECOVERING)
                    return LevelResult(success=False, reason="no_tiles_stalled", steps=steps)
                time.sleep(0.5)
                continue
            if len(boxes) < self.min_tiles_for_action:
                watchdog.mark_no_progress_action()
                no_progress_streak += 1
                self.log.info(
                    "Skip action due to unstable sparse detection: tiles=%s min_required=%s",
                    len(boxes),
                    self.min_tiles_for_action,
                )
                if watchdog.is_stalled():
                    self.sm.transition(BotState.RECOVERING)
                    return LevelResult(success=False, reason="sparse_tiles_stalled", steps=steps)
                time.sleep(0.3)
                continue
            board_signature = self._boxes_signature(boxes)
            if last_board_signature is None:
                last_board_signature = board_signature
                same_board_cycles = 0
            elif board_signature != last_board_signature:
                last_board_signature = board_signature
                same_board_cycles = 0
            else:
                same_board_cycles += 1
                if same_board_cycles % 5 == 0:
                    self.log.info("Board unchanged cycles=%s", same_board_cycles)
                if same_board_cycles >= self.max_same_board_cycles:
                    self.log.warning("Board unchanged too long: cycles=%s", same_board_cycles)
                    self.sm.transition(BotState.RECOVERING)
                    return LevelResult(success=False, reason="same_board_stalled", steps=steps)

            tiles: list[Tile] = []
            tile_images = []
            for idx, box in enumerate(boxes):
                x, y, w, h = box.x, box.y, box.w, box.h
                tile_images.append(image[y : y + h, x : x + w])
                tiles.append(
                    Tile(
                        id=idx,
                        bbox=(x, y, w, h),
                        center=(x + w // 2, y + h // 2),
                        confidence=box.confidence,
                    )
                )

            self.classifier.reset()
            class_ids = self.classifier.classify(tile_images)
            for tile, class_id in zip(tiles, class_ids):
                tile.class_id = class_id
            tile_image_map = {tile.id: tile_img for tile, tile_img in zip(tiles, tile_images)}

            board = self.analyzer.build_board_state(tiles=tiles, timestamp=time.time())
            pairs = self.matcher.find_pairs(board)
            available_pairs = [
                pair
                for pair in pairs
                if self._pair_key(pair) not in failed_pair_keys
                and not self._pair_matches_failed_centers(pair, failed_pair_centers)
            ]
            action = self.strategy.choose_action(available_pairs, board)
            clickable_count = sum(1 for tile in board.tiles if tile.clickable)
            self.log.info(
                "Board analyzed: tiles=%s clickable=%s pairs=%s available_pairs=%s",
                len(board.tiles),
                clickable_count,
                len(pairs),
                len(available_pairs),
            )
            draw_tiles_debug(image, board.tiles, f"{self.debug_dir}/step_{steps:04d}.png")

            if action is None:
                watchdog.mark_no_progress_action()
                no_progress_streak += 1
                if no_progress_streak >= self.hint_trigger_no_progress_actions:
                    if self._attempt_hint_fallback(pre_signature=board_signature):
                        watchdog.mark_progress()
                        no_progress_streak = 0
                        failed_pair_keys.clear()
                        failed_pair_centers.clear()
                        steps += 1
                        continue
                if watchdog.is_stalled():
                    self.sm.transition(BotState.RECOVERING)
                    return LevelResult(success=False, reason="no_pairs_stalled", steps=steps)
                time.sleep(0.5)
                continue
            pair_key = self._pair_key(action)
            similarity, structure_sim, color_sim, mean_color_dist = self.classifier.pair_similarity(
                tile_image_map[action[0].id],
                tile_image_map[action[1].id],
            )
            if (
                similarity < self.min_pair_similarity
                or structure_sim < self.min_pair_structure_similarity
                or color_sim < self.min_pair_color_similarity
                or mean_color_dist > self.max_pair_mean_color_distance
            ):
                failed_pair_keys.add(pair_key)
                failed_pair_centers.append(self._pair_centers(action))
                watchdog.mark_no_progress_action()
                no_progress_streak += 1
                self.log.info(
                    "Skip pair (sim=%.3f/%.3f struct=%.3f/%.3f color=%.3f/%.3f mean_color=%.1f/%.1f): %s",
                    similarity,
                    self.min_pair_similarity,
                    structure_sim,
                    self.min_pair_structure_similarity,
                    color_sim,
                    self.min_pair_color_similarity,
                    mean_color_dist,
                    self.max_pair_mean_color_distance,
                    pair_key,
                )
                if watchdog.is_stalled():
                    self.sm.transition(BotState.RECOVERING)
                    return LevelResult(success=False, reason="no_pairs_stalled", steps=steps)
                if no_progress_streak >= self.hint_trigger_no_progress_actions:
                    if self._attempt_hint_fallback(pre_signature=board_signature):
                        watchdog.mark_progress()
                        no_progress_streak = 0
                        failed_pair_keys.clear()
                        failed_pair_centers.clear()
                        steps += 1
                continue
            self.log.info(
                "Trying pair: %s <-> %s (sim=%.3f struct=%.3f color=%.3f mean_color=%.1f)",
                action[0].center,
                action[1].center,
                similarity,
                structure_sim,
                color_sim,
                mean_color_dist,
            )

            self.sm.transition(BotState.ACTING)
            self.controller.tap(*action[0].center)
            self.controller.tap(*action[1].center)

            self.sm.transition(BotState.WAIT_ANIMATION)
            time.sleep(self.animation_wait_sec)
            steps += 1

            post_image = self.controller.screencap()
            post_boxes = self.detector.detect(post_image)
            if len(post_boxes) < self.min_tiles_for_action:
                retry_post_image = self.controller.screencap()
                retry_post_boxes = self.detector.detect(retry_post_image)
                if len(retry_post_boxes) > len(post_boxes):
                    post_boxes = retry_post_boxes
            post_signature = self._boxes_signature(post_boxes)
            pair_still_visible = self._has_pair_centers_in_boxes(action, post_boxes)
            if post_signature == board_signature or pair_still_visible:
                failed_pair_keys.add(pair_key)
                failed_pair_centers.append(self._pair_centers(action))
                watchdog.mark_no_progress_action()
                no_progress_streak += 1
                self.log.info(
                    "Pair had no effect, blocked pair: %s (blocked=%s, still_visible=%s).",
                    pair_key,
                    len(failed_pair_keys),
                    int(pair_still_visible),
                )
                if watchdog.is_stalled():
                    self.sm.transition(BotState.RECOVERING)
                    return LevelResult(success=False, reason="no_pairs_stalled", steps=steps)
                if no_progress_streak >= self.hint_trigger_no_progress_actions:
                    if self._attempt_hint_fallback(pre_signature=board_signature):
                        watchdog.mark_progress()
                        no_progress_streak = 0
                        failed_pair_keys.clear()
                        failed_pair_centers.clear()
                        steps += 1
                continue

            watchdog.mark_progress()
            no_progress_streak = 0
            last_board_signature = post_signature
            same_board_cycles = 0

        self.sm.transition(BotState.RECOVERING)
        return LevelResult(success=False, reason="max_steps_reached", steps=steps)

    @staticmethod
    def _pair_key(pair: tuple[Tile, Tile]) -> tuple[tuple[int, int], tuple[int, int]]:
        a, b = pair
        pa = (a.center[0] // 24, a.center[1] // 24)
        pb = (b.center[0] // 24, b.center[1] // 24)
        if pa <= pb:
            return (pa, pb)
        return (pb, pa)

    @staticmethod
    def _boxes_signature(boxes: list) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(sorted((box.x // 8, box.y // 8, box.w // 8, box.h // 8) for box in boxes))

    @staticmethod
    def _has_pair_centers_in_boxes(pair: tuple[Tile, Tile], boxes: list, tolerance: int = 22) -> bool:
        centers = [pair[0].center, pair[1].center]
        for cx, cy in centers:
            for box in boxes:
                bx = box.x + (box.w // 2)
                by = box.y + (box.h // 2)
                if abs(cx - bx) <= tolerance and abs(cy - by) <= tolerance:
                    return True
        return False

    @staticmethod
    def _pair_centers(pair: tuple[Tile, Tile]) -> tuple[tuple[int, int], tuple[int, int]]:
        a, b = pair[0].center, pair[1].center
        if a <= b:
            return (a, b)
        return (b, a)

    @staticmethod
    def _pair_matches_failed_centers(
        pair: tuple[Tile, Tile],
        failed_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
        tolerance: int = 30,
    ) -> bool:
        a, b = LevelRunner._pair_centers(pair)
        for fa, fb in failed_pairs:
            if abs(a[0] - fa[0]) <= tolerance and abs(a[1] - fa[1]) <= tolerance and abs(b[0] - fb[0]) <= tolerance and abs(b[1] - fb[1]) <= tolerance:
                return True
        return False

    def _attempt_hint_fallback(self, pre_signature: tuple[tuple[int, int, int, int], ...]) -> bool:
        self.log.info("Trigger hint fallback via hint button.")
        # Hint may be triggered while state is WAIT_ANIMATION after a previous click.
        # Normalize back to ANALYZING before issuing hint-driven clicks.
        self.sm.transition(BotState.ANALYZING)
        self.controller.keyevent(1001)
        time.sleep(self.hint_wait_sec)
        hint_image = self.controller.screencap()
        hint_boxes = self._detect_hint_red_boxes(hint_image)
        if len(hint_boxes) < 2:
            self.log.info("Hint fallback: red hint boxes not found.")
            return False
        box_a, box_b = hint_boxes[0], hint_boxes[1]
        ax = box_a[0] + (box_a[2] // 2)
        ay = box_a[1] + (box_a[3] // 2)
        bx = box_b[0] + (box_b[2] // 2)
        by = box_b[1] + (box_b[3] // 2)
        self.log.info("Hint fallback tapping red boxes: (%s,%s) and (%s,%s)", ax, ay, bx, by)
        self.sm.transition(BotState.ACTING)
        self.controller.tap(ax, ay)
        self.controller.tap(bx, by)
        self.sm.transition(BotState.WAIT_ANIMATION)
        time.sleep(self.animation_wait_sec)
        post_image = self.controller.screencap()
        post_boxes = self.detector.detect(post_image)
        post_signature = self._boxes_signature(post_boxes)
        changed = post_signature != pre_signature
        self.log.info("Hint fallback result changed=%s", int(changed))
        return changed

    @staticmethod
    def _detect_hint_red_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
        if cv2 is None:
            return []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 90, 90], dtype=np.uint8)
        upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([168, 90, 90], dtype=np.uint8)
        upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 900:
                continue
            if not (28 <= w <= 120 and 36 <= h <= 150):
                continue
            aspect = h / max(1, w)
            if not (0.7 <= aspect <= 2.8):
                continue
            boxes.append((x, y, w, h, area))
        boxes.sort(key=lambda item: item[4], reverse=True)
        dedup: list[tuple[int, int, int, int]] = []
        for x, y, w, h, _ in boxes:
            cx = x + (w // 2)
            cy = y + (h // 2)
            if any(abs(cx - (dx + dw // 2)) <= 18 and abs(cy - (dy + dh // 2)) <= 18 for dx, dy, dw, dh in dedup):
                continue
            dedup.append((x, y, w, h))
        return dedup[:2]

