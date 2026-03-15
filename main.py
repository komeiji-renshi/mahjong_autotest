from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from core.adb_controller import AdbController, DeviceConfig
from core.controller_protocol import Controller
from core.pc_controller import PcConfig, PcController
from core.state_machine import StateMachine
from core.web_controller import WebConfig, WebController
from runner.game_runner import GameRunner
from runner.level_runner import LevelRunner
from solver.pair_matcher import PairMatcher
from solver.strategy import Strategy
from vision.overlap_analyzer import BoardAnalyzer
from vision.tile_classifier import TileClassifier
from vision.tile_detector import TileDetector
from vision.ui_recognizer import UiRecognizer


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def configure_logging(log_cfg: dict) -> None:
    level = log_cfg.get("level", "INFO")
    Path(log_cfg.get("dir", "logs")).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_controller(game_cfg: dict, device_cfg: dict) -> Controller:
    platform = str(device_cfg.get("platform", "adb")).lower()
    if platform == "web":
        startup_click_selectors = tuple(
            str(item)
            for item in device_cfg.get(
                "web_startup_click_selectors",
                [
                    "button.tutorial-btn-primary",
                    "button.tutorial-btn-skip",
                    "div.start-links button",
                ],
            )
        )
        web_cfg = WebConfig(
            url=device_cfg.get("web_url", "https://ffalt.github.io/mah/"),
            browser=device_cfg.get("web_browser", "chromium"),
            headless=bool(device_cfg.get("web_headless", False)),
            viewport_width=int(device_cfg.get("web_viewport_width", 1280)),
            viewport_height=int(device_cfg.get("web_viewport_height", 720)),
            startup_click_selectors=startup_click_selectors,
        )
        return WebController(web_cfg)

    if platform == "pc":
        pc_cfg = PcConfig(
            window_title_keyword=device_cfg.get("window_title_keyword", "Mahjong"),
            app_aumid=device_cfg.get("app_aumid", ""),
            launch_cmd=device_cfg.get("launch_cmd", ""),
            process_name=device_cfg.get("process_name", ""),
        )
        return PcController(pc_cfg)

    device = DeviceConfig(
        serial=device_cfg.get("serial"),
        adb_path=device_cfg.get("adb_path", "adb"),
        package_name=game_cfg.get("package_name", ""),
        activity_name=game_cfg.get("activity_name", ""),
    )
    return AdbController(device)


def build_bot(game_cfg: dict, device_cfg: dict, level_timeout_sec_override: float | None = None) -> GameRunner:
    controller = build_controller(game_cfg=game_cfg, device_cfg=device_cfg)
    raw_size_templates = game_cfg.get("tile_detector", {}).get("size_templates")
    size_templates = None
    if isinstance(raw_size_templates, list):
        parsed_templates: list[tuple[int, int]] = []
        for item in raw_size_templates:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                parsed_templates.append((int(item[0]), int(item[1])))
        size_templates = tuple(parsed_templates) if parsed_templates else None
    detector = TileDetector(
        template_path=game_cfg.get("tile_detector", {}).get("template_path"),
        template_threshold=game_cfg.get("tile_detector", {}).get("template_threshold", 0.84),
        min_tile_w=game_cfg.get("tile_detector", {}).get("min_tile_w", 50),
        max_tile_w=game_cfg.get("tile_detector", {}).get("max_tile_w", 180),
        min_tile_h=game_cfg.get("tile_detector", {}).get("min_tile_h", 60),
        max_tile_h=game_cfg.get("tile_detector", {}).get("max_tile_h", 220),
        size_templates=size_templates,
        size_match_tolerance_ratio=game_cfg.get("tile_detector", {}).get("size_match_tolerance_ratio", 0.30),
        center_dedup_distance=game_cfg.get("tile_detector", {}).get("center_dedup_distance", 16),
        min_white_ratio=game_cfg.get("tile_detector", {}).get("min_white_ratio", 0.30),
        white_sat_max=game_cfg.get("tile_detector", {}).get("white_sat_max", 90),
        white_val_min=game_cfg.get("tile_detector", {}).get("white_val_min", 130),
        playfield_left_ratio=game_cfg.get("tile_detector", {}).get("playfield_left_ratio", 0.12),
        playfield_right_ratio=game_cfg.get("tile_detector", {}).get("playfield_right_ratio", 0.88),
        playfield_top_ratio=game_cfg.get("tile_detector", {}).get("playfield_top_ratio", 0.05),
        playfield_bottom_ratio=game_cfg.get("tile_detector", {}).get("playfield_bottom_ratio", 0.92),
    )
    classifier = TileClassifier(
        similarity_threshold=game_cfg.get("classifier", {}).get("similarity_threshold", 0.9),
        min_structure_similarity=game_cfg.get("classifier", {}).get("min_structure_similarity", 0.86),
        min_color_similarity=game_cfg.get("classifier", {}).get("min_color_similarity", 0.72),
        structure_weight=game_cfg.get("classifier", {}).get("structure_weight", 0.7),
        color_weight=game_cfg.get("classifier", {}).get("color_weight", 0.3),
        color_hist_bins=game_cfg.get("classifier", {}).get("color_hist_bins", 16),
        max_mean_color_distance=game_cfg.get("classifier", {}).get("max_mean_color_distance", 26.0),
    )
    analyzer = BoardAnalyzer(
        overlap_ratio_threshold=game_cfg.get("board", {}).get("overlap_ratio_threshold", 0.18),
        side_margin_ratio=game_cfg.get("board", {}).get("side_margin_ratio", 0.35),
        same_layer_y_ratio=game_cfg.get("board", {}).get("same_layer_y_ratio", 0.35),
        min_side_y_overlap_ratio=game_cfg.get("board", {}).get("min_side_y_overlap_ratio", 0.45),
    )
    matcher = PairMatcher(max_pairs_per_class=game_cfg.get("matcher", {}).get("max_pairs_per_class", 24))
    strategy = Strategy()
    ui = UiRecognizer(templates=game_cfg.get("ui_templates", {}), threshold=game_cfg.get("ui_threshold", 0.82))
    state_machine = StateMachine()
    level_runner = LevelRunner(
        controller=controller,
        detector=detector,
        classifier=classifier,
        analyzer=analyzer,
        matcher=matcher,
        strategy=strategy,
        ui=ui,
        state_machine=state_machine,
        debug_dir=game_cfg.get("debug_dir", "logs/debug"),
        animation_wait_sec=game_cfg.get("runtime", {}).get("animation_wait_sec", 0.6),
        level_timeout_sec=(
            level_timeout_sec_override
            if level_timeout_sec_override is not None
            else game_cfg.get("runtime", {}).get("level_timeout_sec", 90.0)
        ),
        max_same_board_cycles=game_cfg.get("runtime", {}).get("max_same_board_cycles", 40),
        min_pair_similarity=game_cfg.get("runtime", {}).get("min_pair_similarity", 0.90),
        min_pair_structure_similarity=game_cfg.get("runtime", {}).get("min_pair_structure_similarity", 0.92),
        min_pair_color_similarity=game_cfg.get("runtime", {}).get("min_pair_color_similarity", 0.86),
        max_pair_mean_color_distance=game_cfg.get("runtime", {}).get("max_pair_mean_color_distance", 26.0),
        min_tiles_for_action=game_cfg.get("runtime", {}).get("min_tiles_for_action", 20),
        watchdog_timeout_sec=game_cfg.get("runtime", {}).get("watchdog_timeout_sec", 30.0),
        watchdog_no_progress_actions=game_cfg.get("runtime", {}).get("watchdog_no_progress_actions", 30),
    )
    return GameRunner(controller=controller, level_runner=level_runner, state_machine=state_machine)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vita Mahjong automation runner")
    parser.add_argument("--levels", type=int, default=10, help="How many levels to run")
    parser.add_argument("--config-dir", default="config", help="Directory containing yaml configs")
    parser.add_argument(
        "--level-timeout-sec",
        type=float,
        default=None,
        help="Per-level hard timeout in seconds (override config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    game_cfg = load_yaml(str(config_dir / "game.yaml"))
    device_cfg = load_yaml(str(config_dir / "device.yaml"))
    log_cfg = load_yaml(str(config_dir / "log.yaml"))
    configure_logging(log_cfg)

    bot = build_bot(
        game_cfg=game_cfg,
        device_cfg=device_cfg,
        level_timeout_sec_override=args.level_timeout_sec,
    )
    summary = bot.run(max_levels=args.levels)
    logging.getLogger("vita_mahjong_bot").info(
        "Finished run: total=%s success=%s fail=%s",
        summary.total_levels,
        summary.success_levels,
        summary.fail_levels,
    )


if __name__ == "__main__":
    main()
