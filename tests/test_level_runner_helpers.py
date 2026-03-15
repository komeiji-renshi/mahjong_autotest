import numpy as np

from model.tile import Tile
from runner.level_runner import LevelRunner
from vision.tile_classifier import TileClassifier
from vision.tile_detector import TileBox


def test_pair_key_is_order_independent():
    a = Tile(id=1, bbox=(100, 100, 50, 65), center=(125, 132), class_id=1, clickable=True)
    b = Tile(id=2, bbox=(200, 100, 50, 65), center=(225, 132), class_id=1, clickable=True)

    key_ab = LevelRunner._pair_key((a, b))
    key_ba = LevelRunner._pair_key((b, a))

    assert key_ab == key_ba


def test_boxes_signature_changes_when_tile_count_changes():
    before = [TileBox(x=100, y=100, w=50, h=65, confidence=0.8), TileBox(x=200, y=100, w=50, h=65, confidence=0.8)]
    after = [TileBox(x=200, y=100, w=50, h=65, confidence=0.8)]

    assert LevelRunner._boxes_signature(before) != LevelRunner._boxes_signature(after)


def test_tile_classifier_pair_similarity_prefers_same_pattern():
    a = np.full((64, 48, 3), 240, dtype=np.uint8)
    b = np.full((64, 48, 3), 240, dtype=np.uint8)
    c = np.full((64, 48, 3), 180, dtype=np.uint8)
    a[20:44, 16:32] = (20, 20, 220)
    b[20:44, 16:32] = (25, 25, 210)
    c[20:44, 16:32] = (220, 20, 20)

    classifier = TileClassifier()
    sim_same, _, _, _ = classifier.pair_similarity(a, b)
    sim_diff, _, _, _ = classifier.pair_similarity(a, c)

    assert sim_same > sim_diff


def test_has_pair_centers_in_boxes_detects_visibility():
    a = Tile(id=1, bbox=(100, 100, 50, 65), center=(125, 132), class_id=1, clickable=True)
    b = Tile(id=2, bbox=(200, 100, 50, 65), center=(225, 132), class_id=1, clickable=True)
    boxes = [TileBox(x=118, y=120, w=14, h=16, confidence=0.9)]
    assert LevelRunner._has_pair_centers_in_boxes((a, b), boxes, tolerance=22) is True


def test_has_pair_centers_in_boxes_false_when_removed():
    a = Tile(id=1, bbox=(100, 100, 50, 65), center=(125, 132), class_id=1, clickable=True)
    b = Tile(id=2, bbox=(200, 100, 50, 65), center=(225, 132), class_id=1, clickable=True)
    boxes = [TileBox(x=420, y=320, w=40, h=60, confidence=0.9)]
    assert LevelRunner._has_pair_centers_in_boxes((a, b), boxes, tolerance=22) is False


def test_pair_matches_failed_centers_true_for_nearby_pair():
    a = Tile(id=1, bbox=(100, 100, 50, 65), center=(125, 132), class_id=1, clickable=True)
    b = Tile(id=2, bbox=(200, 100, 50, 65), center=(225, 132), class_id=1, clickable=True)
    failed = [((120, 130), (226, 134))]
    assert LevelRunner._pair_matches_failed_centers((a, b), failed, tolerance=8) is True


def test_pair_matches_failed_centers_false_for_far_pair():
    a = Tile(id=1, bbox=(100, 100, 50, 65), center=(125, 132), class_id=1, clickable=True)
    b = Tile(id=2, bbox=(200, 100, 50, 65), center=(225, 132), class_id=1, clickable=True)
    failed = [((300, 300), (420, 420))]
    assert LevelRunner._pair_matches_failed_centers((a, b), failed, tolerance=12) is False
