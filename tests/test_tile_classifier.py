import numpy as np

from vision.tile_classifier import TileClassifier


def _make_tile(bg: int, mark: tuple[int, int, int]) -> np.ndarray:
    tile = np.full((72, 56, 3), bg, dtype=np.uint8)
    tile[10:62, 8:48] = mark
    return tile


def test_classifier_separates_visually_different_tiles():
    clf = TileClassifier(similarity_threshold=0.9, min_structure_similarity=0.86, min_color_similarity=0.72)
    tile_red = _make_tile(240, (230, 20, 20))
    tile_blue = _make_tile(240, (20, 20, 230))

    c1 = clf.assign_class(tile_red)
    c2 = clf.assign_class(tile_blue)

    assert c1 != c2


def test_pair_similarity_reports_components():
    clf = TileClassifier()
    a = _make_tile(240, (30, 30, 220))
    b = _make_tile(240, (35, 35, 210))

    sim, structure, color, mean_color_dist = clf.pair_similarity(a, b)

    assert 0.0 <= sim <= 1.0
    assert 0.0 <= structure <= 1.0
    assert 0.0 <= color <= 1.0
    assert mean_color_dist >= 0.0
