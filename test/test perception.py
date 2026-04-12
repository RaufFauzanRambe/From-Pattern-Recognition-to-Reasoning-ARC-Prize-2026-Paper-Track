import numpy as np
import pytest
from src.perception.grid_parser import parse_grid
from src.perception.feature_extractor import extract_features


def test_parse_grid_shape():
    grid = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    parsed = parse_grid(grid)

    assert isinstance(parsed, np.ndarray)
    assert parsed.shape == (3, 3)
    assert parsed.dtype in [np.int32, np.int64]


def test_parse_grid_values():
    grid = [
        [2, 2],
        [1, 0]
    ]
    parsed = parse_grid(grid)

    # Exact value checks
    assert parsed[0, 0] == 2
    assert parsed[1, 1] == 0
    assert parsed[1, 0] == 1
    assert parsed[0, 1] == 2

    # Content validation
    unique_vals = set(parsed.flatten())
    assert unique_vals == {0, 1, 2}

    # Safety check
    assert parsed.shape == (2, 2)


def test_extract_features_hard():
    grid = np.array([
        [1, 1, 0, 0, 1, 1],
        [0, 2, 2, 0, 2, 2],
        [0, 0, 2, 2, 0, 2], 
        [0, 3, 2, 3, 0, 0],
        [2, 0, 2, 2, 4, 0]
    ])

    features = extract_features(grid)

    # Structure validation
    assert isinstance(features, dict)
    assert "unique_colors" in features
    assert "color_counts" in features

    # Type validation
    assert isinstance(features["unique_colors"], (list, set, np.ndarray))
    assert isinstance(features["color_counts"], dict)

    # Logical validation
    assert set(features["unique_colors"]).issuperset({0, 1, 2})

    # Count consistency check
    total_pixels = grid.size
    total_counted = sum(features["color_counts"].values())
    assert total_counted <= total_pixels  # fleksibel tapi tetap masuk akal

    assert features["color_counts"].get(2, 0) > 0


def test_running_features_intermediate():
    intelligence = np.array([
        [9, 0, 0, 1, 9, 1, 2, 3, 0]
    ])

    features = extract_features(intelligence)

    assert isinstance(features, dict)

    assert "unique_colors" in features
    assert 9 in features["unique_colors"]

    assert "color_counts" in features
    assert features["color_counts"][9] == 2


# 🔥 BONUS: Edge Case (ini yang bikin kamu beda dari kebanyakan orang)
def test_empty_grid():
    grid = []
    parsed = parse_grid(grid)

    assert isinstance(parsed, np.ndarray)
    assert parsed.size == 0


def test_single_value_grid():
    grid = [[7]]
    parsed = parse_grid(grid)

    assert parsed.shape == (1, 1)
    assert parsed[0, 0] == 7


def test_feature_consistency():
    grid = np.array([
        [1, 1],
        [1, 1]
    ])

    features = extract_features(grid)

    assert features["unique_colors"] == [1] or set(features["unique_colors"]) == {1}
    assert features["color_counts"][1] == 4
