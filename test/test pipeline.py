import numpy as np
import pytest
import copy

from src.perception.grid_parser import parse_grid
from src.reasoning.program_synthesis import synthesize_program
from src.reasoning.rule_engine import apply_rules


# =========================================================
# NUMPY-BASED GRID GENERATION UTILITIES
# =========================================================

def generate_grid_from_arange(n: int, shape: tuple) -> np.ndarray:
    """Generate a grid using np.arange and reshape."""
    return np.arange(n).reshape(shape)


def generate_random_grid(shape: tuple, max_val: int = 5, seed: int = 42) -> np.ndarray:
    """Generate a random integer grid with fixed seed for reproducibility."""
    np.random.seed(seed)
    return np.random.randint(0, max_val, size=shape)


# =========================================================
# BASIC NUMPY PIPELINE TEST
# =========================================================

def test_pipeline_with_arange():
    """Test pipeline using a simple arange-generated grid."""
    input_grid = generate_grid_from_arange(4, (2, 2))
    expected_output = np.flipud(input_grid).tolist()

    parsed = parse_grid(input_grid)
    program = synthesize_program(parsed.tolist(), expected_output)
    prediction = apply_rules(parsed.tolist(), program)

    assert prediction == expected_output


# =========================================================
# MULTIPLE NUMPY STRUCTURE TESTS
# =========================================================

@pytest.mark.parametrize("grid", [
    np.array([[1, 2], [3, 4]]),
    np.arange(9).reshape(3, 3),
    np.ones((2, 2), dtype=int),
    np.eye(3, dtype=int),
])
def test_various_numpy_inputs(grid: np.ndarray):
    """Ensure pipeline works across diverse NumPy-generated grids."""
    expected_output = grid[::-1].tolist()

    parsed = parse_grid(grid)
    program = synthesize_program(parsed.tolist(), expected_output)
    prediction = apply_rules(parsed.tolist(), program)

    assert isinstance(prediction, list)
    assert len(prediction) == len(expected_output)


# =========================================================
# TRAIN / TEST SPLIT (ML-STYLE EVALUATION)
# =========================================================

def train_test_split(grids, test_size: float = 0.3):
    """Simple deterministic train-test split."""
    split_idx = int(len(grids) * (1 - test_size))
    return grids[:split_idx], grids[split_idx:]


def test_pipeline_train_test_style():
    """Simulate training and testing phases like an ML pipeline."""
    
    dataset = [
        np.arange(4).reshape(2, 2),
        np.arange(9).reshape(3, 3),
        np.array([[0, 1], [1, 0]]),
        np.ones((2, 2)),
    ]

    train_set, test_set = train_test_split(dataset)

    # TRAIN PHASE
    programs = []
    for grid in train_set:
        expected_output = grid[::-1].tolist()
        parsed = parse_grid(grid)
        program = synthesize_program(parsed.tolist(), expected_output)
        programs.append(program)

    # TEST PHASE
    for grid in test_set:
        parsed = parse_grid(grid)

        for program in programs:
            prediction = apply_rules(parsed.tolist(), program)
            assert isinstance(prediction, list)


# =========================================================
# IMMUTABILITY TEST (NO SIDE EFFECTS)
# =========================================================

def test_numpy_input_immutability():
    """Ensure original NumPy input is not modified."""
    input_grid = np.arange(4).reshape(2, 2)
    original_copy = input_grid.copy()

    parsed = parse_grid(input_grid)
    program = synthesize_program(parsed.tolist(), input_grid.tolist())
    _ = apply_rules(parsed.tolist(), program)

    assert np.array_equal(input_grid, original_copy)


# =========================================================
# SHAPE GENERALIZATION TEST
# =========================================================

@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 4)])
def test_different_shapes(shape: tuple):
    """Ensure pipeline generalizes across different grid sizes."""
    grid = generate_random_grid(shape)
    expected_output = grid[::-1].tolist()

    parsed = parse_grid(grid)
    program = synthesize_program(parsed.tolist(), expected_output)
    prediction = apply_rules(parsed.tolist(), program)

    assert len(prediction) == shape[0]


# =========================================================
# CONSISTENCY / DETERMINISM TEST
# =========================================================

def test_multiple_runs_consistency():
    """Ensure consistent outputs across multiple runs."""
    grid = np.arange(4).reshape(2, 2)
    expected_output = grid[::-1].tolist()

    parsed = parse_grid(grid)

    results = []
    for _ in range(5):
        program = synthesize_program(parsed.tolist(), expected_output)
        prediction = apply_rules(parsed.tolist(), program)
        results.append(prediction)

    assert all(result == results[0] for result in results)


# =========================================================
# RANDOMIZED STRESS TEST
# =========================================================

def test_randomized_grids():
    """Run pipeline on multiple random grids."""
    for _ in range(5):
        grid = generate_random_grid((3, 3))
        expected_output = grid[::-1].tolist()

        parsed = parse_grid(grid)
        program = synthesize_program(parsed.tolist(), expected_output)
        prediction = apply_rules(parsed.tolist(), program)

        assert isinstance(prediction, list)


# =========================================================
# VALUE PRESERVATION TEST
# =========================================================

def test_value_preservation():
    """Ensure transformation does not lose or introduce values."""
    grid = np.arange(9).reshape(3, 3)
    expected_output = grid[::-1].tolist()

    parsed = parse_grid(grid)
    program = synthesize_program(parsed.tolist(), expected_output)
    prediction = apply_rules(parsed.tolist(), program)

    input_values = set(grid.flatten())
    output_values = set(np.array(prediction).flatten())

    assert input_values == output_values
