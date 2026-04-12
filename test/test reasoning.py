import pytest
import numpy as np
import copy

from src.reasoning.rule_engine import apply_rules
from src.reasoning.program_synthesis import synthesize_program


# =========================================================
# NUMPY GRID TESTS (REALISTIC + STRICT VALIDATION)
# =========================================================

@pytest.mark.parametrize("grid,rules,expected", [
    (
        np.array([[0, 1, 0], [3, 0, 2], [4, 0, 1]]),
        ["horizontal_flip"],
        np.array([[0, 1, 0], [2, 0, 3], [1, 0, 4]])
    ),
    (
        np.array([[0, 1, 0, 3, 0, 2, 4]]),
        ["horizontal_flip"],
        np.array([[4, 2, 0, 3, 0, 1, 0]])
    ),
])
def test_apply_rules_numpy(grid, rules, expected):
    """
    Ensure rule engine correctly transforms NumPy grids
    without mutating the original input.
    """
    original = copy.deepcopy(grid)

    output = apply_rules(grid, rules)

    # Type and shape checks
    assert isinstance(output, np.ndarray)
    assert output.shape == expected.shape

    # Value correctness
    assert np.array_equal(output, expected)

    # Immutability check
    assert np.array_equal(grid, original), "Input grid must remain unchanged"


def test_apply_rules_deterministic():
    """
    Same input should always produce the same output.
    """
    grid = np.array([[1, 2], [3, 4]])
    rules = ["horizontal_flip"]

    out1 = apply_rules(grid, rules)
    out2 = apply_rules(grid, rules)

    assert np.array_equal(out1, out2)


# =========================================================
# PROGRAM SYNTHESIS (NUMPY SUPPORT + VALIDATION)
# =========================================================

def test_program_synthesis_numpy():
    """
    Synthesized program should correctly transform input → output.
    """
    input_grid = np.array([[1, 0], [0, 1]])
    output_grid = np.array([[0, 1], [1, 0]])

    program = synthesize_program(input_grid, output_grid)

    assert program is not None
    assert isinstance(program, list)
    assert all(isinstance(rule, str) for rule in program)

    result = apply_rules(input_grid, program)

    assert np.array_equal(result, output_grid)


def test_program_synthesis_minimal_identity():
    """
    Identity transformation should return minimal program.
    """
    grid = np.array([[5, 6], [7, 8]])

    program = synthesize_program(grid, grid)

    assert program == [] or program == ["identity"]


def test_program_synthesis_unsolvable_case():
    """
    Clearly impossible mappings should fail gracefully.
    """
    input_grid = np.array([[1]])
    output_grid = np.array([[999]])

    program = synthesize_program(input_grid, output_grid)

    assert program is None or program == []


# =========================================================
# STRING / CONVERSATION TEST (BRYAN CASE)
# =========================================================

def test_bryan_conversation():
    """
    Test rule-based transformation on natural language input.
    """
    input_text = "Hello how i can help you"
    expected_output = "Hello, how can I help you?"

    program = synthesize_program(input_text, expected_output)

    assert program is not None
    assert isinstance(program, list)

    result = apply_rules(input_text, program)

    assert result == expected_output


# =========================================================
# EDGE CASES (NUMERIC + STRUCTURAL)
# =========================================================

def test_1d_array_flip():
    """
    Ensure 1D NumPy arrays are handled correctly.
    """
    grid = np.array([0, 1, 0, 3, 0, 2, 4])

    flipped = apply_rules(grid, ["horizontal_flip"])

    assert isinstance(flipped, np.ndarray)
    assert np.array_equal(flipped, np.array([4, 2, 0, 3, 0, 1, 0]))


def test_large_grid_consistency():
    """
    Double flip should return original grid.
    """
    grid = np.arange(100).reshape(10, 10)

    flipped = apply_rules(grid, ["horizontal_flip"])
    double_flipped = apply_rules(flipped, ["horizontal_flip"])

    assert np.array_equal(double_flipped, grid)


def test_invalid_rule_raises():
    """
    Unknown rules should raise a clear error.
    """
    grid = np.array([[1]])

    with pytest.raises(ValueError):
        apply_rules(grid, ["non_existent_rule"])
