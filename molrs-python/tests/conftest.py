import numpy as np
import pytest
import molrs


@pytest.fixture
def cubic_box():
    """A 10x10x10 cubic box."""
    return molrs.Box.cube(10.0)


@pytest.fixture
def ortho_box():
    """A 5x10x15 orthorhombic box."""
    return molrs.Box.ortho(np.array([5.0, 10.0, 15.0], dtype=np.float32))


@pytest.fixture
def sample_points():
    """5 random points inside a 10x10x10 box."""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
            [9.0, 8.0, 7.0],
            [0.1, 0.1, 0.1],
            [9.9, 9.9, 9.9],
        ],
        dtype=np.float32,
    )
