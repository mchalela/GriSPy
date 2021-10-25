import numpy as np
import pytest

from grispy import Grid, GriSPy

# =========================================================================
# Grid fixtures
# =========================================================================


@pytest.fixture
def grid():
    rng = np.random.default_rng(4321)
    data = rng.uniform(0, 1, size=(500, 3))
    return Grid(data, 3)


@pytest.fixture
def grid_input():
    rng = np.random.default_rng(1234)
    d = dict()
    # Define valid input data
    d["data"] = rng.random((10, 3))
    d["points"] = rng.uniform(0.3, 0.7, size=(10, 3))
    d["inside_points"] = rng.uniform(0.3, 0.7, size=(10, 3))
    d["outside_points"] = rng.uniform(10, 11, size=(10, 3))
    d["mix_points"] = np.vstack((d["inside_points"], d["outside_points"]))
    return d


# =========================================================================
# GriSPy fixtures
# =========================================================================


@pytest.fixture
def gsp():
    data = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    return GriSPy(data)


@pytest.fixture
def gsp_periodic():
    data = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    periodic = {0: (0.0, 1.0)}
    return GriSPy(data, periodic=periodic)


@pytest.fixture
def grispy_init():
    # Define valid input data
    rng = np.random.default_rng(seed=42)
    d = dict()
    d["data"] = rng.uniform(-1, 1, size=(100, 3))
    d["periodic"] = {0: (-1, 1), 1: (-1, 1), 2: None}
    d["metric"] = "euclid"
    d["N_cells"] = 10
    d["copy_data"] = True
    return d


@pytest.fixture
def grispy_input():
    rng = np.random.default_rng(1234)
    d = dict()
    # Define valid input data
    d["centres"] = rng.random((5, 3))
    d["upper_radii"] = 0.7
    d["lower_radii"] = 0.5
    d["n_nearest"] = 5
    d["kind"] = "quicksort"
    d["sorted"] = True
    return d
