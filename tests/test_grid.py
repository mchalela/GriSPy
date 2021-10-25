#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


import numpy as np
import numpy.testing as npt
import pytest

from grispy import Grid

# =========================================================
# INITIALIZATION
# =========================================================


def test_grid_copy_data(grid_input):
    data = grid_input["data"]
    grid = Grid(data, copy_data=False)
    assert grid.data is data

    grid_copy = Grid(data, copy_data=True)
    assert grid_copy.data is not data


def test_grid_arguments(grid):
    assert isinstance(grid.data, np.ndarray)
    assert isinstance(grid.N_cells, int)
    assert isinstance(grid.copy_data, bool)


def test_grid_attrs_post_init(grid):
    assert isinstance(grid.k_bins_, np.ndarray)
    assert isinstance(grid.grid_, dict)


def test_invalid_data_inf():
    bad_data = np.random.uniform(-1, 1, size=(100, 3))
    bad_data[42, 1] = np.inf  # add one invalid value
    with pytest.raises(ValueError):
        Grid(bad_data)


def test_invalid_data_type():
    # Data type
    bad_data = 4
    with pytest.raises(TypeError):
        Grid(data=bad_data)


def test_invalid_data_empty_array():
    # Data format
    bad_data = np.array([])
    with pytest.raises(ValueError):
        Grid(data=bad_data)

    # Data format with shape
    bad_data = np.array([[]])
    with pytest.raises(ValueError):
        Grid(data=bad_data)


def test_invalid_data_format():
    # Data format
    bad_data = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        Grid(data=bad_data)


def test_invalid_Ncells(grid_input):
    # N_cells is not an integer
    bad_N_cells = 10.5
    with pytest.raises(TypeError):
        Grid(grid_input["data"], N_cells=bad_N_cells)

    # N_cells is not positive
    bad_N_cells = -10
    with pytest.raises(ValueError):
        Grid(grid_input["data"], N_cells=bad_N_cells)


def test_invalid_copy_data(grid_input):
    # copy_data is not boolean
    bad_copy_data = 42
    with pytest.raises(TypeError):
        Grid(grid_input["data"], copy_data=bad_copy_data)


def test_invalid_points_type(grid):
    # Invalid type
    bad_points = [[1, 1, 1], [2, 2, 2]]
    with pytest.raises(TypeError):
        grid.contains(bad_points)

    with pytest.raises(TypeError):
        grid.cell_id(bad_points)

    with pytest.raises(TypeError):
        grid.cell_digits(bad_points)


def test_invalid_points_single_value_type(grid):
    rng = np.random.default_rng(88)
    bad_points = rng.uniform(-1, 1, size=(10, 3))
    bad_points[4, 1] = np.inf  # add one invalid value
    with pytest.raises(ValueError):
        grid.contains(bad_points)

    with pytest.raises(ValueError):
        grid.cell_id(bad_points)

    with pytest.raises(ValueError):
        grid.cell_digits(bad_points)


def test_invalid_points_shape(grid):
    rng = np.random.default_rng(88)
    # Invalid shape
    bad_points = rng.uniform(-1, 1, size=(10, 2))
    with pytest.raises(ValueError):
        grid.contains(bad_points)

    with pytest.raises(ValueError):
        grid.cell_id(bad_points)

    with pytest.raises(ValueError):
        grid.cell_digits(bad_points)


def test_invalid_points_empty(grid):
    # Invalid shape
    bad_points = np.array([[], [], []]).reshape((0, 3))
    with pytest.raises(ValueError):
        grid.contains(bad_points)

    with pytest.raises(ValueError):
        grid.cell_id(bad_points)

    with pytest.raises(ValueError):
        grid.cell_digits(bad_points)


def test_invalid_id(grid):
    bad_ids = np.array([-1, 0, 1, 2, 3])
    with pytest.raises(ValueError):
        grid.cell_id2digits(bad_ids)

    large_id = grid.size
    bad_ids = np.array([0, 1, 2, 3, large_id])
    with pytest.raises(ValueError):
        grid.cell_id2digits(bad_ids)


def test_invalid_digits(grid):
    ids = np.array([0, 1, 2, 3])
    bad_digits = grid.cell_id2digits(ids)
    bad_digits[0, 0] = -1
    with pytest.raises(ValueError):
        grid.cell_digits2id(bad_digits)

    ids = np.array([0, 1, 2, 3])
    bad_digits = grid.cell_id2digits(ids)
    bad_digits[0, 0] = grid.N_cells
    with pytest.raises(ValueError):
        grid.cell_digits2id(bad_digits)


# =========================================================
# PROPERTIES
# =========================================================


def test_grid_properties_dim(grid):
    assert isinstance(grid.dim, int)
    assert grid.dim == grid.data.shape[1]


def test_grid_properties_edges(grid):
    assert isinstance(grid.edges, np.ndarray)
    assert grid.edges.shape == (2, grid.dim)


def test_grid_properties_epsilon(grid):
    assert isinstance(grid.epsilon, float)
    assert 0 < grid.epsilon <= 0.1


def test_grid_properties_ndata(grid):
    assert isinstance(grid.ndata, int)
    assert grid.ndata == len(grid.data)


def test_grid_properties_shape(grid):
    assert isinstance(grid.shape, tuple)
    assert grid.shape == (grid.N_cells,) * grid.dim


def test_grid_properties_size(grid):
    assert isinstance(grid.size, int)
    assert grid.size == np.prod(grid.shape)


def test_grid_properties_cell_width(grid):
    assert isinstance(grid.cell_width, np.ndarray)
    assert grid.cell_width.dtype == float
    assert (grid.cell_width > 0).all()


# =========================================================
# METHODS
# =========================================================


def test_contains(grid, grid_input):
    result = grid.contains(grid_input["points"])
    assert isinstance(result, np.ndarray)
    assert len(result) == len(grid_input["points"])
    assert result.shape == (len(grid_input["points"]),)
    assert result.dtype == bool


def test_cell_digits(grid, grid_input):
    result = grid.cell_digits(grid_input["points"])
    assert isinstance(result, np.ndarray)
    assert len(result) == len(grid_input["points"])
    assert result.shape == grid_input["points"].shape
    assert result.dtype == np.int16


def test_cell_id(grid, grid_input):
    result = grid.cell_id(grid_input["points"])
    assert isinstance(result, np.ndarray)
    assert len(result) == len(grid_input["points"])
    assert result.shape == (len(grid_input["points"]),)
    assert result.dtype == int


def test_cell_digits2id(grid, grid_input):
    digits = grid.cell_digits(grid_input["points"])

    result = grid.cell_digits2id(digits)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(grid_input["points"])
    assert result.shape == (len(grid_input["points"]),)
    assert result.dtype == int


def test_cell_id2digits(grid, grid_input):
    ids = grid.cell_id(grid_input["points"])

    result = grid.cell_id2digits(ids)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(grid_input["points"])
    assert result.shape == grid_input["points"].shape
    assert result.dtype == np.int16


def test_cell_walls(grid, grid_input):
    digits = grid.cell_digits(grid_input["points"])

    result = grid.cell_walls(digits)
    assert isinstance(result, tuple)
    assert len(result) == 2

    lower, upper = result
    assert isinstance(lower, np.ndarray)
    assert isinstance(upper, np.ndarray)
    assert lower.shape == (len(grid_input["points"]), grid.dim)
    assert upper.shape == (len(grid_input["points"]), grid.dim)
    assert lower.dtype == float
    assert upper.dtype == float
    assert (upper - lower > 0).all()


def test_cell_centre(grid, grid_input):
    digits = grid.cell_digits(grid_input["points"])

    result = grid.cell_centre(digits)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(grid_input["points"]), grid.dim)
    assert result.dtype == float


def test_cell_centre_within_walls(grid, grid_input):
    digits = grid.cell_digits(grid_input["points"])

    centres = grid.cell_centre(digits)
    lower, upper = grid.cell_walls(digits)
    assert (upper - centres > 0).all()
    assert (centres - lower > 0).all()


def test_cell_count(grid, grid_input):
    digits = grid.cell_digits(grid_input["points"])

    result = grid.cell_count(digits)
    assert isinstance(result, np.ndarray)
    assert result.dtype == int
    assert (result >= 0).all()


def test_cell_points(grid, grid_input):
    digits = grid.cell_digits(grid_input["points"])

    result = grid.cell_points(digits)
    assert isinstance(result, list)
    assert len(result) == len(digits)
    for points in result:
        assert isinstance(points, tuple)


# =========================================================================
# OUTPUT VALIDATION
# =========================================================================


def test_grid_property_edges(grid):
    dmin = grid.data.min(axis=0) - grid.epsilon
    dmax = grid.data.max(axis=0) + grid.epsilon
    k_bins = np.linspace(dmin, dmax, grid.N_cells + 1)
    expected = k_bins[[0, -1], :]
    npt.assert_almost_equal(grid.edges, expected, 14)


@pytest.mark.parametrize(
    "pname", ["inside_points", "outside_points", "mix_points"]
)
def test_grid_contains_inside(grid, grid_input, pname):
    points = grid_input[pname]
    lower = 0 < points
    upper = points < 1
    expected = (lower & upper).prod(axis=1, dtype=bool)
    result = grid.contains(points)
    npt.assert_equal(expected, result)


def test_cell_digits_result(grid, grid_input):
    expected = np.array(
        [
            [2, 1, 2],
            [0, 0, 0],
            [0, 0, 2],
            [0, 1, 1],
            [2, 2, 2],
            [1, 2, 0],
            [0, 2, 0],
            [2, 2, 1],
            [0, 2, 1],
            [-1, -1, -1],
        ]
    )
    result = grid.cell_digits(grid_input["data"])
    npt.assert_equal(expected, result)


def test_cell_digits_outside(grid, grid_input):
    expected = np.array(
        [
            [2, 1, 2],
            [0, 0, 0],
            [0, 0, 2],
            [0, 1, 1],
            [2, 2, 2],
            [1, 2, 0],
            [0, 2, 0],
            [2, 2, 1],
            [0, 2, 1],
            [-1, -1, -1],
        ]
    )
    result = grid.cell_digits(grid_input["data"])
    npt.assert_equal(expected, result)


def test_cell_id_result(grid, grid_input):
    expected = np.array([23, 0, 18, 12, 26, 7, 6, 17, 15, -1])
    result = grid.cell_id(grid_input["data"])
    npt.assert_equal(expected, result)


def test_cell_id_outside(grid, grid_input):
    expected = -np.ones(len(grid_input["outside_points"]))
    result = grid.cell_id(grid_input["outside_points"])
    npt.assert_equal(expected, result)


def test_cell_digits2id_result(grid):
    digits = np.array(
        [
            [0, 0, 0],
            [0, 0, 2],
            [0, 2, 0],
            [0, 2, 2],
            [2, 0, 0],
            [2, 0, 2],
            [2, 2, 0],
            [2, 2, 2],
        ]
    )
    expected = np.array([0, 18, 6, 24, 2, 20, 8, 26])
    result = grid.cell_digits2id(digits)
    npt.assert_equal(expected, result)


def test_cell_id2digits_result(grid):
    ids = np.array([0, 18, 6, 24, 2, 20, 8, 26])
    expected = np.array(
        [
            [0, 0, 0],
            [0, 0, 2],
            [0, 2, 0],
            [0, 2, 2],
            [2, 0, 0],
            [2, 0, 2],
            [2, 2, 0],
            [2, 2, 2],
        ]
    )
    result = grid.cell_id2digits(ids)
    npt.assert_equal(expected, result)
