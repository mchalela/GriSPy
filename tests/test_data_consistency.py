#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


import numpy as np
import pytest

from grispy import Grid, GriSPy
from grispy.core import PeriodicityConf

# =========================================================================
# Test Grid class
# =========================================================================


class Test_Grid_data_consistency:
    """Test that input and output params types.

    - Check that input params are stored in their corresponding attributes
    and have the expected type.
    - Check that output params have the expected types.
    """

    @pytest.fixture
    def grid(self):
        rng = np.random.default_rng(4321)
        data = rng.uniform(0, 1, size=(500, 3))
        return Grid(data, 6)

    @pytest.fixture
    def valid_input(self):
        rng = np.random.default_rng(1234)
        d = dict()
        # Define valid input data
        d["points"] = rng.uniform(0.3, 0.7, size=(10, 3))
        d["inside_points"] = rng.uniform(0.3, 0.7, size=(10, 3))
        d["outside_points"] = rng.uniform(10, 11, size=(10, 3))
        return d

    def test_grid_copy_data(self):
        rng = np.random.default_rng(11)
        data = rng.random((10, 4))
        grid = Grid(data, copy_data=False)
        assert grid.data is data

        grid_copy = Grid(data, copy_data=True)
        assert grid_copy.data is not data

    def test_grid_arguments(self, grid):
        assert isinstance(grid.data, np.ndarray)
        assert isinstance(grid.N_cells, int)
        assert isinstance(grid.copy_data, bool)

    def test_grid_attrs_post_init(self, grid):
        assert isinstance(grid.k_bins_, np.ndarray)
        assert isinstance(grid.grid_, dict)

    # =========================================================
    # PROPERTIES
    # =========================================================

    def test_grid_properties_dim(self, grid):
        assert isinstance(grid.dim, int)
        assert grid.dim == grid.data.shape[1]

    def test_grid_properties_edges(self, grid):
        assert isinstance(grid.edges, np.ndarray)
        assert grid.edges.shape == (2, grid.dim)

    def test_grid_properties_epsilon(self, grid):
        assert isinstance(grid.epsilon, float)
        assert 0 < grid.epsilon <= 0.1

    def test_grid_properties_ndata(self, grid):
        assert isinstance(grid.ndata, int)
        assert grid.ndata == len(grid.data)

    def test_grid_properties_shape(self, grid):
        assert isinstance(grid.shape, tuple)
        assert grid.shape == (grid.N_cells,) * grid.dim

    def test_grid_properties_size(self, grid):
        assert isinstance(grid.size, int)
        assert grid.size == np.prod(grid.shape)

    def test_grid_properties_cell_width(self, grid):
        assert isinstance(grid.cell_width, np.ndarray)
        assert grid.cell_width.dtype == float
        assert (grid.cell_width > 0).all()

    # =========================================================
    # METHODS
    # =========================================================

    def test_contains(self, grid, valid_input):
        result = grid.contains(valid_input["points"])
        assert isinstance(result, np.ndarray)
        assert len(result) == len(valid_input["points"])
        assert result.shape == (len(valid_input["points"]),)
        assert result.dtype == bool

    def test_cell_digits(self, grid, valid_input):
        result = grid.cell_digits(valid_input["points"])
        assert isinstance(result, np.ndarray)
        assert len(result) == len(valid_input["points"])
        assert result.shape == valid_input["points"].shape
        assert result.dtype == np.int16

    def test_cell_id(self, grid, valid_input):
        result = grid.cell_id(valid_input["points"])
        assert isinstance(result, np.ndarray)
        assert len(result) == len(valid_input["points"])
        assert result.shape == (len(valid_input["points"]),)
        assert result.dtype == int

    def test_cell_digits2id(self, grid, valid_input):
        digits = grid.cell_digits(valid_input["points"])

        result = grid.cell_digits2id(digits)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(valid_input["points"])
        assert result.shape == (len(valid_input["points"]),)
        assert result.dtype == int

    def test_cell_id2digits(self, grid, valid_input):
        ids = grid.cell_id(valid_input["points"])

        result = grid.cell_id2digits(ids)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(valid_input["points"])
        assert result.shape == valid_input["points"].shape
        assert result.dtype == np.int16

    def test_cell_walls(self, grid, valid_input):
        digits = grid.cell_digits(valid_input["points"])

        result = grid.cell_walls(digits)
        assert isinstance(result, tuple)
        assert len(result) == 2

        lower, upper = result
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert lower.shape == (len(valid_input["points"]), grid.dim)
        assert upper.shape == (len(valid_input["points"]), grid.dim)
        assert lower.dtype == float
        assert upper.dtype == float
        assert (upper - lower > 0).all()

    def test_cell_centre(self, grid, valid_input):
        digits = grid.cell_digits(valid_input["points"])

        result = grid.cell_centre(digits)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(valid_input["points"]), grid.dim)
        assert result.dtype == float

    def test_cell_centre_within_walls(self, grid, valid_input):
        digits = grid.cell_digits(valid_input["points"])

        centres = grid.cell_centre(digits)
        lower, upper = grid.cell_walls(digits)
        assert (upper - centres > 0).all()
        assert (centres - lower > 0).all()

    def test_cell_count(self, grid, valid_input):
        digits = grid.cell_digits(valid_input["points"])

        result = grid.cell_count(digits)
        assert isinstance(result, np.ndarray)
        assert result.dtype == int
        assert (result > 0).all()

    def test_cell_points(self, grid, valid_input):
        digits = grid.cell_digits(valid_input["points"])

        result = grid.cell_points(digits)
        assert isinstance(result, list)
        assert len(result) == len(digits)
        for points in result:
            assert isinstance(points, tuple)


class Test_Grid_valid_init:
    """Test that input and output params types.

    - Check that input params are stored in their corresponding attributes
    and have the expected type.
    - Check that output params have the expected types.
    """

    @pytest.fixture
    def grid(self):
        rng = np.random.default_rng(909)
        data = rng.random((10, 3))
        return Grid(data)

    @pytest.fixture
    def valid_input(self):
        rng = np.random.default_rng(1234)
        d = dict()
        # Define valid input data
        d["data"] = rng.random((10, 3))
        d["points"] = rng.random((5, 3))
        d["inside_points"] = rng.random((5, 3))
        d["outside_points"] = rng.random((5, 3)) + 10
        return d

    def test_invalid_data_inf(self):
        bad_data = np.random.uniform(-1, 1, size=(100, 3))
        bad_data[42, 1] = np.inf  # add one invalid value
        with pytest.raises(ValueError):
            Grid(bad_data)

    def test_invalid_data_type(self):
        # Data type
        bad_data = 4
        with pytest.raises(TypeError):
            Grid(data=bad_data)

    def test_invalid_data_empty_array(self):
        # Data format
        bad_data = np.array([])
        with pytest.raises(ValueError):
            Grid(data=bad_data)

        # Data format with shape
        bad_data = np.array([[]])
        with pytest.raises(ValueError):
            Grid(data=bad_data)

    def test_invalid_data_format(self):
        # Data format
        bad_data = np.array([1, 1, 1])
        with pytest.raises(ValueError):
            Grid(data=bad_data)

    def test_invalid_Ncells(self, valid_input):
        # N_cells is not an integer
        bad_N_cells = 10.5
        with pytest.raises(TypeError):
            Grid(valid_input["data"], N_cells=bad_N_cells)

        # N_cells is not positive
        bad_N_cells = -10
        with pytest.raises(ValueError):
            Grid(valid_input["data"], N_cells=bad_N_cells)

    def test_invalid_copy_data(self, valid_input):
        # copy_data is not boolean
        bad_copy_data = 42
        with pytest.raises(TypeError):
            Grid(valid_input["data"], copy_data=bad_copy_data)

    def test_invalid_points_type(self, grid):
        # Invalid type
        bad_points = [[1, 1, 1], [2, 2, 2]]
        with pytest.raises(TypeError):
            grid.contains(bad_points)

        with pytest.raises(TypeError):
            grid.cell_id(bad_points)

        with pytest.raises(TypeError):
            grid.cell_digits(bad_points)

    def test_invalid_points_single_value_type(self, grid):
        rng = np.random.default_rng(88)
        bad_points = rng.uniform(-1, 1, size=(10, 3))
        bad_points[4, 1] = np.inf  # add one invalid value
        with pytest.raises(ValueError):
            grid.contains(bad_points)

        with pytest.raises(ValueError):
            grid.cell_id(bad_points)

        with pytest.raises(ValueError):
            grid.cell_digits(bad_points)

    def test_invalid_points_shape(self, grid):
        rng = np.random.default_rng(88)
        # Invalid shape
        bad_points = rng.uniform(-1, 1, size=(10, 2))
        with pytest.raises(ValueError):
            grid.contains(bad_points)

        with pytest.raises(ValueError):
            grid.cell_id(bad_points)

        with pytest.raises(ValueError):
            grid.cell_digits(bad_points)

    def test_invalid_points_empty(self, grid):
        # Invalid shape
        bad_points = np.array([[], [], []]).reshape((0, 3))
        with pytest.raises(ValueError):
            grid.contains(bad_points)

        with pytest.raises(ValueError):
            grid.cell_id(bad_points)

        with pytest.raises(ValueError):
            grid.cell_digits(bad_points)

    def test_invalid_id(self, grid):
        bad_ids = np.array([-1, 0, 1, 2, 3])
        with pytest.raises(ValueError):
            grid.cell_id2digits(bad_ids)

        large_id = grid.size
        bad_ids = np.array([0, 1, 2, 3, large_id])
        with pytest.raises(ValueError):
            grid.cell_id2digits(bad_ids)

    def test_invalid_digits(self, grid):
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


# =========================================================================
# Test GriSPy class
# =========================================================================


class Test_GriSPy_data_consistency:
    """Test that input and output params types.

    - Check that input params are stored in their corresponding attributes
    and have the expected type.
    - Check that output params have the expected types.
    """

    @pytest.fixture
    def gsp(self):
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
    def valid_input(self):
        rng = np.random.default_rng(1234)
        d = dict()
        # Define valid input data
        d["centres"] = rng.random((5, 3))
        d["upper_radii"] = 0.7
        d["lower_radii"] = 0.5
        d["n_nearest"] = 5
        return d

    def test_grispy_arguments(self, gsp):
        assert isinstance(gsp.data, np.ndarray)
        assert isinstance(gsp.metric, str)
        assert isinstance(gsp.N_cells, int)
        assert isinstance(gsp.periodic, dict)
        assert isinstance(gsp.copy_data, bool)

    def test_grispy_attrs(self, gsp):
        assert isinstance(gsp.k_bins_, np.ndarray)
        assert isinstance(gsp.grid_, dict)
        assert isinstance(gsp.dim, int)
        assert isinstance(gsp.periodic_flag_, bool)
        assert isinstance(gsp.periodic_conf_, PeriodicityConf)

    def test_bubble_single_query(self, gsp, valid_input):

        b, ind = gsp.bubble_neighbors(
            np.array([[0, 0, 0]]),
            distance_upper_bound=valid_input["upper_radii"],
        )
        assert isinstance(b, list)
        assert isinstance(ind, list)
        assert len(b) == len(ind)
        assert len(b) == 1
        assert len(ind) == 1

    def test_shell_single_query(self, gsp, valid_input):

        b, ind = gsp.shell_neighbors(
            np.array([[0, 0, 0]]),
            distance_lower_bound=valid_input["lower_radii"],
            distance_upper_bound=valid_input["upper_radii"],
        )
        assert isinstance(b, list)
        assert isinstance(ind, list)
        assert len(b) == len(ind)
        assert len(b) == 1
        assert len(ind) == 1

    def test_nearest_neighbors_single_query(self, gsp, valid_input):

        b, ind = gsp.nearest_neighbors(
            np.array([[0, 0, 0]]), n=valid_input["n_nearest"]
        )
        assert isinstance(b, list)
        assert isinstance(ind, list)
        assert len(b) == len(ind)
        assert len(b) == 1
        assert len(ind) == 1
        assert np.shape(b[0]) == (valid_input["n_nearest"],)
        assert np.shape(ind[0]) == (valid_input["n_nearest"],)

    def test_bubble_multiple_query(self, gsp, valid_input):

        b, ind = gsp.bubble_neighbors(
            valid_input["centres"],
            distance_upper_bound=valid_input["upper_radii"],
        )
        assert isinstance(b, list)
        assert isinstance(ind, list)
        assert len(b) == len(ind)
        assert len(b) == len(valid_input["centres"])
        assert len(ind) == len(valid_input["centres"])

    def test_shell_multiple_query(self, gsp, valid_input):

        b, ind = gsp.shell_neighbors(
            valid_input["centres"],
            distance_lower_bound=valid_input["lower_radii"],
            distance_upper_bound=valid_input["upper_radii"],
        )
        assert isinstance(b, list)
        assert isinstance(ind, list)
        assert len(b) == len(ind)
        assert len(b) == len(valid_input["centres"])
        assert len(ind) == len(valid_input["centres"])

    def test_nearest_neighbors_multiple_query(self, gsp, valid_input):

        b, ind = gsp.nearest_neighbors(
            valid_input["centres"], n=valid_input["n_nearest"]
        )
        assert isinstance(b, list)
        assert isinstance(ind, list)
        assert len(b) == len(ind)
        assert len(b) == len(valid_input["centres"])
        assert len(ind) == len(valid_input["centres"])
        for i in range(len(b)):
            assert b[i].shape == (valid_input["n_nearest"],)
            assert ind[i].shape == (valid_input["n_nearest"],)


class Test_GriSPy_data_consistency_periodic:
    """Test that input and output params types.

    - Check that input params are stored in their corresponding attributes
    and have the expected type.
    - Check that output params have the expected types.
    """

    @pytest.fixture
    def gsp(self):
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
    def valid_input(self):
        rng = np.random.default_rng(1234)
        d = dict()
        # Define valid input data
        d["centres"] = rng.random((5, 3))
        d["upper_radii"] = 0.7
        d["lower_radii"] = 0.5
        d["n_nearest"] = 5
        return d

    def test_mirror_universe(self, gsp, valid_input):
        # Private methods should not be tested, but the idea is to make
        # some mirror method public.. so they can stay for now
        r_cen = np.array([[0, 0, 0]])
        t_cen, t_ind = gsp._mirror_universe(
            r_cen, distance_upper_bound=[valid_input["upper_radii"]]
        )
        assert isinstance(t_cen, np.ndarray)
        assert isinstance(t_ind, np.ndarray)
        assert len(t_cen) == len(t_ind)
        assert t_cen.ndim == t_cen.ndim
        assert t_cen.shape[1] == r_cen.shape[1]

    def test_mirror(self, gsp, valid_input):
        # Private methods should not be tested, but the idea is to make
        # some mirror method public.. so they can stay for now
        t_cen = gsp._mirror(
            np.array([[0, 0, 0]]),
            distance_upper_bound=[valid_input["upper_radii"]],
        )
        assert isinstance(t_cen, np.ndarray)

    def test_near_boundary(self, gsp, valid_input):
        # Private methods should not be tested, but the idea is to make
        # some mirror method public.. so they can stay for now
        mask = gsp._near_boundary(
            np.array([[0, 0, 0]]),
            distance_upper_bound=[valid_input["upper_radii"]],
        )
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.ndim == 1


class Test_GriSPy_valid_init:
    @pytest.fixture
    def valid_input(self):
        # Define valid input data
        rng = np.random.default_rng(seed=42)
        d = dict()
        d["data"] = rng.uniform(-1, 1, size=(100, 3))
        d["periodic"] = {0: (-1, 1), 1: (-1, 1), 2: None}
        d["metric"] = "euclid"
        d["N_cells"] = 10
        d["copy_data"] = True
        return d

    def test_invalid_periodic_as_string(self, valid_input):
        # Axis 0 with invalid type: string instead of dict
        bad_periodic = "{0: [-1, 1], 1: (-1, 1), 2: None}"
        with pytest.raises(TypeError):
            GriSPy(valid_input["data"], periodic=bad_periodic)

    def test_invalid_periodic_with_list(self, valid_input):
        # Axis 0 with invalid value type: list instead of tuple
        bad_periodic = {0: [-1, 1], 1: (-1, 1), 2: None}
        with pytest.raises(TypeError):
            GriSPy(valid_input["data"], periodic=bad_periodic)

    def test_invalid_periodic_with_string(self, valid_input):
        # Axis is not integer
        bad_periodic = {"A": (-1, 1), 1: (-1, 1), 2: None}
        with pytest.raises(TypeError):
            GriSPy(valid_input["data"], periodic=bad_periodic)

    def test_invalid_periodic_bad_edge_values(self, valid_input):
        # Edge 0 is larger than edge 1
        bad_periodic = {0: (1, -1), 1: (-1, 1), 2: None}
        with pytest.raises(ValueError):
            GriSPy(valid_input["data"], periodic=bad_periodic)

    def test_invalid_periodic_bad_edge_type(self, valid_input):
        # Edge has wrong type
        bad_periodic = {0: (-1, [1]), 1: (-1, 1), 2: None}
        with pytest.raises(TypeError):
            GriSPy(valid_input["data"], periodic=bad_periodic)

    def test_invalid_metric(self, valid_input):
        # Metric name is not a string
        bad_metric = 42
        with pytest.raises(ValueError):
            GriSPy(valid_input["data"], metric=bad_metric)

        # Metric name is wrong
        bad_metric = "euclidean"
        with pytest.raises(ValueError):
            GriSPy(valid_input["data"], metric=bad_metric)


class Test_GriSPy_set_periodicity:
    @pytest.fixture
    def gsp(self):
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

    def test_set_periodicity_inplace(self, gsp):
        periodicity = {0: (-50, 50)}

        assert gsp.periodic_flag_ is False
        assert gsp.periodic == {}

        result = gsp.set_periodicity(periodicity, inplace=True)

        assert result is None
        assert gsp.periodic_flag_
        assert gsp.periodic == {0: (-50, 50), 1: None, 2: None}

    def test_set_periodicity_no_inplace(self, gsp):
        periodicity = {0: (-50, 50)}

        assert gsp.periodic_flag_ is False
        assert gsp.periodic == {}

        result = gsp.set_periodicity(periodicity)

        assert isinstance(result, GriSPy)
        assert result is not gsp
        assert result.periodic_flag_
        assert result.periodic == {0: (-50, 50), 1: None, 2: None}

        assert gsp.periodic_flag_ is False
        assert gsp.periodic == {}


class Test_GriSPy_valid_query_input:
    @pytest.fixture
    def gsp(self):
        rng = np.random.default_rng(678)
        data = rng.uniform(-1, 1, size=(100, 3))
        periodic = {0: (-1, 1)}
        return GriSPy(data, periodic=periodic)

    @pytest.fixture
    def valid_input(self):
        rng = np.random.default_rng(1234)
        d = dict()
        # Define valid input data
        d["centres"] = rng.uniform(-1, 1, size=(10, 3))
        d["upper_radii"] = 0.8
        d["lower_radii"] = 0.4
        d["n_nearest"] = 5
        d["kind"] = "quicksort"
        d["sorted"] = True
        return d

    def test_invalid_centres_type(self, gsp, valid_input):
        # Invalid type
        bad_centres = [[1, 1, 1], [2, 2, 2]]
        with pytest.raises(TypeError):
            gsp.bubble_neighbors(
                bad_centres, distance_upper_bound=valid_input["upper_radii"]
            )

    def test_invalid_centres_single_value_type(self, gsp, valid_input):
        rng = np.random.default_rng(987)
        bad_centres = rng.uniform(-1, 1, size=(10, 3))
        bad_centres[4, 1] = np.inf  # add one invalid value
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=valid_input["upper_radii"],
            )

    def test_invalid_centres_shape(self, gsp, valid_input):
        rng = np.random.default_rng(987)
        # Invalid shape
        bad_centres = rng.uniform(-1, 1, size=(10, 2))
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=valid_input["upper_radii"],
            )

    def test_invalid_centres_empty(self, gsp, valid_input):
        # Invalid shape
        bad_centres = np.array([[], [], []]).reshape((0, 3))
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=valid_input["upper_radii"],
            )

    def test_invalid_bounds_type_bubble(self, gsp, valid_input):
        rng = np.random.default_rng(987)
        # Invalid type
        bad_upper_radii = list(rng.uniform(0.6, 1, size=10))
        with pytest.raises(TypeError):
            gsp.bubble_neighbors(
                valid_input["centres"],
                distance_upper_bound=bad_upper_radii,
            )

    def test_invalid_bounds_value_bubble(self, gsp, valid_input):
        rng = np.random.default_rng(987)
        # Invalid value
        bad_upper_radii = rng.uniform(0.6, 1, size=10)
        bad_upper_radii[5] = -1.0
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                valid_input["centres"],
                distance_upper_bound=bad_upper_radii,
            )

    def test_invalid_bounds_size_bubble(self, gsp, valid_input):
        rng = np.random.default_rng(987)
        # Different lenght than centres
        bad_upper_radii = rng.uniform(0.6, 1, size=11)
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                valid_input["centres"],
                distance_upper_bound=bad_upper_radii,
            )

    def test_invalid_bounds_larger_than_periodic_bubble(
        self, gsp, valid_input
    ):
        # Invalid value
        bad_upper_radii = 10.0  # larger than periodic range
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                valid_input["centres"],
                distance_upper_bound=bad_upper_radii,
            )

    def test_invalid_bounds_lenghts_shell(self, gsp, valid_input):
        rng = np.random.default_rng(987)
        # Different lenght than centres
        lower_radii = rng.uniform(0.1, 0.5, size=10)
        bad_upper_radii = rng.uniform(0.6, 1, size=11)
        with pytest.raises(ValueError):
            gsp.shell_neighbors(
                valid_input["centres"],
                distance_upper_bound=bad_upper_radii,
                distance_lower_bound=lower_radii,
            )

    def test_invalid_bounds_values_shell(self, gsp, valid_input):
        rng = np.random.default_rng(354)
        # Upper bound is lower than lower bound
        lower_radii = rng.uniform(0.1, 0.5, size=10)
        bad_upper_radii = rng.uniform(0.6, 1, size=10)
        bad_upper_radii[4] = lower_radii[4] - 0.05
        with pytest.raises(ValueError):
            gsp.shell_neighbors(
                valid_input["centres"],
                distance_upper_bound=bad_upper_radii,
                distance_lower_bound=lower_radii,
            )

    def test_invalid_sorted_type(self, gsp, valid_input):
        # Invalid type
        bad_sorted = "True"
        with pytest.raises(TypeError):
            gsp.bubble_neighbors(
                valid_input["centres"],
                distance_upper_bound=valid_input["upper_radii"],
                sorted=bad_sorted,
            )

    def test_invalid_sortkind_type(self, gsp, valid_input):
        # Invalid type
        bad_kind = ["quicksort"]  # string inside list
        with pytest.raises(TypeError):
            gsp.bubble_neighbors(
                valid_input["centres"],
                distance_upper_bound=valid_input["upper_radii"],
                kind=bad_kind,
            )

    def test_invalid_sortkind_value(self, gsp, valid_input):
        # Invalid name
        bad_kind = "quick_sort"
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                valid_input["centres"],
                distance_upper_bound=valid_input["upper_radii"],
                kind=bad_kind,
            )

    def test_invalid_nnearest_type(self, gsp, valid_input):

        # Invalid type
        bad_n = np.array([2])  # array instead of integer
        with pytest.raises(TypeError):
            gsp.nearest_neighbors(valid_input["centres"], n=bad_n)

    def test_invalid_nnearest_value(self, gsp, valid_input):
        # Invalid value
        bad_n = -5
        with pytest.raises(ValueError):
            gsp.nearest_neighbors(valid_input["centres"], n=bad_n)
