#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


import pytest

import numpy as np

from grispy import Grid, GriSPy
from grispy.core import BuildStats, PeriodicityConf

from numpy.testing import assert_equal, assert_


# =========================================================================
# Test Grid class
# =========================================================================


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
        assert_(isinstance(gsp.data, np.ndarray))
        assert_(isinstance(gsp.metric, str))
        assert_(isinstance(gsp.N_cells, int))
        assert_(isinstance(gsp.periodic, dict))

    def test_grispy_attrs(self, gsp):
        assert_(isinstance(gsp.k_bins_, np.ndarray))
        assert_(isinstance(gsp.grid_, dict))
        assert_(isinstance(gsp.dim, int))
        assert_(isinstance(gsp.periodic_flag_, bool))
        assert_(isinstance(gsp.periodic_conf_, PeriodicityConf))
        assert_(isinstance(gsp.time_, BuildStats))

    def test_bubble_single_query(self, gsp, valid_input):

        b, ind = gsp.bubble_neighbors(
            np.array([[0, 0, 0]]),
            distance_upper_bound=valid_input["upper_radii"],
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)

    def test_shell_single_query(self, gsp, valid_input):

        b, ind = gsp.shell_neighbors(
            np.array([[0, 0, 0]]),
            distance_lower_bound=valid_input["lower_radii"],
            distance_upper_bound=valid_input["upper_radii"],
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)

    def test_nearest_neighbors_single_query(self, gsp, valid_input):

        b, ind = gsp.nearest_neighbors(
            np.array([[0, 0, 0]]), n=valid_input["n_nearest"]
        )

        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)
        assert_equal(np.shape(b[0]), (valid_input["n_nearest"],))
        assert_equal(np.shape(ind[0]), (valid_input["n_nearest"],))

    def test_bubble_multiple_query(self, gsp, valid_input):

        b, ind = gsp.bubble_neighbors(
            valid_input["centres"],
            distance_upper_bound=valid_input["upper_radii"],
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(valid_input["centres"]))
        assert_equal(len(ind), len(valid_input["centres"]))

    def test_shell_multiple_query(self, gsp, valid_input):

        b, ind = gsp.shell_neighbors(
            valid_input["centres"],
            distance_lower_bound=valid_input["lower_radii"],
            distance_upper_bound=valid_input["upper_radii"],
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(valid_input["centres"]))
        assert_equal(len(ind), len(valid_input["centres"]))

    def test_nearest_neighbors_multiple_query(self, gsp, valid_input):

        b, ind = gsp.nearest_neighbors(
            valid_input["centres"], n=valid_input["n_nearest"]
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(valid_input["centres"]))
        assert_equal(len(ind), len(valid_input["centres"]))
        for i in range(len(b)):
            assert_equal(np.shape(b[i]), (valid_input["n_nearest"],))
            assert_equal(np.shape(ind[i]), (valid_input["n_nearest"],))


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
        assert_(isinstance(t_cen, np.ndarray))
        assert_(isinstance(t_ind, np.ndarray))
        assert_equal(len(t_cen), len(t_ind))
        assert_equal(t_cen.ndim, t_cen.ndim)
        assert_equal(t_cen.shape[1], r_cen.shape[1])

    def test_mirror(self, gsp, valid_input):
        # Private methods should not be tested, but the idea is to make
        # some mirror method public.. so they can stay for now
        t_cen = gsp._mirror(
            np.array([[0, 0, 0]]),
            distance_upper_bound=[valid_input["upper_radii"]],
        )
        assert_(isinstance(t_cen, np.ndarray))

    def test_near_boundary(self, gsp, valid_input):
        # Private methods should not be tested, but the idea is to make
        # some mirror method public.. so they can stay for now
        mask = gsp._near_boundary(
            np.array([[0, 0, 0]]),
            distance_upper_bound=[valid_input["upper_radii"]],
        )
        assert_(isinstance(mask, np.ndarray))
        assert_equal(mask.ndim, 1)


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

    def test_invalid_data_inf(self):
        bad_data = np.random.uniform(-1, 1, size=(100, 3))
        bad_data[42, 1] = np.inf  # add one invalid value
        with pytest.raises(ValueError):
            GriSPy(bad_data)

    def test_invalid_data_type(self):
        # Data type
        data = 4
        with pytest.raises(TypeError):
            GriSPy(data=data)

    def test_invalid_data_empty_array(self):
        # Data format
        data = np.array([])
        with pytest.raises(ValueError):
            GriSPy(data=data)

        # Data format with shape
        data = np.array([[]])
        with pytest.raises(ValueError):
            GriSPy(data=data)

    def test_invalid_data_format(self):
        # Data format
        data = np.array([1, 1, 1])
        with pytest.raises(ValueError):
            GriSPy(data=data)

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

    def test_invalid_Ncells(self, valid_input):
        # N_cells is not integer
        bad_N_cells = 10.5
        with pytest.raises(TypeError):
            GriSPy(valid_input["data"], N_cells=bad_N_cells)

        # N_cells is not positive
        bad_N_cells = -10
        with pytest.raises(ValueError):
            GriSPy(valid_input["data"], N_cells=bad_N_cells)

    def test_invalid_copy_data(self, valid_input):
        # copy_data is not bool
        bad_copy_data = 42
        with pytest.raises(TypeError):
            GriSPy(valid_input["data"], copy_data=bad_copy_data)


class Test_GriSPy_set_periodicity:
    @pytest.fixture
    def gsp(self):
        rng = np.random.default_rng(3596)
        self.centres = rng.random((5, 3))
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
