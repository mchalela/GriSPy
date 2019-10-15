#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


import pytest
import numpy as np
from grispy import GriSPy
from numpy.testing import assert_equal, assert_


class Test_data_consistency:

    @pytest.fixture
    def gsp(self):

        np.random.seed(1234)
        self.centres = np.random.rand(3, 5).T

        self.data = np.array(
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

        self.upper_radii = [0.7]
        self.lower_radii = [0.5]
        self.n_nearest = 5

        return GriSPy(self.data)

    def test_grid_data(self, gsp):
        assert_(isinstance(gsp.dim, int))
        assert_(isinstance(gsp.data, np.ndarray))
        assert_(isinstance(gsp.k_bins, np.ndarray))
        assert_(isinstance(gsp.metric, str))
        assert_(isinstance(gsp.N_cells, int))
        assert_(isinstance(gsp.grid, dict))
        assert_(isinstance(gsp.periodic, dict))
        assert_(isinstance(gsp.periodic_flag, bool))
        assert_(isinstance(gsp.time, dict))

    def test_bubble_single_query(self, gsp):

        b, ind = gsp.bubble_neighbors(
            np.array([[0, 0, 0]]), distance_upper_bound=self.upper_radii[0]
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)

    def test_shell_single_query(self, gsp):

        b, ind = gsp.shell_neighbors(
            np.array([[0, 0, 0]]),
            distance_lower_bound=self.lower_radii[0],
            distance_upper_bound=self.upper_radii[0],
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)

    def test_nearest_neighbors_single_query(self, gsp):

        b, ind = gsp.nearest_neighbors(
            np.array([[0, 0, 0]]), n=self.n_nearest
        )

        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)
        assert_equal(np.shape(b[0]), (self.n_nearest,))
        assert_equal(np.shape(ind[0]), (self.n_nearest,))

    def test_bubble_multiple_query(self, gsp):

        b, ind = gsp.bubble_neighbors(
            self.centres, distance_upper_bound=self.upper_radii[0]
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(self.centres))
        assert_equal(len(ind), len(self.centres))

    def test_shell_multiple_query(self, gsp):

        b, ind = gsp.shell_neighbors(
            self.centres,
            distance_lower_bound=self.lower_radii[0],
            distance_upper_bound=self.upper_radii[0],
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(self.centres))
        assert_equal(len(ind), len(self.centres))

    def test_nearest_neighbors_multiple_query(self, gsp):

        b, ind = gsp.nearest_neighbors(self.centres, n=self.n_nearest)
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(self.centres))
        assert_equal(len(ind), len(self.centres))
        for i in range(len(b)):
            assert_equal(np.shape(b[i]), (self.n_nearest,))
            assert_equal(np.shape(ind[i]), (self.n_nearest,))


class Test_data_consistency_periodic:

    @pytest.fixture
    def gsp(self):

        np.random.seed(1234)
        self.centres = np.random.rand(3, 5).T

        self.data = np.array(
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

        self.upper_radii = [0.7]
        self.lower_radii = [0.5]
        self.n_nearest = 5

        self.periodic = {0: (0.0, 1.0)}

        return GriSPy(self.data, periodic=self.periodic)

    def test_mirror_universe(self, gsp):
        r_cen = np.array([[0, 0, 0]])
        t_cen, t_ind = gsp._mirror_universe(
            r_cen, distance_upper_bound=self.upper_radii
        )
        assert_(isinstance(t_cen, np.ndarray))
        assert_(isinstance(t_ind, np.ndarray))
        assert_equal(len(t_cen), len(t_ind))
        assert_equal(t_cen.ndim, t_cen.ndim)
        assert_equal(t_cen.shape[1], r_cen.shape[1])

    def test_mirror(self, gsp):
        t_cen = gsp._mirror(
            np.array([[0, 0, 0]]), distance_upper_bound=self.upper_radii
        )
        assert_(isinstance(t_cen, np.ndarray))

    def test_near_boundary(self, gsp):
        mask = gsp._near_boundary(
            np.array([[0, 0, 0]]), distance_upper_bound=self.upper_radii
        )
        assert_(isinstance(mask, np.ndarray))
        assert_equal(mask.ndim, 1)


def test__init__A_01():
    # Data type
    data = 4
    periodic = {0: None, 1: None}
    with pytest.raises(TypeError, match=r".*must be a numpy array*"):
        GriSPy(
            data=data,
            N_cells=2,
            copy_data=False,
            periodic=periodic,
            metric="sphere",
        )


def test__init__A_02():
    # Data format
    data = np.array([])
    periodic = {0: None, 1: None}
    with pytest.raises(ValueError):
        GriSPy(
            data=data,
            N_cells=2,
            copy_data=False,
            periodic=periodic,
            metric="sphere",
        )


def test__init__A_03():
    # Data format
    data = np.array([1, 1, 1])
    periodic = {0: None, 1: None}
    with pytest.raises(ValueError):
        GriSPy(
            data=data,
            N_cells=2,
            copy_data=False,
            periodic=periodic,
            metric="sphere",
        )


def test__init__A_04():
    # Data value
    data = np.array([[]])
    periodic = {0: None, 1: None}
    with pytest.raises(ValueError):
        GriSPy(
            data=data,
            N_cells=2,
            copy_data=False,
            periodic=periodic,
            metric="sphere",
        )


class Test_valid_query_input:
    @pytest.fixture
    def gsp(self):

        # Define valid input data
        self.centres = np.random.uniform(-1, 1, size=(10, 3))
        self.upper_radii = 0.8
        self.lower_radii = 0.4
        self.kind = "quicksort"
        self.sorted = True
        self.n = 5

        data = np.random.uniform(-1, 1, size=(100, 3))
        periodic = {0: (-1, 1)}
        return GriSPy(data, periodic=periodic)

    def test_invalid_centres(self, gsp):
        # Invalid type
        bad_centres = [[1, 1, 1], [2, 2, 2]]
        with pytest.raises(TypeError):
            gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=self.upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        bad_centres = np.random.uniform(-1, 1, size=(10, 3))
        bad_centres[4, 1] = np.inf    # add one invalid value
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=self.upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Invalid shape
        bad_centres = np.random.uniform(-1, 1, size=(10, 2))
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=self.upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Invalid shape
        bad_centres = np.array([[], [], []]).reshape((0, 3))
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=self.upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

    def test_invalid_bounds_bubble(self, gsp):

        # Invalid type
        bad_upper_radii = list(np.random.uniform(0.6, 1, size=10))
        with pytest.raises(TypeError):
            gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Invalid value
        bad_upper_radii = np.random.uniform(0.6, 1, size=10)
        bad_upper_radii[5] = -1.
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Different lenght than centres
        bad_upper_radii = np.random.uniform(0.6, 1, size=11)
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Invalid value
        bad_upper_radii = 10.  # larger than periodic range
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

    def test_invalid_bounds_shell(self, gsp):

        # Different lenght than centres
        lower_radii = np.random.uniform(0.1, 0.5, size=10)
        bad_upper_radii = np.random.uniform(0.6, 1, size=11)
        with pytest.raises(ValueError):
            gsp.shell_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                distance_lower_bound=lower_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Upper bound is lower than lower bound
        lower_radii = np.random.uniform(0.1, 0.5, size=10)
        bad_upper_radii = np.random.uniform(0.6, 1, size=10)
        bad_upper_radii[4] = lower_radii[4] - 0.05
        with pytest.raises(ValueError):
            gsp.shell_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                distance_lower_bound=lower_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

    def test_invalid_bool(self, gsp):

        # Invalid type
        bad_sorted = "True"
        with pytest.raises(TypeError):
            gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=self.upper_radii,
                sorted=bad_sorted,
                kind=self.kind,
            )

    def test_invalid_sortkind(self, gsp):

        # Invalid type
        bad_kind = ["quicksort"]    # string inside list
        with pytest.raises(TypeError):
            gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=self.upper_radii,
                sorted=self.sorted,
                kind=bad_kind,
            )

        # Invalid name
        bad_kind = "quick_sort"
        with pytest.raises(ValueError):
            gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=self.upper_radii,
                sorted=self.sorted,
                kind=bad_kind,
            )

    def test_invalid_nnearest(self, gsp):

        # Invalid type
        bad_n = np.array([2])    # array instead of integer
        with pytest.raises(TypeError):
            gsp.nearest_neighbors(
                self.centres,
                n=bad_n,
                kind=self.kind,
            )

        # Invalid value
        bad_n = -5
        with pytest.raises(ValueError):
            gsp.nearest_neighbors(
                self.centres,
                n=bad_n,
                kind=self.kind,
            )

        # Invalid value
        bad_n = 10**10   # too large
        with pytest.raises(ValueError):
            gsp.nearest_neighbors(
                self.centres,
                n=bad_n,
                kind=self.kind,
            )


class Test_valid_init:

    @pytest.fixture
    def gsp(self):
        # Define valid input data
        self.data = np.random.uniform(-1, 1, size=(100, 3))
        self.periodic = {0: (-1, 1), 1: (-1, 1), 2: None}
        self.metric = "euclid"
        self.N_cells = 10
        self.copy_data = True

    def test_invalid_data(self, gsp):
        bad_data = np.random.uniform(-1, 1, size=(100, 3))
        bad_data[42, 1] = np.inf    # add one invalid value
        with pytest.raises(ValueError):
            GriSPy(
                bad_data,
                N_cells=self.N_cells,
                periodic=self.periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

    def test_invalid_periodic(self, gsp):
        # Axis 0 with invalid type: string instead of dict
        bad_periodic = '{0: [-1, 1], 1: (-1, 1), 2: None}'
        with pytest.raises(TypeError):
            GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=bad_periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

        # Axis 0 with invalid value type: list instead of tuple
        bad_periodic = {0: [-1, 1], 1: (-1, 1), 2: None}
        with pytest.raises(TypeError):
            GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=bad_periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

        # Axis is not integer
        bad_periodic = {'A': (-1, 1), 1: (-1, 1), 2: None}
        with pytest.raises(TypeError):
            GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=bad_periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

        # Edge 0 is larger than edge 1
        bad_periodic = {0: (1, -1), 1: (-1, 1), 2: None}
        with pytest.raises(ValueError):
            GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=bad_periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

        # Edge has wrong type
        bad_periodic = {0: (-1, [1]), 1: (-1, 1), 2: None}
        with pytest.raises(TypeError):
            GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=bad_periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

    def test_invalid_metric(self, gsp):
        # Metric name is not a string
        bad_metric = 42
        with pytest.raises(TypeError):
            GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=self.periodic,
                metric=bad_metric,
                copy_data=self.copy_data,
            )

        # Metric name is wrong
        bad_metric = "euclidean"
        with pytest.raises(ValueError):
            GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=self.periodic,
                metric=bad_metric,
                copy_data=self.copy_data,
            )

    def test_invalid_Ncells(self, gsp):
        # N_cells is not integer
        bad_N_cells = 10.5
        with pytest.raises(TypeError):
            GriSPy(
                self.data,
                N_cells=bad_N_cells,
                periodic=self.periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

        # N_cells is not positive
        bad_N_cells = -10
        with pytest.raises(ValueError):
            GriSPy(
                self.data,
                N_cells=bad_N_cells,
                periodic=self.periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

    def test_invalid_copy_data(self, gsp):
        # copy_data is not bool
        bad_copy_data = 42
        with pytest.raises(TypeError):
            GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=self.periodic,
                metric=self.metric,
                copy_data=bad_copy_data,
            )
