from __future__ import division, print_function, absolute_import

from numpy.testing import (
    assert_equal,
    assert_
)

import numpy as np
from grispy import GriSPy
import pytest

class Test_data_consistency:

    @pytest.fixture
    def setUp(self):

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

        self.gsp = GriSPy(self.data)

    def test_grid_data(self, setUp):
        assert_(isinstance(self.gsp.dim, int))
        assert_(isinstance(self.gsp.data, np.ndarray))
        assert_(isinstance(self.gsp.k_bins, np.ndarray))
        assert_(isinstance(self.gsp.metric, str))
        assert_(isinstance(self.gsp.N_cells, int))
        assert_(isinstance(self.gsp.grid, dict))
        assert_(isinstance(self.gsp.periodic, dict))
        assert_(isinstance(self.gsp.periodic_flag, bool))
        assert_(isinstance(self.gsp.time, dict))

    def test_bubble_single_query(self, setUp):

        b, ind = self.gsp.bubble_neighbors(
            np.array([[0, 0, 0]]), distance_upper_bound=self.upper_radii[0]
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)

    def test_shell_single_query(self, setUp):

        b, ind = self.gsp.shell_neighbors(
            np.array([[0, 0, 0]]),
            distance_lower_bound=self.lower_radii[0],
            distance_upper_bound=self.upper_radii[0],
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)

    def test_nearest_neighbors_single_query(self, setUp):

        b, ind = self.gsp.nearest_neighbors(
            np.array([[0, 0, 0]]), n=self.n_nearest
        )

        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)
        assert_equal(np.shape(b[0]), (self.n_nearest,))
        assert_equal(np.shape(ind[0]), (self.n_nearest,))

    def test_bubble_multiple_query(self, setUp):

        b, ind = self.gsp.bubble_neighbors(
            self.centres, distance_upper_bound=self.upper_radii[0]
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(self.centres))
        assert_equal(len(ind), len(self.centres))

    def test_shell_multiple_query(self, setUp):

        b, ind = self.gsp.shell_neighbors(
            self.centres,
            distance_lower_bound=self.lower_radii[0],
            distance_upper_bound=self.upper_radii[0],
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(self.centres))
        assert_equal(len(ind), len(self.centres))

    def test_nearest_neighbors_multiple_query(self, setUp):

        b, ind = self.gsp.nearest_neighbors(self.centres, n=self.n_nearest)
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
    def setUp(self):

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

        self.gsp = GriSPy(self.data, periodic=self.periodic)

    def test_mirror_universe(self, setUp):
        r_cen = np.array([[0, 0, 0]])
        t_cen, t_ind = self.gsp._mirror_universe(
            r_cen, distance_upper_bound=self.upper_radii
        )
        assert_(isinstance(t_cen, np.ndarray))
        assert_(isinstance(t_ind, np.ndarray))
        assert_equal(len(t_cen), len(t_ind))
        assert_equal(t_cen.ndim, t_cen.ndim)
        assert_equal(t_cen.shape[1], r_cen.shape[1])

    def test_mirror(self, setUp):
        t_cen = self.gsp._mirror(
            np.array([[0, 0, 0]]), distance_upper_bound=self.upper_radii
        )
        assert_(isinstance(t_cen, np.ndarray))

    def test_near_boundary(self, setUp):
        mask = self.gsp._near_boundary(
            np.array([[0, 0, 0]]), distance_upper_bound=self.upper_radii
        )
        assert_(isinstance(mask, np.ndarray))
        assert_equal(mask.ndim, 1)

def test__init__A_01():
    # Data type
    data = 4
    periodic = {0: None, 1: None}
    with pytest.raises(TypeError, match=r".*must be a numpy array*"):
        gsp = GriSPy(
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
        gsp = GriSPy(
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
        gsp = GriSPy(
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
        gsp = GriSPy(
            data=data,
            N_cells=2,
            copy_data=False,
            periodic=periodic,
            metric="sphere",
        )       
