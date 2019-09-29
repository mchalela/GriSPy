from numpy.testing import (assert_equal, assert_array_equal,
    assert_almost_equal, assert_array_almost_equal, assert_, run_module_suite)

import sys
import numpy as np
from grispy import GriSPy
import pytest

class Test_data_consistency():

    @pytest.fixture
    def setUp(self):

        np.random.seed(1234)
        self.centres = np.random.rand(3,5).T

        self.data = np.array([[0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1]])

        self.upper_radii = [0.7]
        self.lower_radii = [0.5]
        self.n_nearest = 5

        self.periodic = {0: (0., 1.)}

        self.gsp = GriSPy(self.data, periodic=self.periodic)

    def test_mirror_universe(self, setUp):
        r_cen = np.array([[0,0,0]])
        t_cen, t_ind = self.gsp._mirror_universe(r_cen, distance_upper_bound=self.upper_radii)
        assert_(isinstance(t_cen,np.ndarray))
        assert_(isinstance(t_ind,np.ndarray))
        assert_equal(len(t_cen),len(t_ind))
        assert_equal(t_cen.ndim,t_cen.ndim)
        assert_equal(t_cen.shape[1],r_cen.shape[1])

    def test_mirror(self, setUp):
        t_cen = self.gsp._mirror(np.array([[0,0,0]]), distance_upper_bound=self.upper_radii)
        assert_(isinstance(t_cen,np.ndarray))

    def test_near_boundary(self, setUp):
        mask = self.gsp._near_boundary(np.array([[0,0,0]]), distance_upper_bound=self.upper_radii)
        assert_(isinstance(mask,np.ndarray))
        assert_equal(mask.ndim,1)


class Test_periodicity_grispy():

    @pytest.fixture
    def setUp_1d(self):

      self.lbox = 10.0
      self.data = np.array([[ 2, 0, 0],
                            [-2, 0, 0],
                            [ 0, 2, 0],
                            [ 0,-2, 0],
                            [ 0, 0, 2],
                            [ 0, 0,-2]])

      self.eps = 1e-6
      self.gsp = GriSPy(self.data)

    def test_periodicity_in_shell(self, setUp_1d):

        centres = np.array([[0.,0.,0.]])
        upper_radii = 0.81 * self.lbox
        lower_radii = 0.79 * self.lbox

        for j in range(3):
            self.gsp.set_periodicity({j: (-self.lbox*0.5, self.lbox*0.5)})

            dis, ind = self.gsp.shell_neighbors(
                centres,
                distance_lower_bound=lower_radii,
                distance_upper_bound=upper_radii
            )
            dis, ind = dis[0], ind[0]

            aux = np.argsort(ind)
            ind = ind[aux]
            dis = dis[aux]

            for i in range(2):
                assert_equal(ind[i], i + (j * 2))
                assert_(dis[i] <= upper_radii)
                assert_(dis[i] >= lower_radii)


        self.gsp.set_periodicity({
            0: (-self.lbox*0.5, self.lbox*0.5),
            1: (-self.lbox*0.5, self.lbox*0.5),
            2: (-self.lbox*0.5, self.lbox*0.5)
        })
        dis, ind = self.gsp.shell_neighbors(
            centres,
            distance_lower_bound=lower_radii,
            distance_upper_bound=upper_radii
        )
        dis, ind = dis[0], ind[0]

        aux = np.argsort(ind)
        ind = ind[aux]
        dis = dis[aux]

        for i in range(6):
            assert_equal(ind[i],i)
            assert_(dis[i] <= upper_radii*(1.+self.eps))
            assert_(dis[i] >= lower_radii*(1.-self.eps))

    def test_periodicity_in_bubble(self, setUp_1d):

        centres = np.array([[5.,0.,0.],
                            [0.,5.,0.],
                            [0.,0.,5.],
                           ])
        upper_radii = 0.3 * self.lbox

        for j in range(3):
            self.gsp.set_periodicity({j: (-self.lbox*0.5, self.lbox*0.5)})

            centre = centres[j].reshape(1,3)

            dis, ind = self.gsp.bubble_neighbors(
                centre,
                distance_upper_bound=upper_radii
            )
            dis, ind = dis[0], ind[0]

            aux = np.argsort(ind)
            ind = ind[aux]
            dis = dis[aux]

            print(ind,dis)

            for i in range(2):
                assert_equal(ind[i], i + (j * 2))
                assert_(dis[i] <= upper_radii * (1. + self.eps))
                assert_(dis[i] >= upper_radii * (1. - self.eps))
