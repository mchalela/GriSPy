#from __future__ import division, print_function, absolute_import

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

      self.upper_radii = 0.7
      self.lower_radii = 0.5
      self.n_nearest = 5

      self.gsp = GriSPy(self.data)
  
    def test_grid_data(self, setUp):
      assert_(isinstance(self.gsp.dim,int))
      assert_(isinstance(self.gsp.data,np.ndarray))
      assert_(isinstance(self.gsp.k_bins,np.ndarray))
      assert_(isinstance(self.gsp.metric,str))
      assert_(isinstance(self.gsp.N_cells,int))
      assert_(isinstance(self.gsp.grid,dict))
      assert_(isinstance(self.gsp.periodic,dict))
      assert_(isinstance(self.gsp.periodic_flag,bool))
      assert_(isinstance(self.gsp.time,dict))

    def test_bubble_single_query(self, setUp):

      b, ind = self.gsp.bubble_neighbors(np.array([[0,0,0]]), distance_upper_bound=self.upper_radii)
      assert_(isinstance(b,list))
      assert_(isinstance(ind,list))
      assert_equal(len(b),len(ind))
      assert_equal(len(b),1)
      assert_equal(len(ind),1)

    def test_shell_single_query(self, setUp):

      b, ind = self.gsp.shell_neighbors(np.array([[0,0,0]]), distance_lower_bound=self.lower_radii, distance_upper_bound=self.upper_radii)
      assert_(isinstance(b,list))
      assert_(isinstance(ind,list))
      assert_equal(len(b),len(ind))
      assert_equal(len(b),1)
      assert_equal(len(ind),1)

    def test_nearest_neighbors_single_query(self, setUp):

      b, ind = self.gsp.nearest_neighbors(np.array([[0,0,0]]), n=self.n_nearest)
      assert_(isinstance(b,list))
      assert_(isinstance(ind,list))
      assert_equal(len(b),len(ind))
      assert_equal(len(b),1)
      assert_equal(len(ind),1)
      assert_equal(np.shape(b[0]),(self.n_nearest,))
      assert_equal(np.shape(ind[0]),(self.n_nearest,))

    def test_bubble_multiple_query(self, setUp):

      b, ind = self.gsp.bubble_neighbors(self.centres, distance_upper_bound=self.upper_radii)
      assert_(isinstance(b,list))
      assert_(isinstance(ind,list))
      assert_equal(len(b),len(ind))
      assert_equal(len(b),len(self.centres))
      assert_equal(len(ind),len(self.centres))

    def test_shell_multiple_query(self, setUp):

      b, ind = self.gsp.shell_neighbors(self.centres, distance_lower_bound=self.lower_radii, distance_upper_bound=self.upper_radii)
      assert_(isinstance(b,list))
      assert_(isinstance(ind,list))
      assert_equal(len(b),len(ind))
      assert_equal(len(b),len(self.centres))
      assert_equal(len(ind),len(self.centres))

    def test_nearest_neighbors_multiple_query(self, setUp):

      b, ind = self.gsp.nearest_neighbors(self.centres, n=self.n_nearest)
      assert_(isinstance(b,list))
      assert_(isinstance(ind,list))
      assert_equal(len(b),len(ind))
      assert_equal(len(b),len(self.centres))
      assert_equal(len(ind),len(self.centres))
      for i in range(len(b)):
        assert_equal(np.shape(b[i]),(self.n_nearest,))
        assert_equal(np.shape(ind[i]),(self.n_nearest,))

class Test_grispy():

    @pytest.fixture
    def setUp_1d(self):
      
      np.random.seed(1234)
      npoints = 10 ** 5
      lbox = 100.0
      periodic = {0: (-lbox*0.5, lbox*0.5)}
      self.centres = lbox*(0.5 - np.random.rand(1,10).T)
      self.data = np.random.uniform(-0.5*lbox, 0.5*lbox, size=(npoints, 1))
      self.upper_radii = 0.25*lbox
      self.lower_radii = 0.20*lbox
      self.n_nearest = 32
      self.eps = 1e-6

      self.gsp = GriSPy(self.data)

    def test_nearest_neighbors_sort(self, setUp_1d):

      b, ind = self.gsp.nearest_neighbors(self.centres, n=self.n_nearest)
      for i in range(len(b)):
        assert_equal(sorted(b[i]), b[i])

    def test_all_in_bubble(self, setUp_1d):

        b, ind = self.gsp.bubble_neighbors(self.centres, distance_upper_bound=self.upper_radii)

        for i, l in enumerate(ind):
            for j in l:
                d = np.fabs(self.centres[i] - self.data[j])
                assert_(d <= self.upper_radii*(1.+self.eps))

    def test_all_in_shell(self, setUp_1d):

        b, ind = self.gsp.shell_neighbors(self.centres, distance_lower_bound=self.lower_radii, distance_upper_bound=self.upper_radii)

        for i, l in enumerate(ind):
            for j in l:
                d = np.fabs(self.centres[i] - self.data[j])
                assert_(d <= self.upper_radii*(1.+self.eps))
                assert_(d >= self.lower_radii*(1.-self.eps))


