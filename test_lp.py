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

      self.data = np.array([[0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1]])
      self.gsp = GriSPy(self.data)
      self.upper_radii = 1.5
      self.lower_radii = 0.5
      self.n_nearest = 5

    def test_single_bubble_query(self, setUp):

      b, ind = self.gsp.bubble_neighbors(np.array([[0,0,0]]), distance_upper_bound=self.upper_radii)
      assert_(isinstance(b,list))
      assert_(isinstance(ind,list))

    def test_single_shell_query(self, setUp):

      b, ind = self.gsp.shell_neighbors(np.array([[0,0,0]]), distance_lower_bound=self.lower_radii, distance_upper_bound=self.upper_radii)
      assert_(isinstance(b,list))
      assert_(isinstance(ind,list))

    def test_single_nearest_neighbors_query(self, setUp):

      b, ind = self.gsp.nearest_neighbors(np.array([[0,0,0]]), n=self.n_nearest)
      assert_(isinstance(b,list))
      assert_(isinstance(ind,list))
