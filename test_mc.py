import os
from numpy.testing import assert_equal, assert_
import numpy as np
from grispy import GriSPy
import pytest


def clean(file):
    # Remove the output file
    os.remove(file)


class Test_Save:
    @pytest.fixture
    def setUp(self):

        self.data = np.random.uniform(-10, 10, size=(10 ** 3, 3))
        self.periodic = {0: (-20, 20), 1: (-5, 5)}
        self.gsp = GriSPy(self.data)

    def test_save_firsttime(self, setUp):
        file = "test_save_grid.npy"
        self.gsp.save_grid(file=file)
        assert_(os.path.isfile(file))
        clean(file)

    def test_save_nooverwrite(self, setUp):
        file = "test_save_grid.npy"

        # Save a first time
        self.gsp.save_grid(file=file)
        assert_(os.path.isfile(file))

        # Attempt to save when file already exists
        with pytest.raises(FileExistsError):
            self.gsp.save_grid(file=file)
        clean(file)

    def test_save_overwrite(self, setUp):
        file = "test_save_grid.npy"

        # Save a first time
        self.gsp.save_grid(file=file)
        assert_(os.path.isfile(file))

        # Attempt to save when file already exists
        self.gsp.save_grid(file=file, overwrite=True)
        assert_(os.path.isfile(file))
        clean(file)


class Test_Load:
    @pytest.fixture
    def setUp(self):

        data = np.random.uniform(-10, 10, size=(10 ** 2, 3))
        periodic = {0: (-20, 20), 1: (-5, 5)}
        self.gsp = GriSPy(data, periodic=periodic)

    def test_load_nofile(self):
        with pytest.raises(FileNotFoundError):
            gsp_tmp = GriSPy(load_grid="this_file_should_not_exist.npy")

    def test_load_samestate(self, setUp):
        file = "test_load_grid.npy"

        # Save a first time
        self.gsp.save_grid(file=file)
        assert_(os.path.isfile(file))

        # Load again to check the state is the same
        gsp_tmp = GriSPy(load_grid=file)
        assert_equal(self.gsp["data"], gsp_tmp["data"])
        assert_equal(self.gsp["dim"], gsp_tmp["dim"])
        assert_equal(self.gsp["N_cells"], gsp_tmp["N_cells"])
        assert_equal(self.gsp["metric"], gsp_tmp["metric"])
        assert_equal(self.gsp["periodic"], gsp_tmp["periodic"])
        assert_equal(self.gsp["grid"], gsp_tmp["grid"])
        assert_equal(self.gsp["k_bins"], gsp_tmp["k_bins"])
        assert_equal(self.gsp["time"], gsp_tmp["time"])
        clean(file)
