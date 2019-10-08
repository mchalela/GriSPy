import os
from numpy.testing import assert_equal, assert_
import numpy as np
from grispy import GriSPy
import pytest
from unittest import mock


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
        file = "test_save_grid.gsp"
        with mock.patch('builtins.open', mock.mock_open()) as mf:
            self.gsp.save_grid(file=file)
            mf().write.assert_called()

    def test_save_nooverwrite(self, setUp):
        file = "test_save_grid.gsp"
        with mock.patch('os.path.isfile', return_value=True):
            with pytest.raises(FileExistsError):
                self.gsp.save_grid(file=file)

    def test_save_overwrite(self, setUp):
        file = "test_save_grid.gsp"
        with mock.patch('builtins.open', mock.mock_open()) as mf:
            with mock.patch('os.path.isfile', return_value=True):
                self.gsp.save_grid(file=file, overwrite=True)
                mf().write.assert_called()

    def test_save_invalidfile(self, setUp):
        file = ["invalid_file_type.gsp"]
        with mock.patch('builtins.open', mock.mock_open()):
            with pytest.raises(TypeError):
                self.gsp.save_grid(file=file)


class Test_Load:
    @pytest.fixture
    def setUp(self):
        data = np.random.uniform(-10, 10, size=(10 ** 2, 3))
        periodic = {0: (-20, 20), 1: (-5, 5)}
        self.gsp = GriSPy(data, periodic=periodic)

    def test_load_nofile(self):
        with mock.patch('os.path.isfile', return_value=False):
            with pytest.raises(
                FileNotFoundError, match=r"There is no file named.*"
            ):
                GriSPy.load_grid("this_file_should_not_exist.gsp")

    def test_load_samestate_ag(self, setUp):
        file = "test_load_grid.gsp"
        with mock.patch('builtins.open', mock.mock_open()):
            with mock.patch('pickle.dump') as pd:
                # Save a first time
                self.gsp.save_grid(file=file)
                args_pd, kwargs_pd = pd.call_args_list[0]

            # Load again to check the state is the same
                with mock.patch('os.path.isfile', return_value=True):
                    with mock.patch('pickle.load', return_value=args_pd[0]):
                        gsp_tmp = GriSPy.load_grid(file)
                        assert_(isinstance(gsp_tmp["dim"], int))
                        assert_(isinstance(gsp_tmp["data"], np.ndarray))
                        assert_(isinstance(gsp_tmp["k_bins"], np.ndarray))
                        assert_(isinstance(gsp_tmp["metric"], str))
                        assert_(isinstance(gsp_tmp["N_cells"], int))
                        assert_(isinstance(gsp_tmp["grid"], dict))
                        assert_(isinstance(gsp_tmp["periodic"], dict))
                        assert_(isinstance(gsp_tmp["periodic_flag"], bool))
                        assert_(isinstance(gsp_tmp["time"], dict))
                        assert_equal(self.gsp["data"], gsp_tmp["data"])
                        assert_equal(self.gsp["dim"], gsp_tmp["dim"])
                        assert_equal(self.gsp["N_cells"], gsp_tmp["N_cells"])
                        assert_equal(self.gsp["metric"], gsp_tmp["metric"])
                        assert_equal(self.gsp["periodic"], gsp_tmp["periodic"])
                        assert_equal(self.gsp["grid"], gsp_tmp["grid"])
                        assert_equal(self.gsp["k_bins"], gsp_tmp["k_bins"])
                        assert_equal(self.gsp["time"], gsp_tmp["time"])

    def test_load_invalidfile(self, setUp):
        # Invalid filename
        file = ["invalid_file.gsp"]
        with mock.patch('builtins.open', mock.mock_open()):
            with pytest.raises(TypeError):
                GriSPy.load_grid(file=file)

            # Invalid instance of GriSPy
            bad_gsp = self.gsp.__dict__
            file = "invalid_file.gsp"
            with mock.patch('os.path.isfile', return_value=True):
                with mock.patch('pickle.load', return_value=bad_gsp):
                    with pytest.raises(TypeError):
                        GriSPy.load_grid(file=file)
