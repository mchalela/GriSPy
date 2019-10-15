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
from unittest import mock


class Test_Save:
    @pytest.fixture
    def grid(self):
        data = np.random.uniform(-10, 10, size=(10 ** 3, 3))
        return GriSPy(data)

    def test_save_firsttime(self, grid):
        file = "test_save_grid.gsp"
        with mock.patch('builtins.open', mock.mock_open()) as mf:
            grid.save_grid(file=file)
            mf().write.assert_called()

    def test_save_nooverwrite(self, grid):
        file = "test_save_grid.gsp"
        with mock.patch('os.path.isfile', return_value=True):
            with pytest.raises(FileExistsError):
                grid.save_grid(file=file)

    def test_save_overwrite(self, grid):
        file = "test_save_grid.gsp"
        with mock.patch('builtins.open', mock.mock_open()) as mf:
            with mock.patch('os.path.isfile', return_value=True):
                grid.save_grid(file=file, overwrite=True)
                mf().write.assert_called()

    def test_save_invalidfile(self, grid):
        file = ["invalid_file_type.gsp"]
        with mock.patch('builtins.open', mock.mock_open()):
            with pytest.raises(TypeError):
                grid.save_grid(file=file)


class Test_Load:
    @pytest.fixture
    def grid(self):
        data = np.random.uniform(-10, 10, size=(10 ** 2, 3))
        periodic = {0: (-20, 20), 1: (-5, 5)}
        return GriSPy(data, periodic=periodic)

    def test_load_nofile(self):
        with mock.patch('os.path.isfile', return_value=False):
            with pytest.raises(
                FileNotFoundError, match=r"There is no file named.*"
            ):
                GriSPy.load_grid("this_file_should_not_exist.gsp")

    def test_load_samestate_ag(self, grid):
        file = "test_load_grid.gsp"
        with mock.patch('builtins.open', mock.mock_open()):
            with mock.patch('pickle.dump') as pd:
                # Save a first time
                grid.save_grid(file=file)
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
                        assert_equal(grid["data"], gsp_tmp["data"])
                        assert_equal(grid["dim"], gsp_tmp["dim"])
                        assert_equal(grid["N_cells"], gsp_tmp["N_cells"])
                        assert_equal(grid["metric"], gsp_tmp["metric"])
                        assert_equal(grid["periodic"], gsp_tmp["periodic"])
                        assert_equal(grid["grid"], gsp_tmp["grid"])
                        assert_equal(grid["k_bins"], gsp_tmp["k_bins"])
                        assert_equal(grid["time"], gsp_tmp["time"])

    def test_load_invalidfile(self, grid):
        # Invalid filename
        file = ["invalid_file.gsp"]
        with mock.patch('builtins.open', mock.mock_open()):
            with pytest.raises(TypeError):
                GriSPy.load_grid(file=file)

            # Invalid instance of GriSPy
            file = "invalid_file.gsp"
            with mock.patch('os.path.isfile', return_value=True):
                with mock.patch('pickle.load', return_value=grid.__dict__):
                    with pytest.raises(TypeError):
                        GriSPy.load_grid(file=file)
