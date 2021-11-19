#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


import numpy as np
import pytest

from grispy import Grid, GriSPy, Periodicity

# =========================================================================
# Grid fixtures
# =========================================================================


@pytest.fixture
def grid():
    rng = np.random.default_rng(4321)
    data = rng.uniform(0, 1, size=(500, 3))
    return Grid(data, 3)


@pytest.fixture
def grid_input():
    rng = np.random.default_rng(1234)
    d = dict()
    # Define valid input data
    d["data"] = rng.random((10, 3))
    d["points"] = rng.uniform(0.3, 0.7, size=(10, 3))
    d["inside_points"] = rng.uniform(0.3, 0.7, size=(10, 3))
    d["outside_points"] = rng.uniform(10, 11, size=(10, 3))
    d["mix_points"] = np.vstack((d["inside_points"], d["outside_points"]))
    return d


# =========================================================================
# GriSPy fixtures
# =========================================================================


@pytest.fixture
def gsp():
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
def gsp_periodic():
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
def grispy_init():
    # Define valid input data
    rng = np.random.default_rng(seed=42)
    d = dict()
    d["data"] = rng.uniform(-1, 1, size=(100, 3))
    d["periodic"] = {0: (-1, 1), 1: (-1, 1), 2: None}
    d["metric"] = "euclid"
    d["N_cells"] = 10
    d["copy_data"] = True
    return d


@pytest.fixture
def grispy_input():
    rng = np.random.default_rng(1234)
    d = dict()
    # Define valid input data
    d["centres"] = rng.random((5, 3))
    d["upper_radii"] = 0.7
    d["lower_radii"] = 0.5
    d["n_nearest"] = 5
    d["kind"] = "quicksort"
    d["sorted"] = True
    return d


@pytest.fixture
def make_grispy():
    def _make(dim=2, N_cells=8, isperiodic=False):
        rng = np.random.default_rng(seed=42)

        data = rng.uniform(0, 100, size=(100, dim))
        periodic = {k: (0, 100) for k in range(dim)} if isperiodic else dict()

        return GriSPy(data, N_cells=N_cells, periodic=periodic)

    return _make


# =========================================================================
# Periodicity fixtures
# =========================================================================


@pytest.fixture
def periodicity_init():
    # Define valid input data
    d = dict()
    d["periodic"] = {0: (0, 100), 1: (0, 100), 2: None}
    d["dim"] = 3
    return d


@pytest.fixture
def make_periodicity():
    def _make(dim):
        periodic = {k: (0, 100) for k in range(dim)}
        return Periodicity(periodic, dim)

    return _make