#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


"""Functions to benchmark GriSPy methods."""

import numpy as np

import pandas as pd

import timeit

from grispy import GriSPy


# =============================================================================
# DEFAULT VALUES
# =============================================================================

NPOINTS = 10 ** np.arange(3, 8)

NCENTRES = 10 ** np.arange(0, 5)

NCELLS = 2 ** np.arange(2, 8)


# =============================================================================
# TIME BENCHMARK
# =============================================================================


def time_benchmark(
    Npoints=NPOINTS,
    Ncentres=NCENTRES,
    Ncells=NCELLS,
    seed=None,
    repeats=1,
    n_jobs=-1,
):
    """Create time benchmark statistics."""
    return None # should return pd.DataFrame
