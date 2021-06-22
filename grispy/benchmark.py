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

import attr

from grispy import GriSPy


# =============================================================================
# GRISPY PARAMS
# =============================================================================

# Default variable params
NPOINTS = 10 ** np.arange(3, 8)
NCENTRES = 10 ** np.arange(0, 5)
NCELLS = 2 ** np.arange(2, 8)

# Constant params
DOMAIN = (0, 100)
upper_radii = 5.
lower_radii = 2.
n_nearest = 100
periodic = {} 


# =============================================================================
# TIME BENCHMARK
# =============================================================================

@attr.s(frozen=True)
class TimeReport:

    df = attr.ib(repr=False)

    def plot(self):

        return


def generate_points(n_data, n_centres, dim, seed=None):
    """Generate uniform random distributions."""
    
    low, high = DOMAIN

    rng = np.random.default_rng(seed=seed)
    data = rng.uniform(low, high, size=(n_data, dim))
    centres = rng.uniform(low, high, size=(n_centres, dim))
    
    return data, centres

@timer
def build_call(build_kwargs):
    """Initialize GriSPy."""
    return GriSPy(**build_kwargs)

@timer
def query_call(gsp, query_kwargs):
    """Query for neighbohrs."""
    return gsp.bubble_neighbors(**query_kwargs)

def time_benchmark(
    n_data=NDATA,
    n_centres=NCENTRES,
    n_cells=NCELLS,
    dim=3,
    seed=None,
    repeats=1,
    n_jobs=-1,
):
    """Create time benchmark statistics."""

    # Empty report
    report = []

    # Compute the parameter space
    ndata, ncent, ncell = np.meshgrid(n_data, n_centres, n_cells)

    for ndt, nct, ncl in zip(ndata.flat, ncent.flat, ncell.flat):
        data, centres = generate_points(npt, nct, dim, seed)
        
        build_kwargs = {"data": data, "N_cells": ncl}
        query_kwargs = {"centres": centres, "distance_upper_bound": upper_radii}

        bt, gsp = build_call(build_kwargs)
        qt, out = query_call(gsp, query_kwargs)

        report.append([ndt, nct, ncl, bt, qt])

    # Prepare report data frame
    col_names = ['n_data', 'n_centres', 'n_cells', 'BT', 'QT']
    df = pd.DataFrame(report, columns=col_names)

    return TimeReport(df)
