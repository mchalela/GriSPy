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

import time

import timeit

import attr

from grispy import GriSPy


# =============================================================================
# GRISPY PARAMS
# =============================================================================

# Default variable params
NDATA = [1_000,  10_000, 100_000]
NCENTRES = [10, 100]
NCELLS = [2, 4, 8, 16, 32]

# Constant params
DOMAIN = (0, 100)
UPPER_RADII = 5.
LOWER_RADII = 2.
N_NEAREST = 100
PERIODICITY = {} 

# Timer statements
NS2S = 1e-9
BUILD_STATEMENT = "GriSPy(**build_kwargs)"
QUERY_STATEMENT = "gsp.bubble_neighbors(**query_kwargs)"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def stats(values):
    """Basic stats for list of values."""
    mean, std = np.mean(values), np.std(values)
    return np.array([mean, std])


def parameter_grid(parameters):
    """Full parameter space combinations from a dict with iterables.
    parameters = {'A': [34, 56, 567], 'C': [12, 0], 'G': [1245]}
    This format is similar to sklearn ParameterGrid
    """
    keys, values = list(zip(*parameters.items()))
    mesh = np.meshgrid(*values)
    flat_mesh = map(np.ravel, mesh)
    grid = []
    for params in zip(*flat_mesh):
        d = {key: params[i] for i, key in enumerate(keys)}
        grid.append(d)
    return grid


def generate_points(n_data, n_centres, dim, seed=None):
    """Generate uniform random distributions."""
    low, high = DOMAIN
    # set random
    rng = np.random.default_rng(seed=seed)
    data = rng.uniform(low, high, size=(n_data, dim))
    centres = rng.uniform(low, high, size=(n_centres, dim))
    return data, centres


# =============================================================================
# TIME BENCHMARK
# =============================================================================

@attr.s(frozen=True)
class TimeReport:
    """Construct a time report for the time benchmark"""

    report = attr.ib(validator=attr.validators.instance_of(pd.DataFrame), repr=True)
    metadata = attr.ib(factory=dict)

    def __getitem__(self, item):
        """x[y] <==> x.__getitem__(y)."""
        return self.report.__getitem__(item)

    def __getattr__(self, a):
        """getattr(x, y) <==> x.__getattr__(y)."""
        return getattr(self.report, a)


def time_benchmark(
    n_data=NDATA,
    n_centres=NCENTRES,
    n_cells=NCELLS,
    dim=3,
    repeats=10,
    n_jobs=-1,
    seed=None,
):
    """Create time benchmark statistics."""
    # Set timer in units of nanoseconds
    timer_ns = time.perf_counter_ns

    # Empty report and metadata
    metadata = {"dim": dim, "repeats": repeats, "n_jobs": n_jobs, "seed": seed}
    report = []

    # Compute the parameter space
    pdict = {'n_data': n_data, 'n_centres': n_centres, 'n_cells': n_cells}
    grid = parameter_grid(pdict)

    for p in grid:
        ndt, nct, ncl = p['n_data'], p['n_centres'], p['n_cells']

        # Prepare grispy inputs
        data, centres = generate_points(ndt, nct, dim, seed)
        build_kwargs = {"data": data, "N_cells": int(ncl)}
        query_kwargs = {"centres": centres, "distance_upper_bound": UPPER_RADII}

        # Initialize Timers
        build_globals = {"GriSPy": GriSPy, "build_kwargs": build_kwargs}
        build_timer = timeit.Timer(stmt=BUILD_STATEMENT, globals=build_globals, timer=timer_ns)
        
        gsp = GriSPy(**build_kwargs)
        query_globals = {"gsp": gsp, "query_kwargs": query_kwargs}
        query_timer = timeit.Timer(stmt=QUERY_STATEMENT, globals=query_globals, timer=timer_ns)

        # Compute times
        build_time = build_timer.repeat(repeat=repeats, number=1)
        query_time = query_timer.repeat(repeat=repeats, number=1)
        
        # Save time values. Convert nanoseconds to seconds.
        bt_mean, bt_std = stats(build_time) * NS2S
        qt_mean, qt_std = stats(query_time) * NS2S
        report.append([ndt, nct, ncl, bt_mean, qt_mean, bt_std, qt_std])

    # Prepare report data frame
    col_names = ['n_data', 'n_centres', 'n_cells', 'BTmean', 'QTmean', 'BTstd', 'QTstd']
    df = pd.DataFrame(report, columns=col_names)

    return TimeReport(report=df, metadata=metadata)
