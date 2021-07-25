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

from timeit import Timer

import matplotlib.pyplot as plt

import attr

from grispy import GriSPy


# =============================================================================
# GRISPY PARAMS
# =============================================================================

# Default parameter space
NDATA = [10_000, 100_000, 1_000_000]
NCENTRES = [10, 100, 1_000]
NCELLS = [4, 8, 16, 32, 64]

# Constant params
DOMAIN = (0, 100)
UPPER_RADII = 5.0
LOWER_RADII = 2.0
N_NEAREST = 100
PERIODICITY = {}

# Timer statements
BUILD_STATEMENT = "GriSPy(**build_kwargs)"
QUERY_STATEMENT = "gsp.bubble_neighbors(**query_kwargs)"

# Others
NS2S = 1e-9  # Nanoseconds to seconds factor


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def stats(values):
    """Return basic stats for list of values."""
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
    # random generator
    rng = np.random.default_rng(seed=seed)
    data = rng.uniform(low, high, size=(n_data, dim))
    centres = rng.uniform(low, high, size=(n_centres, dim))
    return data, centres


# =============================================================================
# TIME BENCHMARK
# =============================================================================


@attr.s(frozen=True)
class TimeReport:
    """Construct a time report for the time benchmark."""

    report = attr.ib(validator=attr.validators.instance_of(pd.DataFrame))
    axes = attr.ib(factory=dict)
    metadata = attr.ib(factory=dict)

    def __getitem__(self, item):
        """x[y] <==> x.__getitem__(y)."""
        return self.report.__getitem__(item)

    def __getattr__(self, a):
        """getattr(x, y) <==> x.__getattr__(y)."""
        return getattr(self.report, a)

    def _plot_row(self, gby, axes):
        """Single row plot for BT, QT, TT."""
        ax_bt, ax_qt, ax_tt = axes
        for ngr in gby:
            name, gr = ngr
            ncells, bt, qt = gr["n_cells"], gr["BT_mean"], gr["QT_mean"]

            ax_bt.plot(ncells, bt, "-", label=name)
            ax_qt.plot(ncells, qt, "-", label=name)
            ax_tt.plot(ncells, bt + qt, "-", label=name)
        [ax.semilogy() for ax in axes]
        return

    def plot(self, ax=None):
        """Time benchmark plot."""

        if ax is None:
            _, ax = plt.subplots(2, 3, sharex=True, figsize=(10, 14))

        # First row: fixed n_centres at higher value.
        fix_n_centres = self.report["n_centres"].max()
        gby = (
            self.report.groupby("n_centres")
            .get_group(fix_n_centres)
            .groupby("n_data")
        )
        self._plot_row(gby, axes=ax[0])

        # Second row: fixed n_data at higher value.
        fix_n_data = self.report["n_data"].max()
        gby = (
            self.report.groupby("n_data")
            .get_group(fix_n_data)
            .groupby("n_centres")
        )
        self._plot_row(gby, axes=ax[1])

        return ax


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
    axes = {"n_data": n_data, "n_centres": n_centres, "n_cells": n_cells}
    metadata = {"dim": dim, "repeats": repeats, "n_jobs": n_jobs, "seed": seed}
    report = []

    # Compute the parameter space
    pdict = {"n_data": n_data, "n_centres": n_centres, "n_cells": n_cells}
    grid = parameter_grid(pdict)

    for p in grid:
        ndt, nct, ncl = p["n_data"], p["n_centres"], p["n_cells"]

        # Prepare grispy inputs
        data, centres = generate_points(ndt, nct, dim, seed)
        build_kwargs = {"data": data, "N_cells": int(ncl)}
        query_kwargs = {
            "centres": centres,
            "distance_upper_bound": UPPER_RADII,
        }

        # Initialize Timers
        build_globals = {"GriSPy": GriSPy, "build_kwargs": build_kwargs}
        build_timer = Timer(
            stmt=BUILD_STATEMENT, globals=build_globals, timer=timer_ns
        )

        gsp = GriSPy(**build_kwargs)
        query_globals = {"gsp": gsp, "query_kwargs": query_kwargs}
        query_timer = Timer(
            stmt=QUERY_STATEMENT, globals=query_globals, timer=timer_ns
        )

        # Compute times
        build_time = build_timer.repeat(repeat=repeats, number=1)
        query_time = query_timer.repeat(repeat=repeats, number=1)

        # Save time values. Convert nanoseconds to seconds.
        bt_mean, bt_std = stats(build_time) * NS2S
        qt_mean, qt_std = stats(query_time) * NS2S
        report.append([ndt, nct, ncl, bt_mean, qt_mean, bt_std, qt_std])

    # Prepare report data frame
    col_names = [
        "n_data",
        "n_centres",
        "n_cells",
        "BT_mean",
        "QT_mean",
        "BT_std",
        "QT_std",
    ]
    df = pd.DataFrame(report, columns=col_names)

    return TimeReport(report=df, axes=axes, metadata=metadata)
