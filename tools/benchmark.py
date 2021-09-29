#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


"""Functions to benchmark GriSPy methods."""

import os
import pickle
import time
from timeit import Timer

import attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grispy import GriSPy
from grispy import __version__ as grispy_version

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

    # =====================================================
    # PLOTTING METHODS
    # =====================================================

    def _plot_row(self, gby, axes, logy):
        """Single row plot for BT, QT, TT."""
        ax_bt, ax_qt, ax_tt = axes
        for ngr in gby:
            name, gr = ngr
            ncells, bt, qt = gr["n_cells"], gr["BT_mean"], gr["QT_mean"]
            bt_std, qt_std = gr["BT_std"], gr["QT_std"]

            tt = bt + qt
            tt_std = (bt_std ** 2 + qt_std ** 2) ** 0.5

            line = ax_bt.plot(ncells, bt, "-", label=name)
            color = line[0].get_color()
            ax_bt.errorbar(ncells, bt, yerr=bt_std, fmt="None", ecolor=color)

            line = ax_qt.plot(ncells, qt, "-", label=name)
            color = line[0].get_color()
            ax_qt.errorbar(ncells, qt, yerr=qt_std, fmt="None", ecolor=color)

            line = ax_tt.plot(ncells, tt, "-", label=name)
            color = line[0].get_color()
            ax_tt.errorbar(ncells, tt, yerr=tt_std, fmt="None", ecolor=color)

        for ax in axes:
            ax.legend()
            ax.set_xlabel("n_cells")
            if logy:
                ax.semilogy()
            else:
                ax.axhline(0, c="gray", linestyle="--", zorder=0)
        return

    def plot(self, ax=None, logy=True):
        """Time benchmark plot."""

        if ax is None:
            _, ax = plt.subplots(2, 3, figsize=(10, 14))

        # First row: fixed n_centres at higher value.
        fix_n_centres = self.report["n_centres"].max()
        gby = (
            self.report.groupby("n_centres")
            .get_group(fix_n_centres)
            .groupby("n_data")
        )
        self._plot_row(gby, axes=ax[0], logy=logy)

        # Second row: fixed n_data at higher value.
        fix_n_data = self.report["n_data"].max()
        gby = (
            self.report.groupby("n_data")
            .get_group(fix_n_data)
            .groupby("n_centres")
        )
        self._plot_row(gby, axes=ax[1], logy=logy)
        return ax

    # =====================================================
    # PICKLE REPORT
    # =====================================================

    def save_report(self, filename=None, overwrite=False):
        """Write this instance to a file using pickle."""

        if filename is None:
            filename = f"benchmark_v{grispy_version}.pickle"

        if os.path.isfile(filename):
            if overwrite:
                os.remove(filename)
            else:
                raise FileExistsError(
                    f"File `{filename}` already exist. "
                    "You may want to use `overwrite=True`."
                )

        with open(filename, mode="wb") as fp:
            pickle.dump(self, fp)


def load_report(filename):
    """Load a pickled TimeReport instance."""

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File `{filename}` not found.")

    with open(filename, mode="rb") as fp:
        report = pickle.load(fp)
    return report


def diff_report(a, b):
    """Difference of times between two TimeReport instances, diff = a - b.

    Note: Both reports must have the same axes atribute.
    """
    if a.axes != b.axes:
        raise ValueError("Reports axes must be equal for a time comparison.")

    # Time difference = a - b
    new_report = a.report.copy()
    for col in ["BT_mean", "QT_mean"]:
        new_report[col] = a.report[col] - b.report[col]

    # Standard error propagation
    for col in ["BT_std", "QT_std"]:
        new_report[col] = (a.report[col] ** 2 + b.report[col] ** 2) ** 0.5

    # Combine both metadata dicts
    new_metadata = {f"{k}_a": v for k, v in a.metadata.items()}
    new_metadata.update({f"{k}_b": v for k, v in b.metadata.items()})

    return TimeReport(report=new_report, axes=a.axes, metadata=new_metadata)


def time_benchmark(
    n_data=NDATA,
    n_centres=NCENTRES,
    n_cells=NCELLS,
    dim=3,
    repeats=10,
    n_jobs=-1,
    seed=42,
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
