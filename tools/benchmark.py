#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


"""Functions to benchmark GriSPy methods."""

import itertools
import json
import os
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
NDATA = [10_000, 100_000, 1_000_000, 10_000_000]
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

# Columns of the timing report DataFrame.
REPORT_COLUMNS = [
    "n_data",
    "n_centres",
    "n_cells",
    "BT_mean",
    "QT_mean",
    "BT_std",
    "QT_std",
]


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
    This format is similar to sklearn ParameterGrid.

    Uses itertools.product so each value keeps its native Python type
    (no implicit int->float promotion as with np.meshgrid).
    """
    keys = list(parameters)
    grid = []
    for combo in itertools.product(*parameters.values()):
        grid.append(dict(zip(keys, combo)))
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

    def fix_and_group(self, fixed_col):
        varied_col = "n_centres" if fixed_col == "n_data" else "n_data"

        fixed_value = self.report[fixed_col].max()
        gby = (
            self.report.groupby(fixed_col)
            .get_group(fixed_value)
            .groupby(varied_col)
        )
        return gby, fixed_value

    # =====================================================
    # PLOTTING METHODS
    # =====================================================

    def _plot_row(self, gby, axes, label, legend_title, logy):
        """Single row plot for BT, QT, TT."""
        ax_bt, ax_qt, ax_tt = axes
        for name, gr in gby:
            ncells, bt, qt = gr["n_cells"], gr["BT_mean"], gr["QT_mean"]
            bt_std, qt_std = gr["BT_std"], gr["QT_std"]

            tt = bt + qt
            tt_std = (bt_std ** 2 + qt_std ** 2) ** 0.5

            label_name = f"{label} = {name}"
            line = ax_bt.plot(ncells, bt, "-", label=label_name)
            color = line[0].get_color()
            ax_bt.errorbar(ncells, bt, yerr=bt_std, fmt="None", ecolor=color)

            line = ax_qt.plot(ncells, qt, "-", label=label_name)
            color = line[0].get_color()
            ax_qt.errorbar(ncells, qt, yerr=qt_std, fmt="None", ecolor=color)

            line = ax_tt.plot(ncells, tt, "-", label=label_name)
            color = line[0].get_color()
            ax_tt.errorbar(ncells, tt, yerr=tt_std, fmt="None", ecolor=color)

        titles = ["BT", "QT", "TT"]
        for i, ax in enumerate(axes):
            ax.set_title(titles[i])
            ax.legend(title=legend_title)
            ax.set_xlabel("n_cells")
            if logy:
                ax.semilogy()
            else:
                ax.axhline(0, c="gray", linestyle="--", zorder=0)
        axes[0].set_ylabel("Time [sec]")
        return

    def plot(self, ax=None, logy=None):
        """Time benchmark plot.

        By default the y-scale is chosen automatically: logarithmic for an
        absolute time report (all positive, spanning several orders of
        magnitude) and linear for a difference report from `diff_report`
        (signed values that cannot be log-scaled). Pass `logy=True/False`
        to force a scale.
        """
        # A difference report has no plain `version` in its metadata
        # (diff_report stores version_a/version_b instead).
        is_diff = "version" not in self.metadata
        if logy is None:
            logy = not is_diff

        if ax is None:
            _, ax = plt.subplots(2, 3, figsize=(10, 14))

        # First row: fixed n_centres at higher value.
        gby, fixed_value = self.fix_and_group(fixed_col="n_centres")
        legend_title = f"Fixed: n_centres={fixed_value}"
        self._plot_row(
            gby,
            axes=ax[0],
            label="n_data",
            legend_title=legend_title,
            logy=logy,
        )

        # Second row: fixed n_data at higher value.
        gby, fixed_value = self.fix_and_group(fixed_col="n_data")
        legend_title = f"Fixed: n_data={fixed_value}"
        self._plot_row(
            gby,
            axes=ax[1],
            label="n_centres",
            legend_title=legend_title,
            logy=logy,
        )
        version = "diff" if is_diff else self.metadata["version"]
        fig = ax[0, 0].figure
        fig.suptitle(f"GriSPy {version} time report")
        return ax

    # =====================================================
    # JSON REPORT
    # =====================================================

    def save_report(self, filename=None, overwrite=False):
        """Write this report to a JSON file.

        The file stores the metadata, axes and the timing table as separate
        keys, so it is self-describing and version-independent: it can be
        loaded and compared with `load_report`/`diff_report` from any
        environment with pandas, without importing grispy or unpickling a
        version-specific class.
        """
        if filename is None:
            filename = f"benchmark_v{grispy_version}.json"

        if os.path.isfile(filename) and not overwrite:
            raise FileExistsError(
                f"File `{filename}` already exist. "
                "You may want to use `overwrite=True`."
            )

        payload = {
            "metadata": self.metadata,
            "axes": self.axes,
            "report": self.report.to_dict(orient="records"),
        }
        with open(filename, mode="w") as fp:
            json.dump(payload, fp, indent=2)


def load_report(filename):
    """Load a TimeReport from a JSON file written by `save_report`.

    No pickle, no grispy import required.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File `{filename}` not found.")

    with open(filename, mode="r") as fp:
        payload = json.load(fp)

    report = pd.DataFrame(payload["report"], columns=REPORT_COLUMNS)
    return TimeReport(
        report=report,
        axes=payload.get("axes", {}),
        metadata=payload.get("metadata", {}),
    )


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
    seed=42,
):
    """Create time benchmark statistics.

    For every point in the parameter grid the build and query times are
    measured over ``repeats`` independent data realizations: each realization
    regenerates the data and centres from a distinct, deterministically
    derived seed, so the reported standard deviation reflects true
    realization-to-realization variance (not timer jitter on a single
    dataset). The whole run is reproducible from the single master ``seed``.
    """
    # Report which version is being measured (useful when running this tool
    # standalone against an older installed GriSPy to compare versions).
    print(
        f"\033[95m\033[1m"
        f"Benchmarking GriSPy version = {grispy_version}"
        f"\033[0m"
    )

    # Set timer in units of nanoseconds
    timer_ns = time.perf_counter_ns

    # Report axes and self-describing metadata.
    axes = {"n_data": n_data, "n_centres": n_centres, "n_cells": n_cells}
    metadata = {
        "version": grispy_version,
        "dim": dim,
        "repeats": repeats,
        "seed": seed,
        "query": "bubble_neighbors",
        "metric": "euclid",
        "radius": UPPER_RADII,
        "domain": list(DOMAIN),
    }
    report = []

    # Derive `repeats` independent, reproducible seeds from the master seed.
    # Each realization uses its own child seed.
    realization_seeds = np.random.SeedSequence(seed).spawn(repeats)

    # Compute the parameter space
    pdict = {"n_data": n_data, "n_centres": n_centres, "n_cells": n_cells}
    grid = parameter_grid(pdict)

    for p in grid:
        ndt, nct, ncl = p["n_data"], p["n_centres"], p["n_cells"]

        build_time, query_time = [], []
        for rseed in realization_seeds:
            # New data realization for each repeat.
            data, centres = generate_points(ndt, nct, dim, rseed)
            build_kwargs = {"data": data, "N_cells": ncl}
            query_kwargs = {
                "centres": centres,
                "distance_upper_bound": UPPER_RADII,
            }

            # Time a single build on this realization.
            build_globals = {"GriSPy": GriSPy, "build_kwargs": build_kwargs}
            build_timer = Timer(
                stmt=BUILD_STATEMENT, globals=build_globals, timer=timer_ns
            )
            build_time.append(build_timer.timeit(number=1))

            # Time a single query on this realization.
            gsp = GriSPy(**build_kwargs)
            query_globals = {"gsp": gsp, "query_kwargs": query_kwargs}
            query_timer = Timer(
                stmt=QUERY_STATEMENT, globals=query_globals, timer=timer_ns
            )
            query_time.append(query_timer.timeit(number=1))

        # Save time values. Convert nanoseconds to seconds.
        bt_mean, bt_std = stats(build_time) * NS2S
        qt_mean, qt_std = stats(query_time) * NS2S
        report.append([ndt, nct, ncl, bt_mean, qt_mean, bt_std, qt_std])

    # Prepare report data frame
    df = pd.DataFrame(report, columns=REPORT_COLUMNS)

    return TimeReport(report=df, axes=axes, metadata=metadata)
