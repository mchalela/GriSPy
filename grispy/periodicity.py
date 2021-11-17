#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Periodicity module."""

# =============================================================================
# IMPORTS
# =============================================================================

# import itertools

import attr

import numpy as np

# from . import validators as vlds

# =============================================================================
# EXPERIMENTAL
# =============================================================================
#
# The idea is to move all periodicity methods inside GriSPy into a new class
# that handles all periodicity related tasks.
#


def complete_periodicity_edges(incomplete_edges, dim):
    """Take a half constructed periodicity dictionary and complete it."""
    default_edges = {key: None for key in range(dim)}
    default_edges.update(incomplete_edges)
    return default_edges


@attr.s
class Periodicity:
    """Handle the periodicity of a k-dimensional domain.

    Parameters
    ----------
    edges: dict
        Dictionary indicating if the data domain is periodic in some or all
        its dimensions. The key is an integer that correspond to the number of
        dimensions, going from 0 to k-1. The value is a tuple with the domain
        limits. If an axis is not specified, or if its value is None, it will
        be considered as non-periodic.
        Important: The periodicity only works within one periodic range.
        Default: all axis set to None.
        Example, edges = { 0: (0, 100), 1: None}.
    dim: int
        The dimension of a single data-point.

    """

    edges = attr.ib()
    dim = attr.ib()

    def __attrs_post_init__(self):
        self.edges = complete_periodicity_edges(self.edges, self.dim)

        self.low_edges, self.high_edges = self.edges_arrays()

        if self.isperiodic:
            self._unraveled_edges = self._unravel_edges()
            self._unraveled_direction = np.sign(self._unraveled_edges)

    def _unravel_edges(self, periodic, dim):
        """Cleanup the periodicity configuration.

        Remove the unnecessary axis from the periodic dict and also creates
        a configuration for use in the search.

        """

        unraveled_edges = []
        for k in range(dim):
            k_edges = periodic.get(k)

            if k_edges:
                k_edges = np.insert(k_edges, 1, 0.0)
            else:
                k_edges = np.zeros((1, 3))

            tiled_k_edges = np.tile(
                k_edges, (3 ** (dim - 1 - k), 3 ** k)
            ).T.ravel()
            unraveled_edges = np.hstack([unraveled_edges, tiled_k_edges])

        unraveled_edges = unraveled_edges.reshape(dim, 3 ** dim).T
        unraveled_edges -= unraveled_edges[::-1]
        unraveled_edges = np.unique(unraveled_edges, axis=0)

        mask = unraveled_edges.sum(axis=1, dtype=bool)
        unraveled_edges = unraveled_edges[mask]

        return unraveled_edges

    @property
    def isperiodic(self):
        if not hasattr(self, "_isperiodic"):
            self._isperiodic = any(
                [x is not None for x in list(self.edges.values())]
            )
        return self._isperiodic

    def edges_asarray(self):
        """Create two arrays with the periodic edges."""
        low = np.full((1, self.dim), -np.inf)
        high = np.full((1, self.dim), np.inf)

        for k in range(self.dim):
            k_edges = self.edges.get(k)

            if k_edges:
                low[0, k] = k_edges[0]
                high[0, k] = k_edges[1]
        return low, high

    def mirror(self, points):
        """Generate Terran points in the Mirror Universe."""
        pass

    def near_edge(self, points, distance):
        """Check if points are near a boundary for a given distance."""
        pass

    def near_edge_mirror(self, points, distance):
        """Generate Terran points in the Mirror Universe if close to edges."""
        pass

    def wrap(self, points):
        """Compute inside-domain coords of points that are outside."""
        pass
