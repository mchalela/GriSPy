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

import itertools as it

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
        self.low_edges, self.high_edges = self.edges_asarray()

    def _modulus(self, x, length):
        """This returns x traslated to the interval (0, length)."""
        mod = x % length
        center_mod = mod - length * (mod // ((length + 1) // 2))
        return center_mod + length / 2

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def isperiodic(self):
        """Return True if at least one dimension is periodic."""
        return any([x is not None for x in list(self.edges.values())])

    @property
    def periodic_edges(self):
        """Return only periodic edges."""
        return {k: v for k, v in self.edges.items() if v is not None}

    @property
    def nonperiodic_edges(self):
        """Return only non-periodic edges."""
        return {k: v for k, v in self.edges.items() if v is None}

    # =========================================================================
    # METHODS
    # =========================================================================

    def multiplicity(self, levels=1):
        """Number of image points per real point."""
        num_levels = 2 * levels + 1
        num_pe = len(self.periodic_edges)
        return num_levels ** num_pe - 1

    def ranges(self, nonperiodic_fill_value=np.inf):
        """Return the range of each dimension."""
        low, high = self.edges_asarray()
        diff = high - low
        diff = np.where(np.isfinite(diff), diff, nonperiodic_fill_value)
        return diff

    def imaging_matrix(self, levels=1):
        """Create the matrix that traslates real points to image points."""
        base = tuple(range(-levels, levels + 1))

        list_ = []
        for i in range(self.dim):
            list_.append(base if self.edges[i] else (0,))

        matrix = np.asarray(list(it.product(*list_)))
        zeros_idx = self.multiplicity(levels) // 2
        return np.delete(matrix, zeros_idx, axis=0)

    def edges_asarray(self):
        """Create two arrays with the periodic edges."""
        low = np.full((1, self.dim), -np.inf)
        high = np.full((1, self.dim), np.inf)

        for k in range(self.dim):
            k_edges = self.edges.get(k)
            if k_edges:
                low[0, k], high[0, k] = k_edges
        return low, high

    def mirror(self, points, levels=1):
        """Generate Terran points in the Mirror Universe."""
        ranges = self.ranges(nonperiodic_fill_value=0.0)
        matrix = self.imaging_matrix(levels)

        mirror_points = np.repeat(points, self.multiplicity(levels), axis=0)
        tiled_matrix = np.tile(matrix.T, len(points)).T

        return mirror_points + ranges * tiled_matrix

    def wrap(self, points):
        """Compute inside-domain coords of points that are outside."""
        wrapped_points = points.copy()
        for k, (low, high) in self.periodic_edges.items():
            length = high - low
            wrapped_points[:, k] = self._modulus(points[:, k], length) + low
        return wrapped_points
