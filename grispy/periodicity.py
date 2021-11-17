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

# import numpy as np

# from . import validators as vlds

# =============================================================================
# EXPERIMENTAL
# =============================================================================
#
# The idea is to move all periodicity methods inside GriSPy into a new class
# that handles all periodicity related tasks.
#


def complete_periodicity_boundaries(incomplete_boundaries, dim):
    """Take a half constructed periodicity dictionary and complete it."""
    default_boundaries = {key: None for key in range(dim)}
    default_boundaries.update(incomplete_boundaries)
    return default_boundaries


@attr.s
class Periodicity:
    """Handle the periodicity of a k-dimensional domain.

    Parameters
    ----------
    boundaries: dict
        Dictionary indicating if the data domain is periodic in some or all its
        dimensions. The key is an integer that correspond to the number of
        dimensions, going from 0 to k-1. The value is a tuple with the domain
        limits. If an axis is not specified, or if its value is None, it will be
        considered as non-periodic.
        Important: The periodicity only works within one periodic range.
        Default: all axis set to None.
        Example, periodic = { 0: (0, 360), 1: None}.
    dim: int
        The dimension of a single data-point.

    """

    boundaries = attr.ib()
    dim = attr.ib()

    def __attrs_post_init__(self):
        self.boundaries = complete_periodicity_boundaries(
            self.boundaries, self.dim
        )
