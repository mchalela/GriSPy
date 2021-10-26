#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE

"""Grid Search in Python.

GriSPy is a regular grid search algorithm for quick nearest-neighbor lookup.
"""

__all__ = ["Grid", "GriSPy"]


__version__ = "0.2.0"


# =============================================================================
# IMPORTS
# =============================================================================

from .core import Grid, GriSPy
