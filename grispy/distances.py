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

"""Distances implementations for GriSPy."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
from scipy.spatial.distance import cdist

# =============================================================================
# FUNCTIONS
# =============================================================================


def euclid(c0, centres, dim):
    """Classic Euclidean distance.

    In mathematics, the Euclidean distance or Euclidean metric is the
    "ordinary" straight-line distance between two points in Euclidean space.
    With this distance, Euclidean space becomes a metric space. The associated
    norm is called the Euclidean norm. Older literature refers to the metric as
    the Pythagorean metric. A generalized term for the Euclidean norm is the L2
    norm or L2 distance. More info:
    https://en.wikipedia.org/wiki/Euclidean_distance

    """
    c0 = c0.reshape((-1, dim))
    d = cdist(c0, centres).reshape((-1,))
    return d


def haversine(c0, centres, dim):
    """Distance using the Haversine formulae.

    The haversine formula determines the great-circle distance between two
    points on a sphere given their longitudes and latitudes. Important in
    navigation, it is a special case of a more general formula in spherical
    trigonometry, the law of haversines, that relates the sides and angles of
    spherical triangles. More info:
    https://en.wikipedia.org/wiki/Haversine_formula

    """
    lon1 = np.deg2rad(c0[0])
    lat1 = np.deg2rad(c0[1])
    lon2 = np.deg2rad(centres[:, 0])
    lat2 = np.deg2rad(centres[:, 1])

    sdlon = np.sin((lon2 - lon1) / 2.0)
    sdlat = np.sin((lat2 - lat1) / 2.0)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)
    num1 = sdlat ** 2
    num2 = clat1 * clat2 * sdlon ** 2
    sep = 2 * np.arcsin(np.sqrt(num1 + num2))
    return np.rad2deg(sep)


def vincenty(c0, centres, dim):
    """Calculate distance  on the surface of a spheroid with Vincenty Formulae.

    Vincenty's formulae are two related iterative methods used in geodesy to
    calculate the distance between two points on the surface of a spheroid,
    developed by Thaddeus Vincenty (1975a). They are based on the assumption
    that the figure of the Earth is an oblate spheroid, and hence are more
    accurate than methods that assume a spherical Earth, such as great-circle
    distance. More info: https://en.wikipedia.org/wiki/Vincenty%27s_formulae

    """
    lon1 = np.deg2rad(c0[0])
    lat1 = np.deg2rad(c0[1])
    lon2 = np.deg2rad(centres[:, 0])
    lat2 = np.deg2rad(centres[:, 1])

    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)
    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon
    sep = np.arctan2(np.sqrt(num1 ** 2 + num2 ** 2), denominator)
    return np.rad2deg(sep)
