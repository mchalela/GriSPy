#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


"""Functions to validate GriSPy input parameters."""

import numpy as np

# ---------------------------------
# Validators for method params
# Meant to be called within each method
# --------------------------------


def validate_digits(digits, N_cells):
    """Validate method params: digits."""
    # Check if inside the grid
    if np.any(digits < 0) or np.any(digits >= N_cells):
        raise ValueError(f"Digits: values must be in the range 0-{N_cells}.")


def validate_ids(ids, size):
    """Validate method params: ids."""
    # Check if inside the grid
    if np.any(ids < 0) or np.any(ids >= size):
        raise ValueError(f"Ids: values must be in the range 0-{size}.")


def validate_centres(centres, data):
    """Validate method params: centres."""
    # Chek if numpy array
    if not isinstance(centres, np.ndarray):
        raise TypeError(
            "Centres: Argument must be a numpy array."
            "Got instead type {}".format(type(centres))
        )

    # Check if data has the expected dimension
    if centres.ndim != 2 or centres.shape[1] != data.shape[1]:
        raise ValueError(
            "Centres: Array has the wrong shape. Expected shape of (n, {}), "
            "got instead {}".format(data.ndim, centres.shape)
        )
    # Check if data has the expected dimension
    if len(centres.flatten()) == 0:
        raise ValueError("Centres: Array must have at least 1 point")

    # Check if every data point is valid
    if not np.isfinite(centres).all():
        raise ValueError("Centres: Array must have real numbers")


def validate_equalsize(a, b):
    """Check if two arrays have the same lenght."""
    if len(a) != len(b):
        raise ValueError("Arrays must have the same lenght.")


def validate_distance_bound(distance, periodic):
    """Distance bounds, upper and lower, can be scalar or numpy array."""
    # Check if type is valid
    if not (np.isscalar(distance) or isinstance(distance, np.ndarray)):
        raise TypeError(
            "Distance: Must be either a scalar or a numpy array."
            "Got instead type {}".format(type(distance))
        )

    # Check if value is valid
    if not np.all(distance >= 0):
        raise ValueError("Distance: Must be positive.")

    # Check distance is not larger than periodic range
    for v in periodic.values():
        if v is None:
            continue
        if np.any(distance > (v[1] - v[0])):
            raise ValueError(
                "Distance can not be higher than the periodicity range"
            )


def validate_shell_distances(lower_bound, upper_bound, periodic):
    """Distance bounds, upper and lower, can be scalar or numpy array."""
    validate_distance_bound(lower_bound, periodic)
    validate_distance_bound(upper_bound, periodic)

    # Check that lower_bound is lower than upper_bound
    if not np.all(lower_bound <= upper_bound):
        raise ValueError(
            "Distance: Lower bound must be lower than higher bound."
        )


def validate_bool(flag):
    """Check if bool."""
    if not isinstance(flag, bool):
        raise TypeError(
            "Flag: Expected boolean. " "Got instead type {}".format(type(flag))
        )


def validate_sortkind(kind):
    """Define valid sorting algorithm names."""
    valid_kind_names = ["quicksort", "mergesort", "heapsort", "stable"]

    # Chek if string
    if not isinstance(kind, str):
        raise TypeError(
            "Kind: Sorting name must be a string. "
            "Got instead type {}".format(type(kind))
        )

    # Check if name is valid
    if kind not in valid_kind_names:
        raise ValueError(
            "Kind: Got an invalid name: '{}'. "
            "Options are: {}".format(kind, valid_kind_names)
        )


def validate_n_nearest(n, data, periodic):
    """Validate method params: n_nearest."""
    # Chek if int
    if not isinstance(n, int):
        raise TypeError(
            "Nth-nearest: Argument must be an integer. "
            "Got instead type {}".format(type(n))
        )
    # Check if number is valid, i.e. higher than 1
    if n < 1:
        raise ValueError(
            "Nth-nearest: Argument must be higher than 1. "
            "Got instead {}".format(n)
        )
