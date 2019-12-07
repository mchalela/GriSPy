#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


"""Functions to validate GriSPy input parameters."""

import numpy as np
import os.path

# ---------------------------------
# Validators for method params
# Meant to be called within each method
# --------------------------------


def validate_periodicity(periodic):
    """Validate method params: periodic.

    Periodicity has a differnt validator format
    because it can be changed in set_periodicity,
    so we only validate it there
    """
    # Chek if dict
    if not isinstance(periodic, dict):
        raise TypeError(
            "Periodicity: Argument must be a dictionary. "
            "Got instead type {}".format(type(periodic))
        )

    # If dict is empty means no periodicity, stop validation.
    if len(periodic) == 0:
        return

    # Check if keys are valid
    for k in periodic.keys():
        # Check if integer
        if not isinstance(k, int):
            raise TypeError(
                "Periodicity: Keys must be integers. "
                "Got instead type {}".format(type(k))
            )
        # Check if positive. No raise because negative values may work
        if k < 0:
            print(
                "WARNING: I got a negative periodic axis. "
                "Yo better know what you are doing."
            )
    # Check if values are valid
    for v in periodic.values():
        # Check if tuple or None
        if not (isinstance(v, tuple) or v is None):
            raise TypeError(
                "Periodicity: Values must be tuples. "
                "Got instead type {}".format(type(v))
            )
        if v is None:
            continue

        # Check if edges are valid numbers
        if not (
            isinstance(v[0], (int, float)) and isinstance(v[1], (int, float))
        ):
            raise TypeError(
                "Periodicity: Argument must be a tuple of "
                "2 real numbers as edge descriptors. "
            )

        # Check that first number is lower than second
        if not v[0] < v[1]:
            raise ValueError(
                "Periodicity: First argument in tuple must be "
                "lower than second argument."
            )


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

    # check that n is not larger than the number of data points
    # within 1 periodic range
    Np = len(data)
    valid_axis = len([v for v in periodic.values() if v is not None])
    Nvalid = Np * 3**valid_axis
    if n > Nvalid:
        raise ValueError(
            "Nth-nearest: Argument must be lower than the number of "
            "available data points within 1 periodic range, {}. "
            "Got instead {}".format(Nvalid, n)
        )


def validate_filename(file):
    """Chek if string."""
    if not isinstance(file, str):
        raise TypeError(
            "File: Argument must be a string. "
            "Got instead type {}".format(type(file))
        )


def validate_canwrite(file, overwrite):
    """Check if file is valid."""
    if not overwrite and os.path.isfile(file):
        raise FileExistsError(
            "The file {} already exists. "
            "You may want to use the keyword overwrite=True.".format(file)
        )
