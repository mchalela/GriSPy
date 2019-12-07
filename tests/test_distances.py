#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from grispy import distances


# =============================================================================
# TESTS
# =============================================================================

def test_euclid_simetric():
    # Distancia a-->b = b-->a (metric='euclid')
    dist1 = distances.euclid(
        c0=np.array([1, 1]),
        centres=np.array([[2, 2]]),
        dim=2)
    dist2 = distances.euclid(
        c0=np.array([2, 2]),
        centres=np.array([[1, 1]]),
        dim=2)
    np.testing.assert_almost_equal(dist1, dist2, decimal=16)


def test_haversine_simetric():
    # Distancia a-->b = b-->a (metric='haversine')
    dist1 = distances.haversine(
        c0=np.array([1, 1]),
        centres=np.array([[2, 2]]),
        dim=2)
    dist2 = distances.haversine(
        c0=np.array([2, 2]),
        centres=np.array([[1, 1]]),
        dim=2)
    np.testing.assert_almost_equal(dist1, dist2, decimal=10)


def test_vincenty_simetric():
    # Distancia a-->b = b-->a (metric='vincenty')
    dist1 = distances.vincenty(
        c0=np.array([1, 1]),
        centres=np.array([[2, 2]]),
        dim=2)
    dist2 = distances.vincenty(
        c0=np.array([2, 2]),
        centres=np.array([[1, 1]]),
        dim=2)
    np.testing.assert_almost_equal(dist1, dist2, decimal=10)


def test_euclid_decomposition():
    # Distancia a-->c <= a-->b + b-->c (metric='euclid')
    p_a = [1, 1]
    p_b = [2, 2]
    p_c = [1, 2]
    dist_ab = distances.euclid(np.array(p_a), np.array([p_b]), 2)
    dist_ac = distances.euclid(np.array(p_a), np.array([p_c]), 2)
    dist_bc = distances.euclid(np.array(p_b), np.array([p_c]), 2)
    assert dist_ac <= dist_ab + dist_bc


def test_haversine_decomposition():
    # Distancia a-->c <= a-->b + b-->c (metric='haversine')
    p_a = [1, 1]
    p_b = [2, 2]
    p_c = [1, 2]
    dist_ab = distances.haversine(np.array(p_a), np.array([p_b]), 2)
    dist_ac = distances.haversine(np.array(p_a), np.array([p_c]), 2)
    dist_bc = distances.haversine(np.array(p_b), np.array([p_c]), 2)
    assert dist_ac <= dist_ab + dist_bc


def test_vincenty_decomposition():
    # Distancia a-->c <= a-->b + b-->c (metric='vincenty')
    p_a = [1, 1]
    p_b = [2, 2]
    p_c = [1, 2]
    dist_ab = distances.vincenty(np.array(p_a), np.array([p_b]), 2)
    dist_ac = distances.vincenty(np.array(p_a), np.array([p_c]), 2)
    dist_bc = distances.vincenty(np.array(p_b), np.array([p_c]), 2)
    assert dist_ac <= dist_ab + dist_bc


def test_euclid_gt0():
    # Distancias >= 0 (metric='euclid')
    random = np.random.RandomState(42)
    data = random.uniform(-10, 10, size=(10, 2))
    centre_0 = random.uniform(-10, 10, size=(2,))
    dist = distances.euclid(centre_0, data, 2)
    assert (dist >= 0).all()


def test_haversine_gt0():
    # Distancias >= 0 (metric='haversine')
    random = np.random.RandomState(42)
    data = random.uniform(-10, 10, size=(10, 2))
    centre_0 = random.uniform(-10, 10, size=(2,))
    dist = distances.haversine(centre_0, data, 2)
    assert (dist >= 0).all()


def test_vincenty_gt0():
    # Distancias >= 0 (metric='vincenty')
    random = np.random.RandomState(42)
    data = random.uniform(-10, 10, size=(10, 2))
    centre_0 = random.uniform(-10, 10, size=(2,))
    dist = distances.vincenty(centre_0, data, 2)
    assert (dist >= 0).all()


def test_euclid_not_nan():
    # Distancias != NaN (metric='euclid')
    random = np.random.RandomState(42)
    data = random.uniform(-10, 10, size=(10, 2))
    centre_0 = random.uniform(-10, 10, size=(2,))
    dist = distances.euclid(centre_0, data, 2)
    assert not np.isnan(dist).any()


def test_haversine_not_nan():
    # Distancias != NaN (metric='haversine')
    random = np.random.RandomState(42)
    data = random.uniform(-10, 10, size=(10, 2))
    centre_0 = random.uniform(-10, 10, size=(2,))
    dist = distances.haversine(centre_0, data, 2)
    assert not np.isnan(dist).any()


def test_vincenty_not_nan():
    # Distancias != NaN (metric='vincenty')
    random = np.random.RandomState(42)
    data = random.uniform(-10, 10, size=(10, 2))
    centre_0 = random.uniform(-10, 10, size=(2,))
    dist = distances.vincenty(centre_0, data, 2)
    assert not np.isnan(dist).any()
