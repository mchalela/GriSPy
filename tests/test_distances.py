#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


import pytest
import numpy as np
from grispy import GriSPy


@pytest.fixture()
def gsp():
    def def_metric(metric):
        data = np.random.uniform(-10, 10, size=(10, 2))
        periodic = {0: None, 1: None}
        return GriSPy(
            data=data,
            N_cells=2,
            copy_data=False,
            periodic=periodic,
            metric=metric,
        )
    return def_metric


def test_distance_A_01(gsp):
    # Distancia a-->b = b-->a (metric='euclid')
    gsp = gsp("euclid")
    dist1 = gsp.distance(np.array([1, 1]), np.array([[2, 2]]))
    dist2 = gsp.distance(np.array([2, 2]), np.array([[1, 1]]))
    np.testing.assert_almost_equal(dist1, dist2, decimal=16)


def test_distance_A_02(gsp):
    # Distancia a-->b = b-->a (metric='sphere')
    gsp = gsp("sphere")
    dist1 = gsp.distance(np.array([1, 1]), np.array([[2, 2]]))
    dist2 = gsp.distance(np.array([2, 2]), np.array([[1, 1]]))
    np.testing.assert_almost_equal(dist1, dist2, decimal=10)


def test_distance_B_01(gsp):
    # Distancia a-->c <= a-->b + b-->c (metric='euclid')
    gsp = gsp("euclid")
    p_a = [1, 1]
    p_b = [2, 2]
    p_c = [1, 2]
    dist_ab = gsp.distance(np.array(p_a), np.array([p_b]))
    dist_ac = gsp.distance(np.array(p_a), np.array([p_c]))
    dist_bc = gsp.distance(np.array(p_b), np.array([p_c]))
    assert dist_ac <= dist_ab + dist_bc


def test_distance_B_02(gsp):
    # Distancia a-->c <= a-->b + b-->c (metric='sphere')
    gsp = gsp("sphere")
    p_a = [1, 1]
    p_b = [2, 2]
    p_c = [1, 2]
    dist_ab = gsp.distance(np.array(p_a), np.array([p_b]))
    dist_ac = gsp.distance(np.array(p_a), np.array([p_c]))
    dist_bc = gsp.distance(np.array(p_b), np.array([p_c]))
    assert dist_ac <= dist_ab + dist_bc


def test_distance_C_01(gsp):
    # Distancias >= 0 (metric='euclid')
    gsp = gsp("euclid")
    centre_0 = np.random.uniform(-10, 10, size=(2,))
    dist = gsp.distance(centre_0, gsp.data)
    assert (dist >= 0).all()


def test_distance_C_02(gsp):
    # Distancias >= 0 (metric='sphere')
    gsp = gsp("sphere")
    centre_0 = np.random.uniform(-10, 10, size=(2,))
    dist = gsp.distance(centre_0, gsp.data)
    assert (dist >= 0).all()


def test_distance_D_01(gsp):
    # Distancias != NaN (metric='euclid')
    gsp = gsp("euclid")
    centre_0 = np.random.uniform(-10, 10, size=(2,))
    dist = gsp.distance(centre_0, gsp.data)
    assert not np.isnan(dist).any()


def test_distance_D_02(gsp):
    # Distancias != NaN (metric='sphere')
    gsp = gsp("sphere")
    centre_0 = np.random.uniform(-10, 10, size=(2,))
    dist = gsp.distance(centre_0, gsp.data)
    assert not np.isnan(dist).any()
