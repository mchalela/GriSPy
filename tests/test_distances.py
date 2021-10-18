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

from scipy.spatial.distance import cdist

import textdistance

from grispy import distances, GriSPy


# =============================================================================
# TESTS
# =============================================================================


def test_euclid_simetric():
    # Distancia a-->b = b-->a (metric='euclid')
    dist1 = distances.euclid(
        c0=np.array([1, 1]), centres=np.array([[2, 2]]), dim=2
    )
    dist2 = distances.euclid(
        c0=np.array([2, 2]), centres=np.array([[1, 1]]), dim=2
    )
    np.testing.assert_almost_equal(dist1, dist2, decimal=16)


def test_haversine_simetric():
    # Distancia a-->b = b-->a (metric='haversine')
    dist1 = distances.haversine(
        c0=np.array([1, 1]), centres=np.array([[2, 2]]), dim=2
    )
    dist2 = distances.haversine(
        c0=np.array([2, 2]), centres=np.array([[1, 1]]), dim=2
    )
    np.testing.assert_almost_equal(dist1, dist2, decimal=10)


def test_vincenty_simetric():
    # Distancia a-->b = b-->a (metric='vincenty')
    dist1 = distances.vincenty(
        c0=np.array([1, 1]), centres=np.array([[2, 2]]), dim=2
    )
    dist2 = distances.vincenty(
        c0=np.array([2, 2]), centres=np.array([[1, 1]]), dim=2
    )
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


def test_custom_distance_lev():
    def levenshtein(c0, centres, dim):
        c0 = tuple(c0)
        distances = np.empty(len(centres))
        for idx, c1 in enumerate(centres):
            c1 = tuple(c1)
            dis = textdistance.levenshtein(c0, c1)
            distances[idx] = dis
        return distances

    random = np.random.RandomState(42)

    Npoints = 10 ** 3
    Ncentres = 2
    dim = 2
    Lbox = 100.0

    data = random.uniform(0, Lbox, size=(Npoints, dim))
    centres = random.uniform(0, Lbox, size=(Ncentres, dim))

    gsp = GriSPy(data, N_cells=20, metric=levenshtein)

    upper_radii = 10.0
    lev_dist, lev_ind = gsp.bubble_neighbors(
        centres, distance_upper_bound=upper_radii
    )

    assert len(centres) == len(lev_dist) == len(lev_ind)
    assert np.all(lev_dist[0] == 2)
    assert np.all(lev_dist[1] == 2)

    assert np.all(
        lev_ind[0]
        == [
            648,
            516,
            705,
            910,
            533,
            559,
            61,
            351,
            954,
            214,
            90,
            645,
            846,
            818,
            39,
            433,
            7,
            700,
            2,
            364,
            547,
            427,
            660,
            548,
            333,
            246,
            193,
            55,
            83,
            159,
            684,
            310,
            777,
            112,
            535,
            780,
            334,
            300,
            467,
            30,
            613,
            564,
            134,
            534,
            435,
            901,
            296,
            800,
            391,
            321,
            763,
            208,
            42,
            413,
            97,
        ]
    )

    assert np.all(
        lev_ind[1]
        == [
            580,
            740,
            498,
            89,
            610,
            792,
            259,
            647,
            58,
            722,
            360,
            685,
            552,
            619,
            6,
            555,
            935,
            268,
            615,
            661,
            680,
            817,
            75,
            919,
            922,
            927,
            52,
            77,
            859,
            70,
            544,
            189,
            340,
            691,
            453,
            570,
            126,
            140,
            67,
            284,
            662,
            590,
            527,
        ]
    )


def test_custom_distance_hamming():
    def hamming(c0, centres, dim):
        c0 = c0.reshape((-1, dim))
        d = cdist(c0, centres, metric="hamming").reshape((-1,))
        return d

    random = np.random.RandomState(42)

    Npoints = 10 ** 3
    Ncentres = 2
    dim = 2
    Lbox = 100.0

    data = random.uniform(0, Lbox, size=(Npoints, dim))
    centres = random.uniform(0, Lbox, size=(Ncentres, dim))

    gsp = GriSPy(data, N_cells=20, metric=hamming)

    upper_radii = 10.0
    ham_dist, ham_ind = gsp.bubble_neighbors(
        centres, distance_upper_bound=upper_radii
    )

    assert len(centres) == len(ham_dist) == len(ham_ind)
    assert np.all(ham_dist[0] == 1)
    assert np.all(ham_dist[1] == 1)

    assert np.all(
        ham_ind[0]
        == [
            648,
            516,
            705,
            910,
            533,
            559,
            61,
            351,
            954,
            214,
            90,
            645,
            846,
            818,
            39,
            433,
            7,
            700,
            2,
            364,
            547,
            427,
            660,
            548,
            333,
            246,
            193,
            55,
            83,
            159,
            684,
            310,
            777,
            112,
            535,
            780,
            334,
            300,
            467,
            30,
            613,
            564,
            134,
            534,
            435,
            901,
            296,
            800,
            391,
            321,
            763,
            208,
            42,
            413,
            97,
        ]
    )

    assert np.all(
        ham_ind[1]
        == [
            580,
            740,
            498,
            89,
            610,
            792,
            259,
            647,
            58,
            722,
            360,
            685,
            552,
            619,
            6,
            555,
            935,
            268,
            615,
            661,
            680,
            817,
            75,
            919,
            922,
            927,
            52,
            77,
            859,
            70,
            544,
            189,
            340,
            691,
            453,
            570,
            126,
            140,
            67,
            284,
            662,
            590,
            527,
        ]
    )
