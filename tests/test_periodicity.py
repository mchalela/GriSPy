#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


import numpy as np
import numpy.testing as npt
import pytest

from grispy import Periodicity

# =========================================================
# INITIALIZATION
# =========================================================


def test_invalid_edges_as_string():
    # Axis 0 with invalid type: string instead of dict
    bad_edges = "{0: [-1, 1], 1: (-1, 1), 2: None}"
    with pytest.raises(TypeError):
        Periodicity(edges=bad_edges, dim=3)


def test_invalid_edges_with_list():
    # Axis 0 with invalid value type: list instead of tuple
    bad_edges = {0: [-1, 1], 1: (-1, 1), 2: None}
    with pytest.raises(TypeError):
        Periodicity(edges=bad_edges, dim=3)


def test_invalid_edges_with_string():
    # Axis is not integer
    bad_edges = {"A": (-1, 1), 1: (-1, 1), 2: None}
    with pytest.raises(TypeError):
        Periodicity(edges=bad_edges, dim=3)


def test_invalid_edges_bad_edge_values():
    # Edge 0 is larger than edge 1
    bad_edges = {0: (1, -1), 1: (-1, 1), 2: None}
    with pytest.raises(ValueError):
        Periodicity(edges=bad_edges, dim=3)


def test_invalid_edges_bad_edge_type():
    # Edge has wrong type
    bad_edges = {0: (-1, [1]), 1: (-1, 1), 2: None}
    with pytest.raises(TypeError):
        Periodicity(edges=bad_edges, dim=3)


def test_valid_edges_empty():
    # both initializations should be equivalent
    edges_explicit = {0: None, 1: None, 2: None}
    edges_implicit = dict()

    exp = Periodicity(edges=edges_explicit, dim=3)
    imp = Periodicity(edges=edges_implicit, dim=3)
    assert exp == imp


@pytest.mark.parametrize("bad_dim_type", [float, np.array])
def test_invalid_dim_type(bad_dim_type):
    bad_dim = bad_dim_type(3)
    edges = {0: None, 1: None, 2: None}
    with pytest.raises(TypeError):
        Periodicity(edges=edges, dim=bad_dim)


@pytest.mark.parametrize("bad_dim", [0, 1, 2])
def test_invalid_dim_value(bad_dim):
    # bad_dim = 2
    edges = {0: None, 1: None, 2: None}
    with pytest.raises(ValueError):
        Periodicity(edges=edges, dim=bad_dim)


# =========================================================
# PROPERTIES
# =========================================================


def test_properties_exist():
    edges = {0: None, 1: None, 2: None}
    periodicity = Periodicity(edges=edges, dim=3)
    assert hasattr(periodicity, "isperiodic")
    assert hasattr(periodicity, "periodic_edges")
    assert hasattr(periodicity, "nonperiodic_edges")


def test_property_isperiodic_False():
    edges = {0: None, 1: None, 2: None}
    periodicity = Periodicity(edges=edges, dim=3)
    assert periodicity.isperiodic is False


def test_property_isperiodic_True():
    edges = {0: (0, 1), 1: None, 2: None}
    periodicity = Periodicity(edges=edges, dim=3)
    assert periodicity.isperiodic is True


def test_property_periodic_edges():
    edges = {0: (0, 1), 3: (0, 4)}
    periodicity = Periodicity(edges=edges, dim=4)

    periodic_edges = {0: (0, 1), 3: (0, 4)}
    nonperiodic_edges = {1: None, 2: None}
    assert periodicity.periodic_edges == periodic_edges
    assert periodicity.nonperiodic_edges == nonperiodic_edges


# =========================================================
# METHODS
# =========================================================


def test_multiplicity_bad_levels():
    edges = {k: (0, 100) for k in range(3)}
    periodicity = Periodicity(edges, 3)
    with pytest.raises(ValueError):
        periodicity.multiplicity(1.0)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("levels", [1, 2, 3])
def test_multiplicity(dim, levels):
    expected_mult = (2 * levels + 1) ** dim - 1

    edges = {k: (0, 100) for k in range(dim)}
    periodicity = Periodicity(edges, dim)
    assert periodicity.multiplicity(levels) == expected_mult


def test_ranges_bad_fill_value():
    edges = {k: (0, 100) for k in range(3)}
    periodicity = Periodicity(edges, 3)
    with pytest.raises(TypeError):
        periodicity.ranges(np.array([0.0]))


def test_ranges():
    edges = {k: (0, 100) for k in range(3)}
    periodicity = Periodicity(edges, 3)

    expected = np.array([[100.0, 100.0, 100.0]])
    result = periodicity.ranges()
    npt.assert_array_equal(result, expected)


@pytest.mark.parametrize("fill", [0.0, np.inf, np.nan])
def test_ranges_fill_value(fill):
    edges = {0: (0, 100)}
    periodicity = Periodicity(edges, 3)

    expected = np.array([[100.0, fill, fill]])
    result = periodicity.ranges(fill)
    npt.assert_array_equal(result, expected)


@pytest.mark.parametrize("levels", [-1, 2.0])
def test_imaging_matrix_bad_levels(levels):
    edges = {k: (0, 100) for k in range(2)}
    periodicity = Periodicity(edges, 2)
    with pytest.raises(ValueError):
        periodicity.imaging_matrix(levels)


def test_imaging_matrix():
    edges = {k: (0, 100) for k in range(2)}
    periodicity = Periodicity(edges, 2)

    expected = np.array(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    )
    result = periodicity.imaging_matrix(1)
    npt.assert_array_equal(result, expected)


def test_edges_asarray():
    edges = {k: (0, 100) for k in range(3)}
    periodicity = Periodicity(edges, 3)

    low_exp = np.array([[0, 0, 0]])
    high_exp = np.array([[100, 100, 100]])

    result = periodicity.edges_asarray()
    assert isinstance(result, tuple)
    assert len(result) == 2

    low, high = result
    npt.assert_array_equal(low, low_exp)
    npt.assert_array_equal(high, high_exp)


def test_mirror_1point():
    edges = {k: (0, 100) for k in range(2)}
    periodicity = Periodicity(edges, 2)

    p1 = np.array([[25.0, 25.0]])

    p1_exp_mirrors = np.array(
        [
            [-75.0, -75.0],
            [-75.0, 25.0],
            [-75.0, 125.0],
            [25.0, -75.0],
            [25.0, 125.0],
            [125.0, -75.0],
            [125.0, 25.0],
            [125.0, 125.0],
        ]
    )

    result = periodicity.mirror(p1, levels=1)
    npt.assert_array_equal(result, p1_exp_mirrors)


def test_mirror_2points():
    edges = {k: (0, 100) for k in range(2)}
    periodicity = Periodicity(edges, 2)

    p1 = np.array([[25.0, 25.0]])
    p2 = np.array([[51.0, 51.0]])
    p = np.vstack((p1, p2))

    p1_exp_mirrors = np.array(
        [
            [-75.0, -75.0],
            [-75.0, 25.0],
            [-75.0, 125.0],
            [25.0, -75.0],
            [25.0, 125.0],
            [125.0, -75.0],
            [125.0, 25.0],
            [125.0, 125.0],
        ]
    )

    p2_exp_mirrors = np.array(
        [
            [-49.0, -49.0],
            [-49.0, 51.0],
            [-49.0, 151.0],
            [51.0, -49.0],
            [51.0, 151.0],
            [151.0, -49.0],
            [151.0, 51.0],
            [151.0, 151.0],
        ]
    )

    p_exp_mirrors = np.vstack((p1_exp_mirrors, p2_exp_mirrors))

    p1_result = periodicity.mirror(p1, levels=1)
    npt.assert_array_equal(p1_result, p1_exp_mirrors)

    p2_result = periodicity.mirror(p2, levels=1)
    npt.assert_array_equal(p2_result, p2_exp_mirrors)

    p_result = periodicity.mirror(p, levels=1)
    npt.assert_array_equal(p_result, p_exp_mirrors)


@pytest.mark.parametrize("levels", [1, 2, 3, 4, 5])
def test_mirror_return_indices(levels):
    edges = {k: (0, 100) for k in range(2)}
    periodicity = Periodicity(edges, 2)

    rng = np.random.default_rng(42)
    points = rng.uniform(0, 100, size=(10, 2))
    result = periodicity.mirror(points, levels, return_indices=True)
    assert len(result) == 2
    assert isinstance(result, tuple)

    _, indices = result
    m = periodicity.multiplicity(levels=levels)
    expected_indices = np.repeat(np.arange(len(points)), m)
    npt.assert_array_equal(indices, expected_indices)


def test_wrap_zero_origin():
    edges = {k: (0, 100) for k in range(2)}
    periodicity = Periodicity(edges, 2)

    p = np.array([[130.0, -60.0]])

    p_exp_wrap = np.array([[30.0, 40.0]])

    result = periodicity.wrap(p)
    npt.assert_array_equal(result, p_exp_wrap)


def test_wrap_any_origin():
    edges = {k: (50, 100) for k in range(2)}
    periodicity = Periodicity(edges, 2)

    p = np.array([[130.0, -60.0]])

    p_exp_wrap = np.array([[80.0, 90.0]])

    result = periodicity.wrap(p)
    npt.assert_array_equal(result, p_exp_wrap)


@pytest.mark.parametrize("levels", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_mirror_to_wrap_and_back(levels, dim):
    # Take some points, mirror them
    # then wrap them back and compare with original
    edges = {k: (66, 98) for k in range(dim)}
    periodicity = Periodicity(edges, dim)

    rng = np.random.default_rng(42)
    p = rng.uniform(66, 98, size=(10, dim))

    p_mirrors = periodicity.mirror(p, levels=levels)
    p_mirrors_to_wrap = periodicity.wrap(p_mirrors)

    # each part must be the same point repeated m times
    m = periodicity.multiplicity(levels=levels)
    nparts = len(p_mirrors_to_wrap) / m
    parts = np.split(p_mirrors_to_wrap, nparts, axis=0)

    for pi, part in zip(p, parts):
        expected = np.vstack((pi,) * m)
        npt.assert_array_almost_equal(part, expected, decimal=12)
