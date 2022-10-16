#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


import numpy as np
import pytest

from grispy import Grid, GriSPy, Periodicity

# =========================================================
# INITIALIZATION
# =========================================================


def test_subclass_of_grid():
    assert issubclass(GriSPy, Grid)


def test_instance_of_grid(gsp, gsp_periodic):
    assert isinstance(gsp, Grid)
    assert isinstance(gsp_periodic, Grid)


def test_invalid_periodic_as_string(grispy_init):
    # Axis 0 with invalid type: string instead of dict
    bad_periodic = "{0: [-1, 1], 1: (-1, 1), 2: None}"
    with pytest.raises(TypeError):
        GriSPy(grispy_init["data"], periodic=bad_periodic)


def test_invalid_periodic_with_list(grispy_init):
    # Axis 0 with invalid value type: list instead of tuple
    bad_periodic = {0: [-1, 1], 1: (-1, 1), 2: None}
    with pytest.raises(TypeError):
        GriSPy(grispy_init["data"], periodic=bad_periodic)


def test_invalid_periodic_with_string(grispy_init):
    # Axis is not integer
    bad_periodic = {"A": (-1, 1), 1: (-1, 1), 2: None}
    with pytest.raises(TypeError):
        GriSPy(grispy_init["data"], periodic=bad_periodic)


def test_invalid_periodic_bad_edge_values(grispy_init):
    # Edge 0 is larger than edge 1
    bad_periodic = {0: (1, -1), 1: (-1, 1), 2: None}
    with pytest.raises(ValueError):
        GriSPy(grispy_init["data"], periodic=bad_periodic)


def test_invalid_periodic_bad_edge_type(grispy_init):
    # Edge has wrong type
    bad_periodic = {0: (-1, [1]), 1: (-1, 1), 2: None}
    with pytest.raises(TypeError):
        GriSPy(grispy_init["data"], periodic=bad_periodic)


def test_invalid_metric(grispy_init):
    # Metric name is not a string
    bad_metric = 42
    with pytest.raises(ValueError):
        GriSPy(grispy_init["data"], metric=bad_metric)

    # Metric name is wrong
    bad_metric = "euclidean"
    with pytest.raises(ValueError):
        GriSPy(grispy_init["data"], metric=bad_metric)


def test_valid_periodic_empty(grispy_init):
    # both initializations should be equivalent
    periodic_explicit = {0: None, 1: None, 2: None}
    periodic_implicit = dict()

    exp = GriSPy(grispy_init["data"], periodic=periodic_explicit)
    imp = GriSPy(grispy_init["data"], periodic=periodic_implicit)
    assert exp.periodic == imp.periodic


def test_valid_periodic_instance(grispy_init):
    # both initializations should be equivalent
    periodicity = Periodicity({0: None, 1: None, 2: None}, dim=3)

    grispy = GriSPy(grispy_init["data"], periodic=periodicity)
    assert grispy.periodic is periodicity


# =========================================================
# QUERY METHODS
# =========================================================


def test_grispy_arguments(gsp):
    assert isinstance(gsp.data, np.ndarray)
    assert isinstance(gsp.metric, str)
    assert isinstance(gsp.N_cells, int)
    assert isinstance(gsp.periodic, Periodicity)
    assert isinstance(gsp.copy_data, bool)


def test_grispy_attrs(gsp):
    assert isinstance(gsp.k_bins, np.ndarray)
    assert isinstance(gsp.grid, dict)
    assert isinstance(gsp.dim, int)
    assert isinstance(gsp.isperiodic, bool)


def test_bubble_single_query(gsp, grispy_input):

    b, ind = gsp.bubble_neighbors(
        np.array([[0, 0, 0]]),
        distance_upper_bound=grispy_input["upper_radii"],
    )
    assert isinstance(b, list)
    assert isinstance(ind, list)
    assert len(b) == len(ind)
    assert len(b) == 1
    assert len(ind) == 1


def test_shell_single_query(gsp, grispy_input):

    b, ind = gsp.shell_neighbors(
        np.array([[0, 0, 0]]),
        distance_lower_bound=grispy_input["lower_radii"],
        distance_upper_bound=grispy_input["upper_radii"],
    )
    assert isinstance(b, list)
    assert isinstance(ind, list)
    assert len(b) == len(ind)
    assert len(b) == 1
    assert len(ind) == 1


def test_nearest_neighbors_single_query(gsp, grispy_input):

    b, ind = gsp.nearest_neighbors(
        np.array([[0, 0, 0]]), n=grispy_input["n_nearest"]
    )
    assert isinstance(b, list)
    assert isinstance(ind, list)
    assert len(b) == len(ind)
    assert len(b) == 1
    assert len(ind) == 1
    assert np.shape(b[0]) == (grispy_input["n_nearest"],)
    assert np.shape(ind[0]) == (grispy_input["n_nearest"],)


def test_bubble_multiple_query(gsp, grispy_input):

    b, ind = gsp.bubble_neighbors(
        grispy_input["centres"],
        distance_upper_bound=grispy_input["upper_radii"],
    )
    assert isinstance(b, list)
    assert isinstance(ind, list)
    assert len(b) == len(ind)
    assert len(b) == len(grispy_input["centres"])
    assert len(ind) == len(grispy_input["centres"])


def test_shell_multiple_query(gsp, grispy_input):

    b, ind = gsp.shell_neighbors(
        grispy_input["centres"],
        distance_lower_bound=grispy_input["lower_radii"],
        distance_upper_bound=grispy_input["upper_radii"],
    )
    assert isinstance(b, list)
    assert isinstance(ind, list)
    assert len(b) == len(ind)
    assert len(b) == len(grispy_input["centres"])
    assert len(ind) == len(grispy_input["centres"])


def test_nearest_neighbors_multiple_query(gsp, grispy_input):

    b, ind = gsp.nearest_neighbors(
        grispy_input["centres"], n=grispy_input["n_nearest"]
    )
    assert isinstance(b, list)
    assert isinstance(ind, list)
    assert len(b) == len(ind)
    assert len(b) == len(grispy_input["centres"])
    assert len(ind) == len(grispy_input["centres"])
    for i in range(len(b)):
        assert b[i].shape == (grispy_input["n_nearest"],)
        assert ind[i].shape == (grispy_input["n_nearest"],)


# =========================================================
# PERIODICITY
# =========================================================


def test_mirror_universe(gsp_periodic, grispy_input):
    # Private methods should not be tested, but the idea is to make
    # some mirror method public.. so they can stay for now
    r_cen = np.array([[0, 0, 0]])
    t_cen, t_ind = gsp_periodic._mirror_universe(
        r_cen, distance_upper_bound=[grispy_input["upper_radii"]]
    )
    assert isinstance(t_cen, np.ndarray)
    assert isinstance(t_ind, np.ndarray)
    assert len(t_cen) == len(t_ind)
    assert t_cen.ndim == t_cen.ndim
    assert t_cen.shape[1] == r_cen.shape[1]


def test_near_boundary(gsp_periodic, grispy_input):
    # Private methods should not be tested, but the idea is to make
    # some mirror method public.. so they can stay for now
    mask = gsp_periodic._near_boundary(
        np.array([[0, 0, 0]]),
        distance_upper_bound=[grispy_input["upper_radii"]],
    )
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.ndim == 1


def test_set_periodicity_inplace(gsp):
    periodicity = {0: (-50, 50)}

    assert gsp.isperiodic is False
    assert gsp.periodic.edges == {0: None, 1: None, 2: None}

    result = gsp.set_periodicity(periodicity, inplace=True)

    assert result is None
    assert gsp.isperiodic
    assert gsp.periodic.edges == {0: (-50, 50), 1: None, 2: None}


def test_set_periodicity_no_inplace(gsp):
    periodicity = {0: (-50, 50)}

    assert gsp.isperiodic is False
    assert gsp.periodic.edges == {0: None, 1: None, 2: None}

    result = gsp.set_periodicity(periodicity)

    assert isinstance(result, GriSPy)
    assert result is not gsp
    assert result.isperiodic
    assert result.periodic.edges == {0: (-50, 50), 1: None, 2: None}

    assert gsp.isperiodic is False
    assert gsp.periodic.edges == {0: None, 1: None, 2: None}


# =========================================================
# QUERY INPUT VALIDATION
# =========================================================


def test_invalid_centres_type(gsp_periodic, grispy_input):
    # Invalid type
    bad_centres = [[1, 1, 1], [2, 2, 2]]
    with pytest.raises(TypeError):
        gsp_periodic.bubble_neighbors(
            bad_centres, distance_upper_bound=grispy_input["upper_radii"]
        )


def test_invalid_centres_single_value_type(gsp_periodic, grispy_input):
    rng = np.random.default_rng(987)
    bad_centres = rng.uniform(-1, 1, size=(10, 3))
    bad_centres[4, 1] = np.inf  # add one invalid value
    with pytest.raises(ValueError):
        gsp_periodic.bubble_neighbors(
            bad_centres,
            distance_upper_bound=grispy_input["upper_radii"],
        )


def test_invalid_centres_shape(gsp_periodic, grispy_input):
    rng = np.random.default_rng(987)
    # Invalid shape
    bad_centres = rng.uniform(-1, 1, size=(10, 2))
    with pytest.raises(ValueError):
        gsp_periodic.bubble_neighbors(
            bad_centres,
            distance_upper_bound=grispy_input["upper_radii"],
        )


def test_invalid_centres_empty(gsp_periodic, grispy_input):
    # Invalid shape
    bad_centres = np.array([[], [], []]).reshape((0, 3))
    with pytest.raises(ValueError):
        gsp_periodic.bubble_neighbors(
            bad_centres,
            distance_upper_bound=grispy_input["upper_radii"],
        )


def test_invalid_bounds_type_bubble(gsp_periodic, grispy_input):
    rng = np.random.default_rng(987)
    # Invalid type
    bad_upper_radii = list(rng.uniform(0.6, 1, size=10))
    with pytest.raises(TypeError):
        gsp_periodic.bubble_neighbors(
            grispy_input["centres"],
            distance_upper_bound=bad_upper_radii,
        )


def test_invalid_bounds_value_bubble(gsp_periodic, grispy_input):
    rng = np.random.default_rng(987)
    # Invalid value
    bad_upper_radii = rng.uniform(0.6, 1, size=10)
    bad_upper_radii[5] = -1.0
    with pytest.raises(ValueError):
        gsp_periodic.bubble_neighbors(
            grispy_input["centres"],
            distance_upper_bound=bad_upper_radii,
        )


def test_invalid_bounds_size_bubble(gsp_periodic, grispy_input):
    rng = np.random.default_rng(987)
    # Different lenght than centres
    bad_upper_radii = rng.uniform(0.6, 1, size=11)
    with pytest.raises(ValueError):
        gsp_periodic.bubble_neighbors(
            grispy_input["centres"],
            distance_upper_bound=bad_upper_radii,
        )


def test_invalid_bounds_larger_than_periodic_bubble(
    gsp_periodic, grispy_input
):
    # Invalid value
    bad_upper_radii = 10.0  # larger than periodic range
    with pytest.raises(ValueError):
        gsp_periodic.bubble_neighbors(
            grispy_input["centres"],
            distance_upper_bound=bad_upper_radii,
        )


def test_invalid_bounds_lenghts_shell(gsp_periodic, grispy_input):
    rng = np.random.default_rng(987)
    # Different lenght than centres
    lower_radii = rng.uniform(0.1, 0.5, size=10)
    bad_upper_radii = rng.uniform(0.6, 1, size=11)
    with pytest.raises(ValueError):
        gsp_periodic.shell_neighbors(
            grispy_input["centres"],
            distance_upper_bound=bad_upper_radii,
            distance_lower_bound=lower_radii,
        )


def test_invalid_bounds_values_shell(gsp_periodic, grispy_input):
    rng = np.random.default_rng(354)
    # Upper bound is lower than lower bound
    lower_radii = rng.uniform(0.1, 0.5, size=10)
    bad_upper_radii = rng.uniform(0.6, 1, size=10)
    bad_upper_radii[4] = lower_radii[4] - 0.05
    with pytest.raises(ValueError):
        gsp_periodic.shell_neighbors(
            grispy_input["centres"],
            distance_upper_bound=bad_upper_radii,
            distance_lower_bound=lower_radii,
        )


def test_invalid_sorted_type(gsp_periodic, grispy_input):
    # Invalid type
    bad_sorted = "True"
    with pytest.raises(TypeError):
        gsp_periodic.bubble_neighbors(
            grispy_input["centres"],
            distance_upper_bound=grispy_input["upper_radii"],
            sorted=bad_sorted,
        )


def test_invalid_sortkind_type(gsp_periodic, grispy_input):
    # Invalid type
    bad_kind = ["quicksort"]  # string inside list
    with pytest.raises(TypeError):
        gsp_periodic.bubble_neighbors(
            grispy_input["centres"],
            distance_upper_bound=grispy_input["upper_radii"],
            kind=bad_kind,
        )


def test_invalid_sortkind_value(gsp_periodic, grispy_input):
    # Invalid name
    bad_kind = "quick_sort"
    with pytest.raises(ValueError):
        gsp_periodic.bubble_neighbors(
            grispy_input["centres"],
            distance_upper_bound=grispy_input["upper_radii"],
            kind=bad_kind,
        )


def test_invalid_nnearest_type(gsp_periodic, grispy_input):

    # Invalid type
    bad_n = np.array([2])  # array instead of integer
    with pytest.raises(TypeError):
        gsp_periodic.nearest_neighbors(grispy_input["centres"], n=bad_n)


def test_invalid_nnearest_value(gsp_periodic, grispy_input):
    # Invalid value
    bad_n = -5
    with pytest.raises(ValueError):
        gsp_periodic.nearest_neighbors(grispy_input["centres"], n=bad_n)
