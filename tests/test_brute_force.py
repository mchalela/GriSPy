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

from grispy import GriSPy

# =========================================================================
# Test GriSPy class
# =========================================================================


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
@pytest.mark.parametrize("isperiodic", [False, True])
def test_self_search_bubble(make_grispy, dim, N_cells, isperiodic):
    """Test using the same indexed data as centres."""
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=isperiodic)
    centres = np.copy(gsp.data)

    dist, index = gsp.bubble_neighbors(centres, distance_upper_bound=1e-15)
    for idx in range(len(centres)):
        assert len(dist[idx]) == 1
        assert len(index[idx]) == 1


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
@pytest.mark.parametrize("isperiodic", [False, True])
def test_self_search_shell(make_grispy, dim, N_cells, isperiodic):
    """Test using the same indexed data as centres with the shell.

    The idea is to test the distance comparisson symbols.
    Points that are within lower_bound <= distance < upper_bound
    """
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=isperiodic)
    centres = np.copy(gsp.data)

    dist, index = gsp.shell_neighbors(
        centres, distance_lower_bound=0.0, distance_upper_bound=1e-15
    )
    for idx in range(len(centres)):
        assert len(dist[idx]) == 1
        assert len(index[idx]) == 1
        assert index[idx] == idx


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
@pytest.mark.parametrize("isperiodic", [False, True])
def test_self_search_nearest(make_grispy, dim, N_cells, isperiodic):
    """Test using the same indexed data as centres with the shell.

    The idea is to test the distance comparisson symbols.
    Points that are within lower_bound <= distance < upper_bound
    """
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=isperiodic)
    centres = np.copy(gsp.data)

    dist, index = gsp.nearest_neighbors(centres, n=1)
    for idx in range(len(centres)):
        assert len(dist[idx]) == 1
        assert len(index[idx]) == 1
        assert index[idx] == idx


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
@pytest.mark.parametrize("isperiodic", [False, True])
def test_nearest_neighbors_sort(make_grispy, dim, N_cells, isperiodic):
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=isperiodic)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(10, dim))
    n_nearest = 32

    b, _ = gsp.nearest_neighbors(centres, n=n_nearest)
    for i in range(len(b)):
        npt.assert_equal(sorted(b[i]), b[i])


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
@pytest.mark.parametrize("isperiodic", [False, True])
def test_nearest_neighbors_same_result_twice(
    make_grispy, dim, N_cells, isperiodic
):
    # Run it twice and make sure it works ok.
    # when executing twice the tutorial cell of this method
    # the second time never ended
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=isperiodic)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(10, dim))

    b, ind = gsp.nearest_neighbors(centres, n=10)
    b2, ind2 = gsp.nearest_neighbors(centres, n=10)

    npt.assert_almost_equal(b, b2, 14)
    npt.assert_equal(ind, ind2)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
def test_all_in_bubble(make_grispy, dim, N_cells):
    gsp = make_grispy(dim=dim, N_cells=N_cells)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(10, dim))
    radii = 10.0

    _, ind = gsp.bubble_neighbors(centres, distance_upper_bound=radii)

    for i, l in enumerate(ind):
        for j in l:
            d = gsp.data[j] - centres[i]
            d = np.linalg.norm(d)
            assert (d <= radii).all()


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
def test_all_in_bubble_with_periodicity(make_grispy, dim, N_cells):
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=True)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(2, dim))
    radii = 10.0
    lbox = 100.0

    dists, ind = gsp.bubble_neighbors(centres, distance_upper_bound=radii)
    for i, l in enumerate(ind):
        for j in l:

            d = gsp.data[j] - centres[i]
            for k in range(dim):
                if d[k] > 0.5 * lbox:
                    d[k] -= lbox
                elif d[k] < -0.5 * lbox:
                    d[k] += lbox
            d = np.linalg.norm(d)
            assert (d <= radii).all()


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
def test_all_in_shell(make_grispy, dim, N_cells):
    gsp = make_grispy(dim=dim, N_cells=N_cells)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(10, dim))
    upper_radii = 50.0
    lower_radii = 10.0

    b, ind = gsp.shell_neighbors(centres, lower_radii, upper_radii)

    for i, l in enumerate(ind):
        for j in l:
            d = gsp.data[j] - centres[i]
            d = np.linalg.norm(d)

            assert (d <= upper_radii).all()
            assert (d >= lower_radii).all()


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
def test_all_in_shell_with_periodicity(make_grispy, dim, N_cells):
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=True)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(10, dim))
    upper_radii = 50.0
    lower_radii = 10.0
    lbox = 100.0

    b, ind = gsp.shell_neighbors(centres, lower_radii, upper_radii)

    for i, l in enumerate(ind):
        for j in l:

            d = gsp.data[j] - centres[i]
            for k in range(dim):
                if d[k] > 0.5 * lbox:
                    d[k] -= lbox
                elif d[k] < -0.5 * lbox:
                    d[k] += lbox

            d = np.linalg.norm(d)
            assert (d <= upper_radii).all()
            assert (d >= lower_radii).all()


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
def test_bubble_precision(make_grispy, dim, N_cells):
    gsp = make_grispy(dim=dim, N_cells=N_cells)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(1, dim))
    upper_radii = 50.0

    b, _ = gsp.bubble_neighbors(centres, upper_radii, sorted=True)

    for i, centre in enumerate(centres):
        d = gsp.data - centre
        d = np.linalg.norm(d, axis=1)
        mask = d <= upper_radii
        d = d[mask]

        assert len(b[i]) == len(d)
        npt.assert_almost_equal(b[i], sorted(d), decimal=14)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
def test_bubble_precision_with_periodicity(make_grispy, dim, N_cells):
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=True)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(1, dim))
    upper_radii = 50.0
    lbox = 100.0

    b, _ = gsp.bubble_neighbors(centres, upper_radii, sorted=True)

    for i, centre in enumerate(centres):
        d = gsp.data - centre

        mask = d > 0.5 * lbox
        d[mask] -= lbox
        mask = d < -0.5 * lbox
        d[mask] += lbox

        d = np.linalg.norm(d, axis=1)
        mask = d <= upper_radii
        d = d[mask]

        assert len(b[i]) == len(d)
        npt.assert_almost_equal(b[i], sorted(d), decimal=14)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
def test_shell_precision(make_grispy, dim, N_cells):
    gsp = make_grispy(dim=dim, N_cells=N_cells)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(10, dim))
    upper_radii = 50.0
    lower_radii = 10.0

    b, _ = gsp.shell_neighbors(centres, lower_radii, upper_radii, sorted=True)

    for i, centre in enumerate(centres):
        d = gsp.data - centre
        d = np.linalg.norm(d, axis=1)
        mask = (d <= upper_radii) * (d >= lower_radii)
        d = d[mask]

        assert len(b[i]) == len(d)
        npt.assert_almost_equal(b[i], sorted(d), decimal=14)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
def test_shell_precision_with_periodicity(make_grispy, dim, N_cells):
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=True)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(10, dim))
    upper_radii = 50.0
    lower_radii = 10.0
    lbox = 100.0

    b, _ = gsp.shell_neighbors(centres, lower_radii, upper_radii, sorted=True)

    for i, centre in enumerate(centres):
        d = gsp.data - centre

        mask = d > 0.5 * lbox
        d[mask] -= lbox
        mask = d < -0.5 * lbox
        d[mask] += lbox

        d = np.linalg.norm(d, axis=1)
        mask = (d <= upper_radii) * (d >= lower_radii)
        d = d[mask]
        assert len(b[i]) == len(d)
        npt.assert_almost_equal(b[i], sorted(d), decimal=14)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
@pytest.mark.parametrize("isperiodic", [False, True])
def test_nearest_neighbors_precision(make_grispy, dim, N_cells, isperiodic):
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=isperiodic)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(10, dim))
    n_nearest = 8
    lbox = 100.0

    b, _ = gsp.nearest_neighbors(centres, n=n_nearest)
    for i, centre in enumerate(centres):
        d = gsp.data - centre

        if isperiodic:
            mask = d > 0.5 * lbox
            d[mask] -= lbox
            mask = d < -0.5 * lbox
            d[mask] += lbox

        d = np.linalg.norm(d, axis=1)
        d = sorted(d)
        d = d[:n_nearest]
        assert len(b[i]) == len(d)
        npt.assert_almost_equal(b[i], d, decimal=14)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
def test_nearest_neighbors_precision_with_periodicity(
    make_grispy, dim, N_cells
):
    gsp = make_grispy(dim=dim, N_cells=N_cells, isperiodic=True)

    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 100, size=(10, dim))
    n_nearest = 8
    lbox = 100.0

    b, _ = gsp.nearest_neighbors(centres, n=n_nearest)

    for i, centre in enumerate(centres):
        d = gsp.data - centre

        mask = d > 0.5 * lbox
        d[mask] -= lbox
        mask = d < -0.5 * lbox
        d[mask] += lbox

        d = np.linalg.norm(d, axis=1)
        d = sorted(d)
        d = d[:n_nearest]
        assert len(b[i]) == len(d)
        npt.assert_almost_equal(b[i], d, decimal=14)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N_cells", [1, 2, 8, 16])
@pytest.mark.parametrize("floatX", [np.float32, np.float64])
def test_floatX_precision(dim, N_cells, floatX):

    rng = np.random.default_rng(1234)
    data_floatX = rng.random(size=(100, dim), dtype=floatX)
    centres_floatX = rng.random(size=(10, dim), dtype=floatX)
    upper_radii = 0.2

    gsp_floatX = GriSPy(data_floatX, N_cells=N_cells)
    dist_floatX, ind_floatX = gsp_floatX.bubble_neighbors(
        centres_floatX, distance_upper_bound=upper_radii
    )

    eps = np.finfo(floatX).resolution
    decimal = np.abs(int(np.log10(eps)))

    for i, ind_list in enumerate(ind_floatX):
        for j, il in enumerate(ind_list):
            gsp_dist = dist_floatX[i][j]

            delta = data_floatX[il] - centres_floatX[i]
            dist = np.linalg.norm(delta)

            assert (dist <= upper_radii).all()
            npt.assert_almost_equal(dist, gsp_dist, decimal)


class Test_periodicity_grispy:
    """Test the periodicity condition"""

    def setup_method(self, *args):
        self.lbox = 10.0
        self.data = np.array(
            [
                [2, 0, 0],
                [-2, 0, 0],
                [0, 2, 0],
                [0, -2, 0],
                [0, 0, 2],
                [0, 0, -2],
            ]
        )
        self.eps = 1e-6

    def make_gsp(self, periodic):
        return GriSPy(self.data, periodic=periodic)

    def test_periodicity_in_shell(self):

        centres = np.array([[0.0, 0.0, 0.0]])
        upper_radii = 0.8 * self.lbox
        lower_radii = 0.8 * self.lbox

        for j in range(3):
            gsp = self.make_gsp({j: (-0.5 * self.lbox, 0.5 * self.lbox)})

            dis, ind = gsp.shell_neighbors(
                centres,
                distance_lower_bound=lower_radii - self.eps,
                distance_upper_bound=upper_radii + self.eps,
            )
            dis, ind = dis[0], ind[0]

            aux = np.argsort(ind)
            ind = ind[aux]
            dis = dis[aux]
            for i in range(2):
                npt.assert_equal(ind[i], i + (j * 2))
                npt.assert_(dis[i] >= lower_radii - self.eps)
                npt.assert_(dis[i] <= upper_radii + self.eps)

        gsp = self.make_gsp(
            {
                0: (-self.lbox * 0.5, self.lbox * 0.5),
                1: (-self.lbox * 0.5, self.lbox * 0.5),
                2: (-self.lbox * 0.5, self.lbox * 0.5),
            }
        )

        dis, ind = gsp.shell_neighbors(
            centres,
            distance_lower_bound=lower_radii - self.eps,
            distance_upper_bound=upper_radii + self.eps,
        )
        dis, ind = dis[0], ind[0]

        aux = np.argsort(ind)
        ind = ind[aux]
        dis = dis[aux]

        for i in range(6):
            npt.assert_equal(ind[i], i)
            npt.assert_(dis[i] >= lower_radii * (1.0 - self.eps))
            npt.assert_(dis[i] <= upper_radii * (1.0 + self.eps))

    def test_periodicity_in_bubble(self):

        centres = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        upper_radii = 0.3 * self.lbox

        for j in range(3):
            gsp = self.make_gsp({j: (-self.lbox * 0.5, self.lbox * 0.5)})

            centre = centres[j].reshape(1, 3)

            dis, ind = gsp.bubble_neighbors(
                centre, distance_upper_bound=upper_radii
            )
            dis, ind = dis[0], ind[0]

            aux = np.argsort(ind)
            ind = ind[aux]
            dis = dis[aux]

            for i in range(2):
                npt.assert_equal(ind[i], i + (j * 2))
                npt.assert_(dis[i] <= upper_radii * (1.0 + self.eps))
                npt.assert_(dis[i] >= upper_radii * (1.0 - self.eps))


class Test_hypersphere_grispy:
    """Test a 4 dimensional space"""

    @pytest.fixture
    def valid_input(self):
        d = dict()
        d["lbox"] = 100.0
        d["upper_radii"] = 0.25 * d["lbox"]
        d["lower_radii"] = 0.20 * d["lbox"]
        d["n_nearest"] = 32
        d["eps"] = 1e-6
        return d

    @pytest.fixture
    def gsp(self, valid_input):
        ############################################
        #
        # Follow
        # http://mathworld.wolfram.com/HyperspherePointPicking.html
        # Marsaglia, G.
        # "Choosing a Point from the Surface of a Sphere."
        # Ann. Math. Stat. 43, 645-646, 1972.
        #
        ############################################
        rng = np.random.default_rng(1234)

        npoints = 10 ** 5
        x = rng.uniform(-1.0, 1.0, size=(npoints, 1))
        y = rng.uniform(-1.0, 1.0, size=(npoints, 1))
        z = rng.uniform(-1.0, 1.0, size=(npoints, 1))
        w = rng.uniform(-1.0, 1.0, size=(npoints, 1))

        tttt = (x ** 2 + y ** 2 < 1.0) * (z ** 2 + w ** 2 < 1.0)
        npoints = np.sum(tttt)
        self.radius = valid_input["lbox"] * rng.random(npoints)
        x = x[tttt]
        y = y[tttt]
        z = z[tttt]
        w = w[tttt]

        tttt = np.sqrt((1.0 - x ** 2 - y ** 2) / (z ** 2 + w ** 2))
        x = self.radius * x
        y = self.radius * y
        z = self.radius * z * tttt
        w = self.radius * w * tttt

        tttt = np.sqrt(x ** 2 + y ** 2 + z ** 2 + w ** 2)
        npt.assert_almost_equal(self.radius, tttt, decimal=12)
        data = np.array([x, y, z, w]).T

        return GriSPy(data)

    def test_in_hiperbubble(self, gsp, valid_input):

        centre = np.array([[0.0, 0.0, 0.0, 0.0]])

        dis, ind = gsp.bubble_neighbors(
            centre,
            distance_upper_bound=valid_input["upper_radii"],
            sorted=True,
        )
        dis, ind = dis[0], ind[0]

        mask = self.radius <= valid_input["upper_radii"] * (
            1.0 + valid_input["eps"]
        )
        npt.assert_equal(len(dis), len(ind))
        npt.assert_equal(len(dis), len(self.radius[mask]))
        npt.assert_almost_equal(dis, sorted(self.radius[mask]), decimal=14)

    def test_in_hipersheell(self, gsp, valid_input):

        centre = np.array([[0.0, 0.0, 0.0, 0.0]])

        dis, ind = gsp.shell_neighbors(
            centre,
            distance_lower_bound=valid_input["lower_radii"],
            distance_upper_bound=valid_input["upper_radii"],
            sorted=True,
        )
        dis, ind = dis[0], ind[0]

        mask = (
            self.radius
            <= valid_input["upper_radii"] * (1.0 + valid_input["eps"])
        ) * (
            self.radius
            >= valid_input["lower_radii"] * (1.0 - valid_input["eps"])
        )
        npt.assert_equal(len(dis), len(ind))
        npt.assert_equal(len(dis), len(self.radius[mask]))
        npt.assert_almost_equal(dis, sorted(self.radius[mask]), decimal=14)

    def test_hipernearest_neighbors(self, gsp, valid_input):

        centre = np.array([[0.0, 0.0, 0.0, 0.0]])

        dis, ind = gsp.nearest_neighbors(centre, n=valid_input["n_nearest"])
        dis, ind = dis[0], ind[0]

        tmp = sorted(self.radius)[: valid_input["n_nearest"]]
        npt.assert_equal(len(dis), len(ind))
        npt.assert_equal(len(dis), len(tmp))
        npt.assert_almost_equal(dis, tmp, decimal=14)
