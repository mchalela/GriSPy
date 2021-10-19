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

# def test_grid_properties_edges(self, grid):
#     dmin = grid.data.min(axis=0) - grid.epsilon
#     dmax = grid.data.max(axis=0) + grid.epsilon
#     k_bins = np.linspace(dmin, dmax, grid.N_cells + 1)
#     expected = k_bins[[0, -1], :]
#     npt.assert_almost_equal(grid.edges, expected, 14)


class Test_auto:
    def setup_method(self, *args):
        self.random = np.random.RandomState(8)
        self.lbox = 100.0
        self.data = self.random.uniform(0, self.lbox, size=(2, 3))
        self.centres = np.copy(self.data)

    def make_gsp(self, periodic):
        return GriSPy(self.data, periodic=periodic)

    def test_auto(self):
        gsp = self.make_gsp({})
        dist, index = gsp.bubble_neighbors(
            self.centres, distance_upper_bound=1.5
        )
        for idx in range(len(self.centres)):
            npt.assert_(len(dist[idx]) == 1)
            npt.assert_(len(index[idx]) == 1)

    def test_auto_peridic(self):

        gsp = self.make_gsp(
            {0: (0, self.lbox), 1: (0, self.lbox), 2: (0, self.lbox)}
        )
        dist, index = gsp.bubble_neighbors(
            self.centres, distance_upper_bound=1.5
        )
        for idx in range(len(self.centres)):
            npt.assert_(len(dist[idx]) == 1)
            npt.assert_(len(index[idx]) == 1)


class Test_grispy:
    def setup_method(self, *args):
        self.random = np.random.RandomState(1234)
        self.lbox = 100.0
        self.centres = self.lbox * (0.5 - self.random.rand(1, 10).T)
        self.data = self.random.uniform(
            -0.5 * self.lbox, 0.5 * self.lbox, size=(10 ** 3, 1)
        )
        self.upper_radii = 0.25 * self.lbox
        self.lower_radii = 0.20 * self.lbox
        self.n_nearest = 32
        self.eps = np.finfo(np.float64).resolution

    def make_gsp(self, periodic):
        return GriSPy(self.data, periodic=periodic)

    def test_nearest_neighbors_sort(self):
        gsp = self.make_gsp({})
        b, ind = gsp.nearest_neighbors(self.centres, n=self.n_nearest)
        for i in range(len(b)):
            npt.assert_equal(sorted(b[i]), b[i])

        gsp = self.make_gsp({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = gsp.nearest_neighbors(self.centres, n=self.n_nearest)
        for i in range(len(b)):
            npt.assert_equal(sorted(b[i]), b[i])

    def test_all_in_bubble(self):
        gsp = self.make_gsp({})
        b, ind = gsp.bubble_neighbors(
            self.centres, distance_upper_bound=self.upper_radii
        )

        for i, l in enumerate(ind):
            for j in l:
                d = np.fabs(self.data[j] - self.centres[i])
                npt.assert_(d <= self.upper_radii + self.eps)

        gsp = self.make_gsp({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = gsp.bubble_neighbors(
            self.centres, distance_upper_bound=self.upper_radii
        )

        for i, l in enumerate(ind):
            for j in l:
                d = self.data[j] - self.centres[i]
                if d > 0.5 * self.lbox:
                    d = d - self.lbox
                if d < -0.5 * self.lbox:
                    d = d + self.lbox
                d = np.fabs(d)
                npt.assert_(d <= self.upper_radii + self.eps)

    def test_all_in_shell(self):
        gsp = self.make_gsp({})
        b, ind = gsp.shell_neighbors(
            self.centres,
            distance_lower_bound=self.lower_radii,
            distance_upper_bound=self.upper_radii,
        )

        for i, l in enumerate(ind):
            for j in l:
                d = np.fabs(self.data[j] - self.centres[i])
                npt.assert_(d <= self.upper_radii + self.eps)
                npt.assert_(d >= self.lower_radii - self.eps)

        gsp = self.make_gsp({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = gsp.shell_neighbors(
            self.centres,
            distance_lower_bound=self.lower_radii,
            distance_upper_bound=self.upper_radii,
        )

        for i, l in enumerate(ind):
            for j in l:
                d = self.data[j] - self.centres[i]
                if d > 0.5 * self.lbox:
                    d = d - self.lbox
                if d < -0.5 * self.lbox:
                    d = d + self.lbox
                d = np.fabs(d)
                npt.assert_(d <= self.upper_radii + self.eps)
                npt.assert_(d >= self.lower_radii - self.eps)

    def test_bubble_precision(self):
        gsp = self.make_gsp({})
        b, ind = gsp.bubble_neighbors(
            self.centres, distance_upper_bound=self.upper_radii, sorted=True
        )

        for i, centre in enumerate(self.centres):
            d = np.fabs(self.data - centre)
            mask = d < self.upper_radii
            d = d[mask]
            npt.assert_equal(len(b[i]), len(d))
            npt.assert_almost_equal(b[i], sorted(d), decimal=14)

        gsp = self.make_gsp({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = gsp.bubble_neighbors(
            self.centres, distance_upper_bound=self.upper_radii, sorted=True
        )

        for i, centre in enumerate(self.centres):
            d = self.data - centre
            mask = d > 0.5 * self.lbox
            d[mask] = d[mask] - self.lbox
            mask = d < -0.5 * self.lbox
            d[mask] = d[mask] + self.lbox
            d = np.fabs(d)

            mask = d < self.upper_radii
            d = d[mask]
            npt.assert_equal(len(b[i]), len(d))
            npt.assert_almost_equal(b[i], sorted(d), decimal=14)

    def test_shell_precision(self):
        gsp = self.make_gsp({})
        b, ind = gsp.shell_neighbors(
            self.centres,
            distance_lower_bound=self.lower_radii,
            distance_upper_bound=self.upper_radii,
            sorted=True,
        )

        for i, centre in enumerate(self.centres):
            d = np.fabs(self.data - centre)
            mask = (d <= self.upper_radii) * (d >= self.lower_radii)
            d = d[mask]
            npt.assert_equal(len(b[i]), len(d))
            npt.assert_almost_equal(b[i], sorted(d), decimal=14)

        gsp = self.make_gsp({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = gsp.shell_neighbors(
            self.centres,
            distance_lower_bound=self.lower_radii,
            distance_upper_bound=self.upper_radii,
            sorted=True,
        )

        for i, centre in enumerate(self.centres):
            d = self.data - centre
            mask = d > 0.5 * self.lbox
            d[mask] = d[mask] - self.lbox
            mask = d < -0.5 * self.lbox
            d[mask] = d[mask] + self.lbox
            d = np.fabs(d)

            mask = (d <= self.upper_radii) * (d >= self.lower_radii)
            d = d[mask]
            npt.assert_equal(len(b[i]), len(d))
            npt.assert_almost_equal(b[i], sorted(d), decimal=14)

    def test_nearest_neighbors_precision(self):
        gsp = self.make_gsp({})
        b, ind = gsp.nearest_neighbors(self.centres, n=self.n_nearest)
        for i, centre in enumerate(self.centres):
            d = np.fabs(self.data - centre)
            d = sorted(np.concatenate(d))
            d = d[: self.n_nearest]
            npt.assert_equal(len(b[i]), len(d))
            npt.assert_almost_equal(b[i], d, decimal=16)

        gsp = self.make_gsp({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = gsp.nearest_neighbors(self.centres, n=self.n_nearest)
        for i, centre in enumerate(self.centres):
            d = self.data - centre
            mask = d > 0.5 * self.lbox
            d[mask] = d[mask] - self.lbox
            mask = d < -0.5 * self.lbox
            d[mask] = d[mask] + self.lbox
            d = np.fabs(d)
            d = sorted(np.concatenate(d))
            d = d[: self.n_nearest]
            npt.assert_equal(len(b[i]), len(d))
            npt.assert_almost_equal(b[i], d, decimal=16)

    @pytest.mark.parametrize("floatX", [np.float32, np.float64])
    def test_floatX_precision(self, floatX):

        rng = np.random.default_rng(1234)
        data_floatX = rng.random(size=(1000, 3), dtype=floatX)
        centres_floatX = rng.random(size=(100, 3), dtype=floatX)
        upper_radii = 0.2

        gsp_floatX = GriSPy(data_floatX)
        dist_floatX, ind_floatX = gsp_floatX.bubble_neighbors(
            centres_floatX, distance_upper_bound=upper_radii
        )

        eps = np.finfo(floatX).resolution

        for i, ind_list in enumerate(ind_floatX):
            for j, il in enumerate(ind_list):
                delta = data_floatX[il] - centres_floatX[i]
                dist = np.sqrt(np.sum(delta ** 2))
                npt.assert_(dist <= upper_radii + eps)
                gsp_dist = dist_floatX[i][j]
                npt.assert_(abs(dist - gsp_dist) <= eps)


class Test_periodicity_grispy:
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
    @pytest.fixture
    def gsp(self):

        np.random.seed(1234)
        self.lbox = 100.0
        self.upper_radii = 0.25 * self.lbox
        self.lower_radii = 0.20 * self.lbox
        self.n_nearest = 32
        self.eps = 1e-6

        ############################################
        #
        # Follow
        # http://mathworld.wolfram.com/HyperspherePointPicking.html
        # Marsaglia, G.
        # "Choosing a Point from the Surface of a Sphere."
        # Ann. Math. Stat. 43, 645-646, 1972.
        #
        ############################################

        npoints = 10 ** 5
        x = np.random.uniform(-1.0, 1.0, size=(npoints, 1))
        y = np.random.uniform(-1.0, 1.0, size=(npoints, 1))
        z = np.random.uniform(-1.0, 1.0, size=(npoints, 1))
        w = np.random.uniform(-1.0, 1.0, size=(npoints, 1))

        tttt = (x ** 2 + y ** 2 < 1.0) * (z ** 2 + w ** 2 < 1.0)
        npoints = np.sum(tttt)
        self.radius = self.lbox * np.random.rand(npoints)
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
        self.data = np.array([x, y, z, w]).T

        ############################################
        ############################################

        return GriSPy(self.data)

    def test_in_hiperbubble(self, gsp):

        centre = np.array([[0.0, 0.0, 0.0, 0.0]])

        dis, ind = gsp.bubble_neighbors(
            centre, distance_upper_bound=self.upper_radii, sorted=True
        )
        dis, ind = dis[0], ind[0]

        mask = self.radius <= self.upper_radii * (1.0 + self.eps)
        npt.assert_equal(len(dis), len(ind))
        npt.assert_equal(len(dis), len(self.radius[mask]))
        npt.assert_almost_equal(dis, sorted(self.radius[mask]), decimal=14)

    def test_in_hipersheell(self, gsp):

        centre = np.array([[0.0, 0.0, 0.0, 0.0]])

        dis, ind = gsp.shell_neighbors(
            centre,
            distance_lower_bound=self.lower_radii,
            distance_upper_bound=self.upper_radii,
            sorted=True,
        )
        dis, ind = dis[0], ind[0]

        mask = (self.radius <= self.upper_radii * (1.0 + self.eps)) * (
            self.radius >= self.lower_radii * (1.0 - self.eps)
        )
        npt.assert_equal(len(dis), len(ind))
        npt.assert_equal(len(dis), len(self.radius[mask]))
        npt.assert_almost_equal(dis, sorted(self.radius[mask]), decimal=14)

    def test_hipernearest_neighbors(self, gsp):

        centre = np.array([[0.0, 0.0, 0.0, 0.0]])

        dis, ind = gsp.nearest_neighbors(centre, n=self.n_nearest)
        dis, ind = dis[0], ind[0]

        tmp = sorted(self.radius)[: self.n_nearest]
        npt.assert_equal(len(dis), len(ind))
        npt.assert_equal(len(dis), len(tmp))
        npt.assert_almost_equal(dis, tmp, decimal=14)
