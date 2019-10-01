from numpy.testing import assert_equal, assert_
import numpy as np
from grispy import GriSPy
import pytest


class Test_grispy:
    @pytest.fixture
    def setUp_1d(self):

        np.random.seed(1234)
        npoints = 10 ** 5
        self.lbox = 100.0
        self.centres = self.lbox * (0.5 - np.random.rand(1, 10).T)
        self.data = np.random.uniform(
            -0.5 * self.lbox, 0.5 * self.lbox, size=(npoints, 1)
        )
        self.upper_radii = 0.25 * self.lbox
        self.lower_radii = 0.20 * self.lbox
        self.n_nearest = 32
        self.eps = 1e-6

        self.gsp = GriSPy(self.data)

    def test_nearest_neighbors_sort(self, setUp_1d):

        b, ind = self.gsp.nearest_neighbors(self.centres, n=self.n_nearest)
        for i in range(len(b)):
            assert_equal(sorted(b[i]), b[i])

        self.gsp.set_periodicity({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = self.gsp.nearest_neighbors(self.centres, n=self.n_nearest)
        for i in range(len(b)):
            assert_equal(sorted(b[i]), b[i])

    def test_all_in_bubble(self, setUp_1d):

        b, ind = self.gsp.bubble_neighbors(
            self.centres, distance_upper_bound=self.upper_radii
        )

        for i, l in enumerate(ind):
            for j in l:
                d = np.fabs(self.data[j] - self.centres[i])

                assert_(d <= self.upper_radii * (1.0 + self.eps))

        self.gsp.set_periodicity({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = self.gsp.bubble_neighbors(
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

                assert_(d <= self.upper_radii * (1.0 + self.eps))

    def test_all_in_shell(self, setUp_1d):

        b, ind = self.gsp.shell_neighbors(
            self.centres,
            distance_lower_bound=self.lower_radii,
            distance_upper_bound=self.upper_radii,
        )

        for i, l in enumerate(ind):
            for j in l:
                d = np.fabs(self.data[j] - self.centres[i])

                assert_(d <= self.upper_radii * (1.0 + self.eps))
                assert_(d >= self.lower_radii * (1.0 - self.eps))

        self.gsp.set_periodicity({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = self.gsp.shell_neighbors(
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

                assert_(d <= self.upper_radii * (1.0 + self.eps))
                assert_(d >= self.lower_radii * (1.0 - self.eps))

    def test_bubble_precision(self, setUp_1d):

        b, ind = self.gsp.bubble_neighbors(
            self.centres,
            distance_upper_bound=self.upper_radii,
            sorted=True
        )

        for i, centre in enumerate(self.centres):
            d = np.fabs(self.data - centre)

            mask = d < self.upper_radii
            d = d[mask]
            assert_equal(len(b[i]), len(d))
            np.testing.assert_almost_equal(b[i], sorted(d), decimal=14)

        self.gsp.set_periodicity({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = self.gsp.bubble_neighbors(
            self.centres,
            distance_upper_bound=self.upper_radii,
            sorted=True
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
            assert_equal(len(b[i]), len(d))
            np.testing.assert_almost_equal(b[i], sorted(d), decimal=14)

    def test_shell_precision(self, setUp_1d):

        b, ind = self.gsp.shell_neighbors(
            self.centres,
            distance_lower_bound=self.lower_radii,
            distance_upper_bound=self.upper_radii,
            sorted=True
        )

        for i, centre in enumerate(self.centres):
            d = np.fabs(self.data - centre)

            mask = (d <= self.upper_radii) * (d >= self.lower_radii)
            d = d[mask]
            assert_equal(len(b[i]), len(d))
            np.testing.assert_almost_equal(b[i], sorted(d), decimal=14)

        self.gsp.set_periodicity({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = self.gsp.shell_neighbors(
            self.centres,
            distance_lower_bound=self.lower_radii,
            distance_upper_bound=self.upper_radii,
            sorted=True
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
            assert_equal(len(b[i]), len(d))
            np.testing.assert_almost_equal(b[i], sorted(d), decimal=14)

    def test_nearest_neighbors_precision(self, setUp_1d):

        b, ind = self.gsp.nearest_neighbors(self.centres, n=self.n_nearest)

        for i, centre in enumerate(self.centres):
            d = np.fabs(self.data - centre)

            d = sorted(np.concatenate(d))
            d = d[: self.n_nearest]
            assert_equal(len(b[i]), len(d))
            np.testing.assert_almost_equal(b[i], d, decimal=16)

        self.gsp.set_periodicity({0: (-self.lbox * 0.5, self.lbox * 0.5)})
        b, ind = self.gsp.nearest_neighbors(self.centres, n=self.n_nearest)

        for i, centre in enumerate(self.centres):
            d = self.data - centre
            mask = d > 0.5 * self.lbox
            d[mask] = d[mask] - self.lbox
            mask = d < -0.5 * self.lbox
            d[mask] = d[mask] + self.lbox
            d = np.fabs(d)

            d = sorted(np.concatenate(d))
            d = d[: self.n_nearest]
            assert_equal(len(b[i]), len(d))
            np.testing.assert_almost_equal(b[i], d, decimal=16)


class Test_periodicity_grispy:
    @pytest.fixture
    def setUp_1d(self):

        self.lbox = 10.0
        self.data = np.array(
            [
                [2, 0, 0],
                [-2, 0, 0],
                [0, 2, 0],
                [0, -2, 0],
                [0, 0, 2],
                [0, 0, -2]
            ]
        )

        self.eps = 1e-6
        self.gsp = GriSPy(self.data)

    def test_periodicity_in_shell(self, setUp_1d):

        centres = np.array([[0.0, 0.0, 0.0]])
        upper_radii = 0.8 * self.lbox
        lower_radii = 0.8 * self.lbox

        for j in range(3):
            self.gsp.set_periodicity({j: (-0.5 * self.lbox, 0.5 * self.lbox)})

            dis, ind = self.gsp.shell_neighbors(
                centres,
                distance_lower_bound=lower_radii - self.eps,
                distance_upper_bound=upper_radii + self.eps,
            )
            dis, ind = dis[0], ind[0]

            aux = np.argsort(ind)
            ind = ind[aux]
            dis = dis[aux]
            for i in range(2):
                assert_equal(ind[i], i + (j * 2))
                assert_(dis[i] >= lower_radii - self.eps)
                assert_(dis[i] <= upper_radii + self.eps)

        self.gsp.set_periodicity(
            {
                0: (-self.lbox * 0.5, self.lbox * 0.5),
                1: (-self.lbox * 0.5, self.lbox * 0.5),
                2: (-self.lbox * 0.5, self.lbox * 0.5),
            }
        )
        dis, ind = self.gsp.shell_neighbors(
            centres,
            distance_lower_bound=lower_radii - self.eps,
            distance_upper_bound=upper_radii + self.eps
        )
        dis, ind = dis[0], ind[0]

        aux = np.argsort(ind)
        ind = ind[aux]
        dis = dis[aux]

        for i in range(6):
            assert_equal(ind[i], i)
            assert_(dis[i] >= lower_radii * (1.0 - self.eps))
            assert_(dis[i] <= upper_radii * (1.0 + self.eps))

    def test_periodicity_in_bubble(self, setUp_1d):

        centres = np.array(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0]
            ]
        )
        upper_radii = 0.3 * self.lbox

        for j in range(3):
            self.gsp.set_periodicity({j: (-self.lbox * 0.5, self.lbox * 0.5)})

            centre = centres[j].reshape(1, 3)

            dis, ind = self.gsp.bubble_neighbors(
                centre, distance_upper_bound=upper_radii
            )
            dis, ind = dis[0], ind[0]

            aux = np.argsort(ind)
            ind = ind[aux]
            dis = dis[aux]

            for i in range(2):
                assert_equal(ind[i], i + (j * 2))
                assert_(dis[i] <= upper_radii * (1.0 + self.eps))
                assert_(dis[i] >= upper_radii * (1.0 - self.eps))
