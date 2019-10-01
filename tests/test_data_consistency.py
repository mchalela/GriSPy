from numpy.testing import assert_equal, assert_
import numpy as np
from grispy import GriSPy
import pytest

class Test_data_consistency:

    @pytest.fixture
    def setUp(self):

        np.random.seed(1234)
        self.centres = np.random.rand(3, 5).T

        self.data = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )

        self.upper_radii = [0.7]
        self.lower_radii = [0.5]
        self.n_nearest = 5

        self.gsp = GriSPy(self.data)

    def test_grid_data(self, setUp):
        assert_(isinstance(self.gsp.dim, int))
        assert_(isinstance(self.gsp.data, np.ndarray))
        assert_(isinstance(self.gsp.k_bins, np.ndarray))
        assert_(isinstance(self.gsp.metric, str))
        assert_(isinstance(self.gsp.N_cells, int))
        assert_(isinstance(self.gsp.grid, dict))
        assert_(isinstance(self.gsp.periodic, dict))
        assert_(isinstance(self.gsp.periodic_flag, bool))
        assert_(isinstance(self.gsp.time, dict))

    def test_bubble_single_query(self, setUp):

        b, ind = self.gsp.bubble_neighbors(
            np.array([[0, 0, 0]]), distance_upper_bound=self.upper_radii[0]
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)

    def test_shell_single_query(self, setUp):

        b, ind = self.gsp.shell_neighbors(
            np.array([[0, 0, 0]]),
            distance_lower_bound=self.lower_radii[0],
            distance_upper_bound=self.upper_radii[0],
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)

    def test_nearest_neighbors_single_query(self, setUp):

        b, ind = self.gsp.nearest_neighbors(
            np.array([[0, 0, 0]]), n=self.n_nearest
        )

        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), 1)
        assert_equal(len(ind), 1)
        assert_equal(np.shape(b[0]), (self.n_nearest,))
        assert_equal(np.shape(ind[0]), (self.n_nearest,))

    def test_bubble_multiple_query(self, setUp):

        b, ind = self.gsp.bubble_neighbors(
            self.centres, distance_upper_bound=self.upper_radii[0]
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(self.centres))
        assert_equal(len(ind), len(self.centres))

    def test_shell_multiple_query(self, setUp):

        b, ind = self.gsp.shell_neighbors(
            self.centres,
            distance_lower_bound=self.lower_radii[0],
            distance_upper_bound=self.upper_radii[0],
        )
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(self.centres))
        assert_equal(len(ind), len(self.centres))

    def test_nearest_neighbors_multiple_query(self, setUp):

        b, ind = self.gsp.nearest_neighbors(self.centres, n=self.n_nearest)
        assert_(isinstance(b, list))
        assert_(isinstance(ind, list))
        assert_equal(len(b), len(ind))
        assert_equal(len(b), len(self.centres))
        assert_equal(len(ind), len(self.centres))
        for i in range(len(b)):
            assert_equal(np.shape(b[i]), (self.n_nearest,))
            assert_equal(np.shape(ind[i]), (self.n_nearest,))

class Test_data_consistency_periodic:

    @pytest.fixture
    def setUp(self):

        np.random.seed(1234)
        self.centres = np.random.rand(3, 5).T

        self.data = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )

        self.upper_radii = [0.7]
        self.lower_radii = [0.5]
        self.n_nearest = 5

        self.periodic = {0: (0.0, 1.0)}

        self.gsp = GriSPy(self.data, periodic=self.periodic)

    def test_mirror_universe(self, setUp):
        r_cen = np.array([[0, 0, 0]])
        t_cen, t_ind = self.gsp._mirror_universe(
            r_cen, distance_upper_bound=self.upper_radii
        )
        assert_(isinstance(t_cen, np.ndarray))
        assert_(isinstance(t_ind, np.ndarray))
        assert_equal(len(t_cen), len(t_ind))
        assert_equal(t_cen.ndim, t_cen.ndim)
        assert_equal(t_cen.shape[1], r_cen.shape[1])

    def test_mirror(self, setUp):
        t_cen = self.gsp._mirror(
            np.array([[0, 0, 0]]), distance_upper_bound=self.upper_radii
        )
        assert_(isinstance(t_cen, np.ndarray))

    def test_near_boundary(self, setUp):
        mask = self.gsp._near_boundary(
            np.array([[0, 0, 0]]), distance_upper_bound=self.upper_radii
        )
        assert_(isinstance(mask, np.ndarray))
        assert_equal(mask.ndim, 1)

def test__init__A_01():
    # Data type
    data = 4
    periodic = {0: None, 1: None}
    with pytest.raises(TypeError, match=r".*must be a numpy array*"):
        gsp = GriSPy(
            data=data,
            N_cells=2,
            copy_data=False,
            periodic=periodic,
            metric="sphere",
        )


def test__init__A_02():
    # Data format
    data = np.array([])
    periodic = {0: None, 1: None}
    with pytest.raises(ValueError):
        gsp = GriSPy(
            data=data,
            N_cells=2,
            copy_data=False,
            periodic=periodic,
            metric="sphere",
        )


def test__init__A_03():
    # Data format
    data = np.array([1, 1, 1])
    periodic = {0: None, 1: None}
    with pytest.raises(ValueError):
        gsp = GriSPy(
            data=data,
            N_cells=2,
            copy_data=False,
            periodic=periodic,
            metric="sphere",
        )


def test__init__A_04():
    # Data value
    data = np.array([[]])
    periodic = {0: None, 1: None}
    with pytest.raises(ValueError):
        gsp = GriSPy(
            data=data,
            N_cells=2,
            copy_data=False,
            periodic=periodic,
            metric="sphere",
        )       


class Test_valid_query_input:
    @pytest.fixture
    def setUp(self):

        data = np.random.uniform(-1, 1, size=(100, 3))
        periodic = {0: (-1, 1)}
        self.gsp = GriSPy(data, periodic=periodic)

        # Define valid input data
        self.centres = np.random.uniform(-1, 1, size=(10, 3))
        self.upper_radii = 0.8
        self.lower_radii = 0.4
        self.kind = "quicksort"
        self.sorted = True
        self.n = 5

    def test_invalid_centres(self, setUp):
        # Invalid type
        bad_centres = [[1, 1, 1], [2, 2, 2]]
        with pytest.raises(TypeError):
            dd, ii = self.gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=self.upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        bad_centres = np.random.uniform(-1, 1, size=(10, 3))
        bad_centres[4, 1] = np.inf    # add one invalid value
        with pytest.raises(ValueError):
            dd, ii = self.gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=self.upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Invalid shape
        bad_centres = np.random.uniform(-1, 1, size=(10, 2))
        with pytest.raises(ValueError):
            dd, ii = self.gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=self.upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Invalid shape
        bad_centres = np.array([[], [], []]).reshape((0, 3))
        with pytest.raises(ValueError):
            dd, ii = self.gsp.bubble_neighbors(
                bad_centres,
                distance_upper_bound=self.upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

    def test_invalid_bounds(self, setUp):

        # Invalid type
        bad_upper_radii = list(np.random.uniform(0.6, 1, size=10))
        with pytest.raises(TypeError):
            dd, ii = self.gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Invalid value
        bad_upper_radii = np.random.uniform(0.6, 1, size=10)
        bad_upper_radii[5] = -1.
        with pytest.raises(ValueError):
            dd, ii = self.gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Different lenght than centres
        bad_upper_radii = np.random.uniform(0.6, 1, size=11)
        with pytest.raises(ValueError):
            dd, ii = self.gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Different lenght than centres
        lower_radii = np.random.uniform(0.1, 0.5, size=10)
        bad_upper_radii = np.random.uniform(0.6, 1, size=11)
        with pytest.raises(ValueError):
            dd, ii = self.gsp.shell_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                distance_lower_bound=lower_radii,
                sorted=self.sorted,
                kind=self.kind,
            )

        # Invalid value
        bad_upper_radii = 10.  # larger than periodic range
        with pytest.raises(ValueError):
            dd, ii = self.gsp.bubble_neighbors(
                self.centres,
                distance_upper_bound=bad_upper_radii,
                sorted=self.sorted,
                kind=self.kind,
            )


class Test_valid_init:
    @pytest.fixture
    def setUp(self):
        # Define valid input data
        self.data = np.random.uniform(-1, 1, size=(100, 3))
        self.periodic = {0: (-1, 1), 1: (-1, 1), 2: None}
        self.metric = "euclid"
        self.N_cells = 10
        self.copy_data = False

    def test_invalid_data(self, setUp):
        bad_data = np.random.uniform(-1, 1, size=(100, 3))
        bad_data[42, 1] = np.inf    # add one invalid value
        with pytest.raises(ValueError):
            gsp = GriSPy(
                bad_data,
                N_cells=self.N_cells,
                periodic=self.periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

    def test_invalid_periodic(self, setUp):
        # Axis 0 with invalid type: string instead of dict
        bad_periodic = '{0: [-1, 1], 1: (-1, 1), 2: None}'
        with pytest.raises(TypeError):
            gsp = GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=bad_periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

        # Axis 0 with invalid value type: list instead of tuple
        bad_periodic = {0: [-1, 1], 1: (-1, 1), 2: None}
        with pytest.raises(TypeError):
            gsp = GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=bad_periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

        # Axis is not integer
        bad_periodic = {'A': (-1, 1), 1: (-1, 1), 2: None}
        with pytest.raises(TypeError):
            gsp = GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=bad_periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

        # Edge 0 is larger than edge 1
        bad_periodic = {0: (1, -1), 1: (-1, 1), 2: None}
        with pytest.raises(ValueError):
            gsp = GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=bad_periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

        # Edge has wrong type
        bad_periodic = {0: (-1, [1]), 1: (-1, 1), 2: None}
        with pytest.raises(TypeError):
            gsp = GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=bad_periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

    def test_invalid_metric(self, setUp):
        # Metric name is not a string
        bad_metric = 42
        with pytest.raises(TypeError):
            gsp = GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=self.periodic,
                metric=bad_metric,
                copy_data=self.copy_data,
            )

        # Metric name is wrong
        bad_metric = "euclidean"
        with pytest.raises(ValueError):
            gsp = GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=self.periodic,
                metric=bad_metric,
                copy_data=self.copy_data,
            )

    def test_invalid_Ncells(self, setUp):
        # N_cells is not integer
        bad_N_cells = 10.5
        with pytest.raises(TypeError):
            gsp = GriSPy(
                self.data,
                N_cells=bad_N_cells,
                periodic=self.periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

        # N_cells is not positive
        bad_N_cells = -10
        with pytest.raises(ValueError):
            gsp = GriSPy(
                self.data,
                N_cells=bad_N_cells,
                periodic=self.periodic,
                metric=self.metric,
                copy_data=self.copy_data,
            )

    def test_invalid_copy_data(self, setUp):
        # copy_data is not bool
        bad_copy_data = 42
        with pytest.raises(TypeError):
            gsp = GriSPy(
                self.data,
                N_cells=self.N_cells,
                periodic=self.periodic,
                metric=self.metric,
                copy_data=bad_copy_data,
            )
