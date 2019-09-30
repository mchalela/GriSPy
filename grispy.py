import numpy as np
import time
import datetime
# from multiprocessing import Process, Queue

###############################################################################


class GriSPy(object):
    """ Grid Search in Python.
    GriSPy is a regular grid search algorithm for quick nearest-neighbor
    lookup.

    This class indexes a set of k-dimensional points in a regular grid
    providing a fast aproach for nearest neighbors queries. Optional periodic
    boundary conditions can be provided for each axis individually.

    The algorithm has the following queries implemented:

    bubble_neighbors: find neighbors within a given radius. A different
        radius for each centre can be provided.
    shell_neighbors: find neighbors within given lower and upper radius.
        Different lower and upper radius can be provided for each centre.
    nearest_neighbors: find the nth nearest neighbors for each centre.
    save_grid: save the grid for future use.

    To be implemented:
    box_neighbors: find neighbors within a k-dimensional squared box of
        a given size and orientation.
    n_jobs: number of cores for parallel computation.

    Parameters
    ----------
    data: ndarray, shape(n,k)
        The n data points of dimension k to be indexed. This array is not
        copied, and so modifying this data may result in erroneous results.
        The data can be copied if the grid is built with copy_data=True.
    N_cells: positive int, optional
        The number of cells of each dimension to build the grid. The final
        grid will have N_cells**k number of cells. Default: 20
    copy_data: bool, optional
        Flag to indicate if the data should be copied in memory.
        Default: False
    periodic: dict, optional
        Dictionary indicating if the data domain is periodic in some or all its
        dimensions. The key is an integer that correspond to the number of
        dimensions in data, going from 0 to k-1. The value is a tuple with the
        domain limits and the data must be contained within these limits. If an
        axis is not specified, or if its value is None, it will be considered
        as non-periodic. Default: all axis set to None.
        Example, periodic = { 0: (0, 360), 1: None}.
    load_grid: str, optional
        String with the file name of a previously saved GriSPy grid (see
        save_grid method). If a grid is loaded, the "data" field stored within
        it will take precedence over the "data" keyword passed on construction.
    metric: str, optional
        Metric definition to compute distances. Options: 'euclid' or 'sphere'.
        Notes: In the case of 'sphere' metric, input units must be degrees.

    Attributes
    ----------
    data: ndarray, shape (n,k)
        The n data points of dimension k to be indexed. This array is not
        copied, and so modifying this data may result in erroneous results.
        The data can be copied if the grid is built with copy_data=True.
    N_cells: int
        The number of cells of each dimension to build the grid.
    dim: int
        The dimension of a single data-point.
    grid: dict
        This dictionary contains the data indexed in a grid. The key is a
        tuple with the k-dimensional index of each grid cell. Empty cells
        do not have a key. The value is a list of data points indices which
        are located within the given cell.
    k_bins: ndarray, shape (N_cells+1,k)
        The limits of the grid cells in each dimension.
    time: dict
        Dictionary containing the building time and the date of build.
        keys: 'buildtime', returns float with the time taken to build the grid,
        in seconds; 'datetime': formated string with the date of build.
    """

    def __init__(
        self,
        data=None,
        N_cells=20,
        copy_data=False,
        periodic={},
        metric="euclid",
        load_grid=None,
    ):

        if load_grid is None:
            if not isinstance(data, np.ndarray):
                raise TypeError("Argument 'data' must be a numpy array.")
            self.data = data.copy() if copy_data else data
            if self._check_data_dimensionality(self.data.shape):
                self.dim = self.data.shape[1]
            self._check_data_amount(self.data)
            self.N_cells = N_cells
            self.metric = metric
            self.set_periodicity(periodic)
            self._build_grid()
        else:
            f = np.load(load_grid, allow_pickle=True).item()
            self.N_cells = f["N_cells"]
            self.dim = f["dim"]
            self.metric = f["metric"]
            self.set_periodicity(f["periodic"])
            self.grid = f["grid"]
            self.k_bins = f["k_bins"]
            self.time = f["time"]
            self.data = f["data"]
            print(
                "Succsefully loaded GriSPy grid created on {}".format(
                    self.time["datetime"])
            )

        self._empty = np.array([], dtype=int)  # Useful for empty arrays

    def __getitem__(self, key):
        return getattr(self, key)

    def _build_grid(self, epsilon=1.0e-6):
        """ Builds the grid
        """
        t0 = time.time()
        data_ind = np.arange(len(self.data))
        self.k_bins = np.zeros((self.N_cells + 1, self.dim))
        k_digit = np.zeros(self.data.shape, dtype=int)
        for k in range(self.dim):
            k_data = self.data[:, k]
            self.k_bins[:, k] = np.linspace(
                k_data.min() - epsilon,
                k_data.max() + epsilon,
                self.N_cells + 1,
            )
            k_digit[:, k] = np.digitize(k_data, bins=self.k_bins[:, k]) - 1

        # Check that there is at least one point per cell
        # if self.N_cells ** self.dim < len(self.data):
        if True:
            compact_ind = np.ravel_multi_index(
                k_digit.T,
                (self.N_cells,) * self.dim,
                order="F",
            )

            compact_ind_sort = np.argsort(compact_ind)
            compact_ind = compact_ind[compact_ind_sort]
            k_digit = k_digit[compact_ind_sort]

            split_ind = np.searchsorted(
                compact_ind, np.arange(self.N_cells ** self.dim)
            )
            deleted_cells = np.diff(np.append(-1, split_ind)).astype(bool)
            split_ind = split_ind[deleted_cells]
            if split_ind[-1] >= data_ind[-1]:
                split_ind = split_ind[:-1]

            list_ind = np.split(data_ind[compact_ind_sort], split_ind[1:])
            k_digit = k_digit[split_ind]

            self.grid = {}
            for i, j in enumerate(k_digit):
                self.grid[tuple(j)] = list(list_ind[i])

        # Record date and build time
        self.time = {"buildtime": time.time() - t0}
        currentDT = datetime.datetime.now()
        self.time["datetime"] = currentDT.strftime("%Y-%b-%d %H:%M:%S")

    def distance(self, centre_0, centres):
        """ Computes the distance between points
        metric: 'euclid', 'sphere'

        Notes: In the case of 'sphere' metric, the input units must be degrees.
        """
        if len(centres) == 0:
            return self._empty
        if self.metric == "euclid":
            return np.sqrt(((centres - centre_0) ** 2).sum(axis=1))
        elif self.metric == "sphere":
            lon1 = np.deg2rad(centre_0[0])
            lat1 = np.deg2rad(centre_0[1])
            lon2 = np.deg2rad(centres[:, 0])
            lat2 = np.deg2rad(centres[:, 1])

            sdlon = np.sin(lon2 - lon1)
            cdlon = np.cos(lon2 - lon1)
            slat1 = np.sin(lat1)
            slat2 = np.sin(lat2)
            clat1 = np.cos(lat1)
            clat2 = np.cos(lat2)
            num1 = clat2 * sdlon
            num2 = clat1 * slat2 - slat1 * clat2 * cdlon
            denominator = slat1 * slat2 + clat1 * clat2 * cdlon
            sep = np.arctan2(np.sqrt(num1 ** 2 + num2 ** 2), denominator)
            return np.rad2deg(sep)

    def _get_neighbor_distance(self, centres, neighbor_cells):
        """ Retrieves the neighbor distances whithin the given cells
        """
        neighbors_indices = []
        neighbors_distances = []
        for i in range(len(centres)):
            if len(neighbor_cells[i]) == 0:  # no hay celdas vecinas
                neighbors_indices += [self._empty]
                neighbors_distances += [self._empty]
                continue
            # Genera una lista con los vecinos de cada celda
            # print neighbor_cells[i]
            ind_tmp = [
                self.grid.get(tuple(neighbor_cells[i][j]), [])
                for j in range(len(neighbor_cells[i]))
            ]
            # Une en una sola lista todos sus vecinos
            neighbors_indices += [np.concatenate(ind_tmp).astype(int)]
            neighbors_distances += [
                self.distance(centres[i], self.data[neighbors_indices[i], :])
            ]
        return neighbors_distances, neighbors_indices

    # Neighbor-cells methods
    def _get_neighbor_cells(
        self,
        centres,
        distance_upper_bound,
        distance_lower_bound=0,
        shell_flag=False,
    ):
        """ Retrieves the cells touched by the search radius
        """
        cell_point = np.zeros((len(centres), self.dim), dtype=int)
        out_of_field = np.zeros(len(cell_point), dtype=bool)
        for k in range(self.dim):
            cell_point[:, k] = (
                np.digitize(centres[:, k], bins=self.k_bins[:, k]) - 1
            )
            out_of_field[
                (centres[:, k] - distance_upper_bound > self.k_bins[-1, k])
            ] = True
            out_of_field[
                (centres[:, k] + distance_upper_bound < self.k_bins[0, k])
            ] = True

        if np.all(out_of_field):
            return [self._empty] * len(centres)  # no neighbor cells

        # Armo la caja con celdas a explorar
        k_cell_min = np.zeros((len(centres), self.dim), dtype=int)
        k_cell_max = np.zeros((len(centres), self.dim), dtype=int)
        for k in range(self.dim):
            k_cell_min[:, k] = (
                np.digitize(
                    centres[:, k] - distance_upper_bound,
                    bins=self.k_bins[:, k],
                ) - 1
            )
            k_cell_max[:, k] = (
                np.digitize(
                    centres[:, k] + distance_upper_bound,
                    bins=self.k_bins[:, k],
                ) - 1
            )

            k_cell_min[k_cell_min[:, k] < 0, k] = 0
            k_cell_max[k_cell_max[:, k] < 0, k] = 0
            k_cell_min[k_cell_min[:, k] >= self.N_cells, k] = self.N_cells - 1
            k_cell_max[k_cell_max[:, k] >= self.N_cells, k] = self.N_cells - 1

        cell_size = self.k_bins[1, :] - self.k_bins[0, :]
        cell_radii = 0.5 * np.sum(cell_size ** 2) ** 0.5

        neighbor_cells = []
        for i in range(len(centres)):
            # Para cada centro i, agrego un arreglo con shape (:,k)
            k_grids = [
                np.arange(k_cell_min[i, k], k_cell_max[i, k] + 1)
                for k in range(self.dim)
            ]
            k_grids = np.meshgrid(*k_grids)
            neighbor_cells += [
                np.array(list(map(np.ndarray.flatten, k_grids))).T
            ]

            # Calculo la distancia de cada centro i a sus celdas vecinas,
            # luego descarto las celdas que no toca el circulo definido por
            # la distancia
            cells_physical = [
                self.k_bins[neighbor_cells[i][:, k], k] + 0.5 * cell_size[k]
                for k in range(self.dim)
            ]
            cells_physical = np.array(cells_physical).T
            mask_cells = (
                self.distance(
                    centres[i], cells_physical
                ) < distance_upper_bound[i] + cell_radii
            )

            if shell_flag:
                mask_cells *= (
                    self.distance(
                        centres[i], cells_physical
                    ) > distance_lower_bound[i] - cell_radii
                )

            if np.any(mask_cells):
                neighbor_cells[i] = neighbor_cells[i][mask_cells]
            else:
                neighbor_cells[i] = self._empty
        return neighbor_cells

    def _near_boundary(self, centres, distance_upper_bound):
        mask = np.zeros((len(centres), self.dim), dtype=bool)
        for k in range(self.dim):
            if self.periodic[k] is None:
                continue
            mask[:, k] = abs(
                centres[:, k] - self.periodic[k][0]
            ) < distance_upper_bound
            mask[:, k] += abs(
                centres[:, k] - self.periodic[k][1]
            ) < distance_upper_bound
        return mask.sum(axis=1, dtype=bool)

    def _mirror(self, centre, distance_upper_bound):
        mirror_centre = centre - self._periodic_edges
        mask = 0.5 * np.abs(mirror_centre) < distance_upper_bound
        mask = np.sum(mask, 1, dtype=bool)
        return mirror_centre[mask]

    def _mirror_universe(self, centres, distance_upper_bound):
        """Generate Terran centres in the Mirror Universe
        """
        terran_centres = np.array([[]] * self.dim).T
        terran_indices = np.array([], dtype=int)
        near_boundary = self._near_boundary(centres, distance_upper_bound)
        if not np.any(near_boundary):
            return terran_centres, terran_indices

        for i, centre in enumerate(centres):
            if not near_boundary[i]:
                continue
            mirror_centre = self._mirror(centre, distance_upper_bound[i])
            if len(mirror_centre) > 0:
                terran_centres = np.concatenate(
                    (terran_centres, mirror_centre), axis=0
                )
                terran_indices = np.concatenate(
                    (terran_indices, np.repeat(i, len(mirror_centre)))
                )
        return terran_centres, terran_indices

    def _check_data_dimensionality(self, shape):
        """ Check if data has the expected dimension
        """
        if len(shape) == 2:
            return True
        else:
            raise ValueError(
                "Data array has the wrong shape. Expected shape of (n, k), "
                "got instead {}".format(shape)
            )

    def _check_data_amount(self, data):
        """ Check if data has the expected dimension
        """
        if len(data.flatten()) > 0:
            return True
        else:
            raise ValueError("Data must have at least 1 point")

    def _check_centre_dimensionality(self, shape):
        """ Check if centres has the same dimension as data
        """
        if len(shape) == 2 and shape[1] == self.dim:
            return None
        else:
            raise ValueError(
                "Centre array has the wrong shape. Expected shape of (m, {}), "
                "got instead {}".format(self.dim, shape)
            )

    # User methods
    def bubble_neighbors(
        self,
        centres,
        distance_upper_bound=-1.0,
        sorted=False,
        kind="quicksort",
    ):
        """
        Find all points within given distances of each centre. Different
        distances for each point can be provided.

        Parameters
        ----------
        centres: ndarray, shape (m,k)
            The point or points to search for neighbors of.
        distance_upper_bound: scalar or ndarray of length m
            The radius of points to return. If a scalar is provided, the same
            distance will apply for every centre. An ndarray with individual
            distances can also be rovided.
        sorted: bool, optional
            If True the returned neighbors will be ordered by increasing
            distance to the centre. Default: False.
        kind: str, optional
            When sorted = True, the sorting algorithm can be specified in this
            keyword. Available algorithms are: ['quicksort', 'mergesort',
            'heapsort', 'stable']. Default: 'quicksort'
        njobs: int, optional
            Number of jobs for parallel computation. Not implemented yet.

        Returns
        -------
        distances: list, length m
            Returns a list of m arrays. Each array has the distances to the
            neighbors of that centre.

        indices: list, length m
            Returns a list of m arrays. Each array has the indices to the
            neighbors of that centre.
        """

        # Check centres has the correct dimension
        self._check_centre_dimensionality(centres.shape)

        # Match distance_upper_bound shape with centres shape
        if np.isscalar(distance_upper_bound):
            distance_upper_bound *= np.ones(len(centres))
        elif len(centres) != len(distance_upper_bound):
            raise ValueError(
                "If an array is given in 'distance_upper_bound', "
                "its size must be the same as the number of centres."
            )

        neighbor_cells = self._get_neighbor_cells(
            centres, distance_upper_bound
        )
        neighbors_distances, neighbors_indices = self._get_neighbor_distance(
            centres, neighbor_cells
        )

        # We need to generate mirror centres for periodic boundaries...
        if self.periodic_flag:
            terran_centres, terran_indices = self._mirror_universe(
                centres, distance_upper_bound
            )
            # terran_centres are the centres in the mirror universe for those
            # near the boundary.
            terran_neighbor_cells = self._get_neighbor_cells(
                terran_centres, distance_upper_bound[terran_indices]
            )
            terran_neighbors_distances, \
                terran_neighbors_indices = self._get_neighbor_distance(
                    terran_centres, terran_neighbor_cells
                )

            for i, t in zip(terran_indices, np.arange(len(terran_centres))):
                # i runs over normal indices that have a terran counterpart
                # t runs over terran indices, 0 to len(terran_centres)
                neighbors_distances[i] = np.concatenate(
                    (neighbors_distances[i], terran_neighbors_distances[t])
                )
                neighbors_indices[i] = np.concatenate(
                    (neighbors_indices[i], terran_neighbors_indices[t])
                )

        for i in range(len(centres)):
            mask_distances = neighbors_distances[i] <= distance_upper_bound[i]
            neighbors_distances[i] = neighbors_distances[i][mask_distances]
            neighbors_indices[i] = neighbors_indices[i][mask_distances]
            if sorted:
                sorted_ind = np.argsort(neighbors_distances[i], kind=kind)
                neighbors_distances[i] = neighbors_distances[i][sorted_ind]
                neighbors_indices[i] = neighbors_indices[i][sorted_ind]
        return neighbors_distances, neighbors_indices

    def shell_neighbors(
        self,
        centres,
        distance_lower_bound=-1.0,
        distance_upper_bound=-1.0,
        sorted=False,
        kind="quicksort",
    ):
        """
        Find all points within given lower and upper distances of each centre.
        Different distances for each point can be provided.

        Parameters
        ----------
        centres: ndarray, shape (m,k)
            The point or points to search for neighbors of.
        distance_lower_bound: scalar or ndarray of length m
            The minimum distance of points to return. If a scalar is provided,
            the same distance will apply for every centre. An ndarray with
            individual distances can also be rovided.
        distance_upper_bound: scalar or ndarray of length m
            The maximum distance of points to return. If a scalar is provided,
            the same distance will apply for every centre. An ndarray with
            individual distances can also be rovided.
        sorted: bool, optional
            If True the returned neighbors will be ordered by increasing
            distance to the centre. Default: False.
        kind: str, optional
            When sorted = True, the sorting algorithm can be specified in this
            keyword. Available algorithms are: ['quicksort', 'mergesort',
            'heapsort', 'stable']. Default: 'quicksort'
        njobs: int, optional
            Number of jobs for parallel computation. Not implemented yet.

        Returns
        -------
        distances: list, length m
            Returns a list of m arrays. Each array has the distances to the
            neighbors of that centre.

        indices: list, length m
            Returns a list of m arrays. Each array has the indices to the
            neighbors of that centre.
        """

        # Check centres has the correct dimension
        self._check_centre_dimensionality(centres.shape)

        # Match distance bounds shapes with centres shape
        if np.isscalar(distance_lower_bound):
            distance_lower_bound *= np.ones(len(centres))
        elif len(centres) != len(distance_lower_bound):
            raise ValueError(
                "If an array is given in 'distance_lower_bound', "
                "its size must be the same as the number of centres."
            )
        if np.isscalar(distance_upper_bound):
            distance_upper_bound *= np.ones(len(centres))
        elif len(centres) != len(distance_upper_bound):
            raise ValueError(
                "If an array is given in 'distance_upper_bound', "
                "its size must be the same as the number of centres."
            )
        if np.any(distance_lower_bound > distance_upper_bound):
            raise ValueError(
                "One or more values in 'distance_lower_bound' is greater "
                "than its 'distance_upper_bound' pair."
            )

        neighbor_cells = self._get_neighbor_cells(
            centres,
            distance_upper_bound=distance_upper_bound,
            distance_lower_bound=distance_lower_bound,
            shell_flag=True,
        )
        neighbors_distances, neighbors_indices = self._get_neighbor_distance(
            centres, neighbor_cells
        )

        # We need to generate mirror centres for periodic boundaries...
        if self.periodic_flag:
            terran_centres, terran_indices = self._mirror_universe(
                centres, distance_upper_bound
            )
            # terran_centres are the centres in the mirror universe for those
            # near the boundary.
            terran_neighbor_cells = self._get_neighbor_cells(
                terran_centres, distance_upper_bound[terran_indices]
            )
            terran_neighbors_distances,\
                terran_neighbors_indices = self._get_neighbor_distance(
                    terran_centres, terran_neighbor_cells
                )

            for i, t in zip(terran_indices, np.arange(len(terran_centres))):
                # i runs over normal indices that have a terran counterpart
                # t runs over terran indices, 0 to len(terran_centres)
                neighbors_distances[i] = np.concatenate(
                    (neighbors_distances[i], terran_neighbors_distances[t])
                )
                neighbors_indices[i] = np.concatenate(
                    (neighbors_indices[i], terran_neighbors_indices[t])
                )

        for i in range(len(centres)):
            mask_distances_upper = (
                neighbors_distances[i] <= distance_upper_bound[i]
            )
            mask_distances_lower = neighbors_distances[i][mask_distances_upper]
            mask_distances_lower = (
                mask_distances_lower > distance_lower_bound[i]
            )
            aux = neighbors_distances[i]
            aux = aux[mask_distances_upper]
            aux = aux[mask_distances_lower]
            neighbors_distances[i] = aux

            aux = neighbors_indices[i]
            aux = aux[mask_distances_upper]
            aux = aux[mask_distances_lower]
            neighbors_indices[i] = aux

            if sorted:
                sorted_ind = np.argsort(neighbors_distances[i], kind=kind)
                neighbors_distances[i] = neighbors_distances[i][sorted_ind]
                neighbors_indices[i] = neighbors_indices[i][sorted_ind]

        return neighbors_distances, neighbors_indices

    def nearest_neighbors(self, centres, n=1, kind="quicksort"):
        """
        Find the n nearest-neighbors for each centre.

        Parameters
        ----------
        centres: ndarray, shape (m,k)
            The point or points to search for neighbors of.
        n: int, optional
            The number of neighbors to fetch for each centre. Default: 1.
        kind: str, optional
            The returned neighbors will be ordered by increasing distance
            to the centre. The sorting algorithm can be specified in this
            keyword. Available algorithms are: ['quicksort', 'mergesort',
            'heapsort', 'stable']. Default: 'quicksort'
        njobs: int, optional
            Number of jobs for parallel computation. Not implemented yet.

        Returns
        -------
        distances: list, length m
            Returns a list of m arrays. Each array has the distances to the
            neighbors of that centre.

        indices: list, length m
            Returns a list of m arrays. Each array has the indices to the
            neighbors of that centre.
        """
        # Check centres has the correct dimension
        self._check_centre_dimensionality(centres.shape)

        # Initial definitions
        N_centres = len(centres)
        centres_lookup_ind = np.arange(0, N_centres)
        n_found = np.zeros(N_centres, dtype=bool)
        lower_distance_tmp = np.zeros(N_centres)
        upper_distance_tmp = np.zeros(N_centres)

        # Abro la celda del centro como primer paso
        centre_cell = self._get_neighbor_cells(
            centres, distance_upper_bound=upper_distance_tmp
        )
        # crear funcion que regrese vecinos sin calcular distancias
        neighbors_distances, neighbors_indices = self._get_neighbor_distance(
            centres, centre_cell
        )

        # Calculo una primera aproximacion con la
        # 'distancia media' = 0.5 * (n/denstiy)**(1/dim)
        # Factor de escala para la distancia inicial
        mean_distance_factor = 1.0
        cell_size = self.k_bins[1, :] - self.k_bins[0, :]
        cell_volume = np.prod(cell_size).astype(float)
        neighbors_number = np.array(list(map(len, neighbors_indices)))
        mean_distance = 0.5 * (n / (neighbors_number / cell_volume)) ** (
            1.0 / self.dim)
        mask_mean = np.isinf(mean_distance)
        mean_distance[mask_mean] = cell_size.min()
        upper_distance_tmp = mean_distance_factor * mean_distance

        neighbors_indices = [self._empty] * N_centres
        neighbors_distances = [self._empty] * N_centres
        while not np.all(n_found):
            neighbors_distances_tmp,\
                neighbors_indices_tmp = self.shell_neighbors(
                    centres[~n_found],
                    distance_lower_bound=lower_distance_tmp[~n_found],
                    distance_upper_bound=upper_distance_tmp[~n_found],
                )

            for i_tmp, i in enumerate(centres_lookup_ind[~n_found]):
                if n_found[i]:
                    continue
                if n <= len(neighbors_indices_tmp[i_tmp]) + len(
                    neighbors_indices[i]
                ):
                    n_more = n - len(neighbors_indices[i])
                    n_found[i] = True
                else:
                    n_more = len(neighbors_indices_tmp[i_tmp])
                    lower_distance_tmp[i_tmp] = upper_distance_tmp[
                        i_tmp
                    ].copy()
                    upper_distance_tmp[i_tmp] += cell_size.min()

                sorted_ind = np.argsort(
                    neighbors_distances_tmp[i_tmp], kind=kind
                )[:n_more]
                neighbors_distances[i] = np.hstack(
                    (
                        neighbors_distances[i],
                        neighbors_distances_tmp[i_tmp][sorted_ind],
                    )
                )
                neighbors_indices[i] = np.hstack(
                    (
                        neighbors_indices[i],
                        neighbors_indices_tmp[i_tmp][sorted_ind],
                    )
                )

        return neighbors_distances, neighbors_indices

    def save_grid(self, file="grispy.npy", overwrite=False):
        """
        Save all grid attributes in a binary file for future use.

        Parameters
        ----------
        file: string, optional
            File name where the grid will be saved. The file format is a numpy
            binary file with extension '.npy'. If the extension is not
            explicitely given it will be added automatically.
            Default: grispy.npy
        overwrite: bool, optional
            If True the file will be overwritten in case it already exists.
            Default: False
        """
        dic = {
            "grid": self.grid,
            "N_cells": self.N_cells,
            "dim": self.dim,
            "metric": self.metric,
            "periodic": self.periodic,
            "k_bins": self.k_bins,
            "time": self.time,
            "data": self.data
        }

        import os
        if not overwrite and os.path.isfile(file):
            raise FileExistsError(
                f"The file {file} already exists. "
                "You may want to use the keyword overwrite=True."
            )

        np.save(file, dic)
        print(f"GriSPy grid attributes saved to: {file}")

    def set_periodicity(self, periodic={}):
        """
        Set periodicity conditions. This allows to define or change the
        periodicity limits without having to construct the grid again.

        Parameters
        ----------
        periodic: dict, optional
            Dictionary indicating if the data domain is periodic in some or all
            its dimensions. The key is an integer that corresponds to the
            number of dimensions in data, going from 0 to k-1. The value is a
            tuple with the domain limits and the data must be contained within
            these limits. If an axis is not specified, or if its value is None,
            it will be considered as non-periodic.
            Default: all axis set to None.
            Example, periodic = { 0: (0, 360), 1: None}.
        """
        if len(periodic) == 0:
            self.periodic_flag = False
            self.periodic = {}
        else:
            self.periodic_flag = any(
                [x is not None for x in list(periodic.values())]
            )
            if self.periodic_flag:
                import itertools

                self.periodic = {k: periodic.get(k) for k in range(self.dim)}
                self._periodic_edges = [
                    (0, 0, 0) if not periodic.get(k)
                    else np.insert(periodic.get(k), 1, 0.)
                    for k in range(self.dim)
                ]
                self._periodic_edges = np.reshape(
                    list(itertools.product(*self._periodic_edges)),
                    (3 ** self.dim, self.dim)
                )
                self._periodic_edges -= self._periodic_edges[::-1]
                self._periodic_edges = np.unique(self._periodic_edges, axis=0)
                mask = self._periodic_edges.sum(axis=1, dtype=bool)
                self._periodic_edges = self._periodic_edges[mask]

###############################################################################
