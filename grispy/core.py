#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""GriSPy core class."""

# =============================================================================
# IMPORTS
# =============================================================================

import time
import datetime

import itertools

import numpy as np

import attr

from . import distances, validators as vlds


# =============================================================================
# CONSTANTS
# =============================================================================

METRICS = {
    "euclid": distances.euclid,
    "haversine": distances.haversine,
    "vincenty": distances.vincenty}


EMPTY_ARRAY = np.array([], dtype=int)


# =============================================================================
#  TIME CLASS
# =============================================================================

@attr.s(frozen=True)
class BuildStats:
    """Statistics about the grid creation.

    Attributes
    ----------
    buildtime: float
        The number of seconds expended in build the grid.
    periodicity_set_at: datetime.datetime
        The date and time when the periodicity was setted.
    datetime: datetime.datetime
        The date and time of build.
    """

    buildtime = attr.ib()
    periodicity_set_at = attr.ib()
    datetime = attr.ib()


@attr.s(frozen=True)
class PeriodicityConf:
    """Internal representation of the periodicity of the Grid."""

    periodic_flag = attr.ib()
    pd_hi = attr.ib()
    pd_low = attr.ib()
    periodic_edges = attr.ib()
    periodic_direc = attr.ib()


# =============================================================================
# MAIN CLASS
# =============================================================================

@attr.s
class Grid:
    """Grid indexing.

    Grid is a regular grid indexing algorithm. This class indexes a set of
    k-dimensional points in a regular grid.

    To be implemented:
    - cell_id: Return grid indices for a given point.
    - cell_center: Return cell center coordinates for a given cell id.
    - cell_count: Return number of points within a given cell id.
    - cell_points: Return indices of points within a given cell id.

    Parameters
    ----------
    data: ndarray, shape(n,k)
        The n data points of dimension k to be indexed. This array is not
        copied, and so modifying this data may result in erroneous results.
        The data can be copied if the grid is built with copy_data=True.
    N_cells: positive int, optional
        The number of cells of each dimension to build the grid. The final
        grid will have N_cells**k number of cells. Default: 64
    copy_data: bool, optional
        Flag to indicate if the data should be copied in memory.
        Default: False

    Attributes
    ----------
    dim_: int
        The dimension of a single data-point.
    grid_: dict
        This dictionary contains the data indexed in a grid. The key is a
        tuple with the k-dimensional index of each grid cell. Empty cells
        do not have a key. The value is a list of data points indices which
        are located within the given cell.
    k_bins_: ndarray, shape (N_cells+1,k)
        The limits of the grid cells in each dimension.
    time_: grispy.core.BuildStats
        Object containing the building time and the date of build.

    """

    # User input params
    data = attr.ib(default=None, kw_only=False, repr=False)
    N_cells = attr.ib(default=64)
    copy_data = attr.ib(
        default=False, validator=attr.validators.instance_of(bool))

    # =========================================================================
    # ATTRS INITIALIZATION
    # =========================================================================

    def __attrs_post_init__(self):
        """Init more params and build the grid."""
        if self.copy_data:
            self.data = self.data.copy()

        t0 = time.time()
        self.grid_, self.k_bins_ = self._build_grid(
            data=self.data,
            N_cells=self.N_cells,
            dim=self.dim_)

        # Record date and build time
        now = datetime.datetime.now()
        self.time_ = BuildStats(
            buildtime=time.time() - t0,
            periodicity_set_at=now, datetime=now)

    @data.validator
    def _validate_data(self, attribute, value):
        """Validate init params: data."""
        # Chek if numpy array
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "Data: Argument must be a numpy array."
                "Got instead type {}".format(type(value)))
        # Check if data has the expected dimension
        if value.ndim != 2:
            raise ValueError(
                "Data: Array has the wrong shape. Expected shape of (n, k), "
                "got instead {}".format(value.shape))
        # Check if data has the expected dimension
        if len(value.flatten()) == 0:
            raise ValueError("Data: Array must have at least 1 point")

        # Check if every data point is valid
        if not np.isfinite(value).all():
            raise ValueError("Data: Array must have real numbers")

    @N_cells.validator
    def _validate_N_cells(self, attr, value):
        """Validate init params: N_cells."""
        # Chek if int
        if not isinstance(value, int):
            raise TypeError(
                "N_cells: Argument must be an integer. "
                "Got instead type {}".format(type(value)))
        # Check if N_cells is valid, i.e. higher than 1
        if value < 1:
            raise ValueError(
                "N_cells: Argument must be higher than 1. "
                "Got instead {}".format(value))

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def dim_(self):
        """Dimension of a single data-point."""
        return self.data.shape[1]

    # =========================================================================
    # INTERNAL IMPLEMENTATION
    # =========================================================================

    def _digitize(self, data, bins):
        """Return data bin index."""
        # allowed indeces with int16: (-32768 to 32767)
        d = ((data - bins[0]) / (bins[1] - bins[0])).astype(np.int16)
        return d

    def _build_grid(self, data, N_cells, dim):
        """Build the grid."""
        # Check the resolution of the input data and increase it
        # one order of magnitude. This works for float{32,64,128}
        # Fix issue #7
        dtype = data.dtype
        if np.issubdtype(dtype, np.integer):
            epsilon = 1e-1
        else:
            # assume floating
            epsilon = np.finfo(dtype).resolution * 10

        data_ind = np.arange(len(data))
        k_bins = np.zeros((N_cells + 1, dim))
        k_digit = np.zeros(data.shape, dtype=int)
        for k in range(dim):
            k_data = data[:, k]
            k_bins[:, k] = np.linspace(
                k_data.min() - epsilon,
                k_data.max() + epsilon,
                N_cells + 1)
            k_digit[:, k] = self._digitize(k_data, bins=k_bins[:, k])

        # Check that there is at least one point per cell
        grid = {}
        if N_cells ** dim < len(data):
            compact_ind = np.ravel_multi_index(
                k_digit.T, (N_cells,) * dim, order="F", mode='clip')

            compact_ind_sort = np.argsort(compact_ind)
            compact_ind = compact_ind[compact_ind_sort]
            k_digit = k_digit[compact_ind_sort]

            split_ind = np.searchsorted(
                compact_ind, np.arange(N_cells ** dim))
            deleted_cells = np.diff(np.append(-1, split_ind)).astype(bool)
            split_ind = split_ind[deleted_cells]
            if split_ind[-1] > data_ind[-1]:
                split_ind = split_ind[:-1]

            list_ind = np.split(data_ind[compact_ind_sort], split_ind[1:])
            k_digit = k_digit[split_ind]

            for i, j in enumerate(k_digit):
                grid[tuple(j)] = tuple(list_ind[i])
        else:
            for i in range(len(data)):
                cell_point = tuple(k_digit[i, :])
                if cell_point not in grid:
                    grid[cell_point] = [i]
                else:
                    grid[cell_point].append(i)
        return grid, k_bins

    # =========================================================================
    # GRID API
    # =========================================================================

    def cell_id(self, points):
        """Return grid indices for a given point."""
        raise NotImplementedError("Method not implemented.")

    def cell_center(self, ids):
        """Return cell center coordinates for a given cell id."""
        raise NotImplementedError("Method not implemented.")

    def cell_count(self, ids):
        """Return number of points within a given cell id."""
        raise NotImplementedError("Method not implemented.")

    def cell_points(self, ids):
        """Return indices of points within a given cell id."""
        raise NotImplementedError("Method not implemented.")


@attr.s
class GriSPy(Grid):
    """Grid Search in Python.

    GriSPy is a regular grid search algorithm for quick nearest-neighbor
    lookup.

    This class indexes a set of k-dimensional points in a regular grid
    providing a fast aproach for nearest neighbors queries. Optional periodic
    boundary conditions can be provided for each axis individually.

    The algorithm has the following queries implemented:
    - bubble_neighbors: find neighbors within a given radius. A different
    radius for each centre can be provided.
    - shell_neighbors: find neighbors within given lower and upper radius.
    Different lower and upper radius can be provided for each centre.
    - nearest_neighbors: find the nth nearest neighbors for each centre.

    Other methods:
    - set_periodicity: set periodicity condition after the grid was built.

    To be implemented:
    - box_neighbors: find neighbors within a k-dimensional squared box of
    a given size and orientation.
    - n_jobs: number of cores for parallel computation.

    Parameters
    ----------
    data: ndarray, shape(n,k)
        The n data points of dimension k to be indexed. This array is not
        copied, and so modifying this data may result in erroneous results.
        The data can be copied if the grid is built with copy_data=True.
    N_cells: positive int, optional
        The number of cells of each dimension to build the grid. The final
        grid will have N_cells**k number of cells. Default: 64
    copy_data: bool, optional
        Flag to indicate if the data should be copied in memory.
        Default: False
    periodic: dict, optional
        Dictionary indicating if the data domain is periodic in some or all its
        dimensions. The key is an integer that correspond to the number of
        dimensions in data, going from 0 to k-1. The value is a tuple with the
        domain limits and the data must be contained within these limits. If an
        axis is not specified, or if its value is None, it will be considered
        as non-periodic. Important: The periodicity only works within one
        periodic range. Default: all axis set to None.
        Example, periodic = { 0: (0, 360), 1: None}.
    metric: str, optional
        Metric definition to compute distances. Options: 'euclid', 'haversine'
        'vincenty' or a custom callable.


    Attributes
    ----------
    dim_: int
        The dimension of a single data-point.
    grid_: dict
        This dictionary contains the data indexed in a grid. The key is a
        tuple with the k-dimensional index of each grid cell. Empty cells
        do not have a key. The value is a list of data points indices which
        are located within the given cell.
    k_bins_: ndarray, shape (N_cells+1,k)
        The limits of the grid cells in each dimension.
    time_: grispy.core.BuildStats
        Object containing the building time and the date of build.
    periodic_flag_: bool
        If any dimension has periodicity.
    periodic_conf_: grispy.core.PeriodicityConf
        Statistics and intermediate results to make easy and fast the searchs
        with periodicity.

    """

    # User input params
    periodic = attr.ib(factory=dict)
    metric = attr.ib(default="euclid")

    # =========================================================================
    # ATTRS INITIALIZATION
    # =========================================================================

    def __attrs_post_init__(self):
        """Init more params and build the grid."""
        super().__attrs_post_init__()

        self.periodic, self.periodic_conf_ = self._build_periodicity(
            periodic=self.periodic, dim=self.dim_)

    @metric.validator
    def _validate_metric(self, attr, value):
        """Validate init params: metric."""
        # Check if name is valid
        if value not in METRICS and not callable(value):
            metric_names = ", ".join(METRICS)
            raise ValueError(
                "Metric: Got an invalid name: '{}'. "
                "Options are: {} or a callable".format(value, metric_names))

    @periodic.validator
    def _validate_periodic(self, attr, value):

        # Chek if dict
        if not isinstance(value, dict):
            raise TypeError(
                "Periodicity: Argument must be a dictionary. "
                "Got instead type {}".format(type(value)))

        # If dict is empty means no perioity, stop validation.
        if len(value) == 0:
            return

        # Check if keys and values are valid
        for k, v in value.items():
            # Check if integer
            if not isinstance(k, int):
                raise TypeError(
                    "Periodicity: Keys must be integers. "
                    "Got instead type {}".format(type(k)))

            # Check if tuple or None
            if not (isinstance(v, tuple) or v is None):
                raise TypeError(
                    "Periodicity: Values must be tuples. "
                    "Got instead type {}".format(type(v)))
            if v is None:
                continue

            # Check if edges are valid numbers
            has_valid_number = all([
                isinstance(v[0], (int, float)),
                isinstance(v[1], (int, float))])
            if not has_valid_number:
                raise TypeError(
                    "Periodicity: Argument must be a tuple of "
                    "2 real numbers as edge descriptors. ")

            # Check that first number is lower than second
            if not v[0] < v[1]:
                raise ValueError(
                    "Periodicity: First argument in tuple must be "
                    "lower than second argument.")

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def periodic_flag_(self):
        """Proxy to ``periodic_conf_.periodic_flag``."""
        return self.periodic_conf_.periodic_flag

    # =========================================================================
    # INTERNAL IMPLEMENTATION
    # =========================================================================

    def _build_periodicity(self, periodic, dim):
        """Cleanup the periodicity configuration.

        Remove the unnecessary axis from the periodic dict and also creates
        a configuration for use in the search.

        """
        cleaned_periodic = {}
        if len(periodic) == 0:
            periodic_flag = False
            pd_hi, pd_low = None, None
            periodic_edges, periodic_direc = None, None
        else:
            periodic_flag = any(
                [x is not None for x in list(periodic.values())])

            if periodic_flag:

                pd_hi = np.ones((1, dim)) * np.inf
                pd_low = np.ones((1, dim)) * -np.inf
                periodic_edges = []
                for k in range(dim):
                    aux = periodic.get(k)
                    cleaned_periodic[k] = aux
                    if aux:
                        pd_low[0, k] = aux[0]
                        pd_hi[0, k] = aux[1]
                        aux = np.insert(aux, 1, 0.)
                    else:
                        aux = np.zeros((1, 3))
                    periodic_edges = np.hstack([
                        periodic_edges,
                        np.tile(aux, (3**(dim - 1 - k), 3**k)).T.ravel()
                    ])

                periodic_edges = periodic_edges.reshape(
                    dim, 3**dim).T
                periodic_edges -= periodic_edges[::-1]
                periodic_edges = np.unique(periodic_edges, axis=0)

                mask = periodic_edges.sum(axis=1, dtype=bool)
                periodic_edges = periodic_edges[mask]

                periodic_direc = np.sign(periodic_edges)

        return cleaned_periodic, PeriodicityConf(
            periodic_flag=periodic_flag,
            pd_hi=pd_hi, pd_low=pd_low,
            periodic_edges=periodic_edges, periodic_direc=periodic_direc)

    def _distance(self, centre_0, centres):
        """Compute distance between points.

        metric options: 'euclid', 'sphere'

        Notes: In the case of 'sphere' metric, the input units must be degrees.

        """
        if len(centres) == 0:
            return EMPTY_ARRAY.copy()
        metric_func = (
            self.metric if callable(self.metric) else METRICS[self.metric])
        return metric_func(centre_0, centres, self.dim_)

    def _get_neighbor_distance(self, centres, neighbor_cells):
        """Retrieve neighbor distances whithin the given cells."""
        # combine the centres with the neighbors
        centres_ngb = zip(centres, neighbor_cells)

        n_idxs, n_dis = [], []
        for centre, neighbors in centres_ngb:

            if len(neighbors) == 0:  # no hay celdas vecinas
                n_idxs.append(EMPTY_ARRAY.copy())
                n_dis.append(EMPTY_ARRAY.copy())
                continue

            # Genera una lista con los vecinos de cada celda
            ind_tmp = [self.grid_.get(nt, []) for nt in map(tuple, neighbors)]

            # Une en una sola lista todos sus vecinos
            inds = np.fromiter(itertools.chain(*ind_tmp), dtype=np.int32)
            n_idxs.append(inds)

            if self.dim_ == 1:
                dis = self._distance(centre, self.data[inds])
            else:
                dis = self._distance(centre, self.data.take(inds, axis=0))
            n_dis.append(dis)

        return n_dis, n_idxs

    # Neighbor-cells methods
    def _get_neighbor_cells(
        self,
        centres,
        distance_upper_bound,
        distance_lower_bound=0,
        shell_flag=False,
    ):
        """Retrieve cells touched by the search radius."""
        cell_point = np.zeros((len(centres), self.dim_), dtype=int)
        out_of_field = np.zeros(len(cell_point), dtype=bool)
        for k in range(self.dim_):
            cell_point[:, k] = (
                self._digitize(centres[:, k], bins=self.k_bins_[:, k])
            )
            out_of_field[
                (centres[:, k] - distance_upper_bound > self.k_bins_[-1, k])
            ] = True
            out_of_field[
                (centres[:, k] + distance_upper_bound < self.k_bins_[0, k])
            ] = True

        if np.all(out_of_field):
            # no neighbor cells
            return [EMPTY_ARRAY.copy() for _ in centres]

        # Armo la caja con celdas a explorar
        k_cell_min = np.zeros((len(centres), self.dim_), dtype=int)
        k_cell_max = np.zeros((len(centres), self.dim_), dtype=int)
        for k in range(self.dim_):
            k_cell_min[:, k] = self._digitize(
                centres[:, k] - distance_upper_bound, bins=self.k_bins_[:, k])
            k_cell_max[:, k] = self._digitize(
                centres[:, k] + distance_upper_bound, bins=self.k_bins_[:, k])

            k_cell_min[k_cell_min[:, k] < 0, k] = 0
            k_cell_max[k_cell_max[:, k] < 0, k] = 0
            k_cell_min[k_cell_min[:, k] >= self.N_cells, k] = self.N_cells - 1
            k_cell_max[k_cell_max[:, k] >= self.N_cells, k] = self.N_cells - 1

        cell_size = self.k_bins_[1, :] - self.k_bins_[0, :]
        cell_radii = 0.5 * np.sum(cell_size ** 2) ** 0.5

        neighbor_cells = []
        for i, centre in enumerate(centres):
            # Para cada centro i, agrego un arreglo con shape (:,k)
            k_grids = [
                np.arange(k_cell_min[i, k], k_cell_max[i, k] + 1)
                for k in range(self.dim_)]
            k_grids = np.meshgrid(*k_grids)
            neighbor_cells += [
                np.array(list(map(np.ndarray.flatten, k_grids))).T]

            # Calculo la distancia de cada centro i a sus celdas vecinas,
            # luego descarto las celdas que no toca el circulo definido por
            # la distancia
            cells_physical = [
                self.k_bins_[neighbor_cells[i][:, k], k] + 0.5 * cell_size[k]
                for k in range(self.dim_)]

            cells_physical = np.array(cells_physical).T
            mask_cells = (
                self._distance(
                    centre, cells_physical
                ) < distance_upper_bound[i] + cell_radii)

            if shell_flag:
                mask_cells *= (
                    self._distance(
                        centre, cells_physical
                    ) > distance_lower_bound[i] - cell_radii)

            if np.any(mask_cells):
                neighbor_cells[i] = neighbor_cells[i][mask_cells]
            else:
                neighbor_cells[i] = EMPTY_ARRAY.copy()
        return neighbor_cells

    def _near_boundary(self, centres, distance_upper_bound):
        mask = np.zeros((len(centres), self.dim_), dtype=bool)
        for k in range(self.dim_):
            if self.periodic[k] is None:
                continue
            mask[:, k] = abs(
                centres[:, k] - self.periodic[k][0]) < distance_upper_bound
            mask[:, k] += abs(
                centres[:, k] - self.periodic[k][1]) < distance_upper_bound
        return mask.sum(axis=1, dtype=bool)

    def _mirror(self, centre, distance_upper_bound):
        pd_hi, pd_low, periodic_edges, periodic_direc = (
            self.periodic_conf_.pd_hi, self.periodic_conf_.pd_low,
            self.periodic_conf_.periodic_edges,
            self.periodic_conf_.periodic_direc)

        mirror_centre = centre - periodic_edges

        mask = periodic_direc * distance_upper_bound
        mask = mask + mirror_centre
        mask = (mask >= pd_low) * (mask <= pd_hi)
        mask = np.prod(mask, 1, dtype=bool)
        return mirror_centre[mask]

    def _mirror_universe(self, centres, distance_upper_bound):
        """Generate Terran centres in the Mirror Universe."""
        terran_centres = np.array([[]] * self.dim_).T
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
                    (terran_centres, mirror_centre), axis=0)
                terran_indices = np.concatenate(
                    (terran_indices, np.repeat(i, len(mirror_centre))))
        return terran_centres, terran_indices

    # =========================================================================
    # PERIODICITY
    # =========================================================================

    def set_periodicity(self, periodic, inplace=False):
        """Set periodicity conditions.

        This allows to define or change the periodicity limits without
        having to construct the grid again.

        Important: The periodicity only works within one periodic range.

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
        inplace: boolean, optional (default=False)
            If its True, set the periodicity on the current GriSPy instance
            and return None. Otherwise a new instance is created and
            returned.

        """
        if inplace:

            periodic_attr = attr.fields(GriSPy).periodic
            periodic_attr.validator(self, periodic_attr, periodic)
            self.periodic, self.periodic_conf_ = self._build_periodicity(
                periodic=periodic, dim=self.dim_)

            self.time_ = BuildStats(
                buildtime=self.time_.buildtime,
                datetime=self.time_.datetime,
                periodicity_set_at=datetime.datetime.now())
        else:
            return GriSPy(
                data=self.data, N_cells=self.N_cells,
                metric=self.metric, copy_data=self.copy_data,
                periodic=periodic)

    # =========================================================================
    # SEARCH API
    # =========================================================================

    def bubble_neighbors(
        self,
        centres,
        distance_upper_bound=-1.0,
        sorted=False,
        kind="quicksort",
    ):
        """Find all points within given distances of each centre.

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
        # Validate iputs
        vlds.validate_centres(centres, self.data)
        vlds.validate_distance_bound(distance_upper_bound, self.periodic)
        vlds.validate_bool(sorted)
        vlds.validate_sortkind(kind)
        # Match distance_upper_bound shape with centres shape
        if np.isscalar(distance_upper_bound):
            distance_upper_bound *= np.ones(len(centres))
        else:
            vlds.validate_equalsize(centres, distance_upper_bound)

        # Get neighbors
        neighbor_cells = self._get_neighbor_cells(
            centres, distance_upper_bound)

        neighbors_distances, neighbors_indices = self._get_neighbor_distance(
            centres, neighbor_cells)

        # We need to generate mirror centres for periodic boundaries...
        if self.periodic_flag_:
            terran_centres, terran_indices = self._mirror_universe(
                centres, distance_upper_bound)

            # terran_centres are the centres in the mirror universe for those
            # near the boundary.
            terran_neighbor_cells = self._get_neighbor_cells(
                terran_centres, distance_upper_bound[terran_indices])

            terran_neighbors_distances, \
                terran_neighbors_indices = self._get_neighbor_distance(
                    terran_centres, terran_neighbor_cells)

            for i, t in zip(terran_indices, np.arange(len(terran_centres))):
                # i runs over normal indices that have a terran counterpart
                # t runs over terran indices, 0 to len(terran_centres)
                neighbors_distances[i] = np.concatenate(
                    (neighbors_distances[i], terran_neighbors_distances[t]))

                neighbors_indices[i] = np.concatenate(
                    (neighbors_indices[i], terran_neighbors_indices[t]))

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
        """Find all points within given lower and upper distances of each centre.

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
        # Validate inputs
        vlds.validate_centres(centres, self.data)
        vlds.validate_bool(sorted)
        vlds.validate_sortkind(kind)
        vlds.validate_shell_distances(
            distance_lower_bound, distance_upper_bound, self.periodic)

        # Match distance bounds shapes with centres shape
        if np.isscalar(distance_lower_bound):
            distance_lower_bound *= np.ones(len(centres))
        else:
            vlds.validate_equalsize(centres, distance_lower_bound)
        if np.isscalar(distance_upper_bound):
            distance_upper_bound *= np.ones(len(centres))
        else:
            vlds.validate_equalsize(centres, distance_upper_bound)

        # Get neighbors
        neighbor_cells = self._get_neighbor_cells(
            centres,
            distance_upper_bound=distance_upper_bound,
            distance_lower_bound=distance_lower_bound,
            shell_flag=True)

        neighbors_distances, neighbors_indices = self._get_neighbor_distance(
            centres, neighbor_cells)

        # We need to generate mirror centres for periodic boundaries...
        if self.periodic_flag_:
            terran_centres, terran_indices = self._mirror_universe(
                centres, distance_upper_bound)

            # terran_centres are the centres in the mirror universe for those
            # near the boundary.
            terran_neighbor_cells = self._get_neighbor_cells(
                terran_centres, distance_upper_bound[terran_indices])

            terran_neighbors_distances,\
                terran_neighbors_indices = self._get_neighbor_distance(
                    terran_centres, terran_neighbor_cells)

            for i, t in zip(terran_indices, np.arange(len(terran_centres))):
                # i runs over normal indices that have a terran counterpart
                # t runs over terran indices, 0 to len(terran_centres)
                neighbors_distances[i] = np.concatenate(
                    (neighbors_distances[i], terran_neighbors_distances[t]))

                neighbors_indices[i] = np.concatenate(
                    (neighbors_indices[i], terran_neighbors_indices[t]))

        for i in range(len(centres)):
            mask_distances_upper = (
                neighbors_distances[i] <= distance_upper_bound[i])

            mask_distances_lower = neighbors_distances[i][mask_distances_upper]
            mask_distances_lower = (
                mask_distances_lower > distance_lower_bound[i])

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
        """Find the n nearest-neighbors for each centre.

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
        # Validate input
        vlds.validate_centres(centres, self.data)
        vlds.validate_n_nearest(n, self.data, self.periodic)
        vlds.validate_sortkind(kind)

        # Initial definitions
        N_centres = len(centres)
        centres_lookup_ind = np.arange(0, N_centres)
        n_found = np.zeros(N_centres, dtype=bool)
        lower_distance_tmp = np.zeros(N_centres)
        upper_distance_tmp = np.zeros(N_centres)

        # First estimation is the cell radii
        cell_size = self.k_bins_[1, :] - self.k_bins_[0, :]
        cell_radii = 0.5 * np.sum(cell_size ** 2) ** 0.5

        upper_distance_tmp = cell_radii * np.ones(N_centres)

        neighbors_indices = [EMPTY_ARRAY.copy() for _ in range(N_centres)]
        neighbors_distances = [EMPTY_ARRAY.copy() for _ in range(N_centres)]
        while not np.all(n_found):
            ndis_tmp, nidx_tmp = self.shell_neighbors(
                centres[~n_found],
                distance_lower_bound=lower_distance_tmp[~n_found],
                distance_upper_bound=upper_distance_tmp[~n_found])

            for i_tmp, i in enumerate(centres_lookup_ind[~n_found]):
                if n <= len(nidx_tmp[i_tmp]) + len(
                    neighbors_indices[i]
                ):
                    n_more = n - len(neighbors_indices[i])
                    n_found[i] = True
                else:
                    n_more = len(nidx_tmp[i_tmp])
                    lower_distance_tmp[i] = upper_distance_tmp[i].copy()
                    upper_distance_tmp[i] += cell_size.min()

                sorted_ind = np.argsort(ndis_tmp[i_tmp], kind=kind)[:n_more]
                neighbors_distances[i] = np.hstack((
                    neighbors_distances[i], ndis_tmp[i_tmp][sorted_ind]))

                neighbors_indices[i] = np.hstack((
                    neighbors_indices[i], nidx_tmp[i_tmp][sorted_ind]))

        return neighbors_distances, neighbors_indices
