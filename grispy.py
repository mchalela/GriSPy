import os
import numpy as np
import time, datetime
from multiprocessing import Process, Queue

######################################################################################################################

class GriSPy(object):
	'''	Grid Search in Python.
	GriSPy is a regular grid search algorithm for quick nearest-neighbor lookup.

	This class indexes a set of k-dimensional points in a regular grid providing
	a fast aproach for nearest neighbors queries. Optional periodic	boundary 
	conditions can be provided for each axis individually.

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
	    Metric definition to compute distances. Not implemented yet. Distances 
	    are computed with euclidean metric.

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
	'''

	def __init__(self, data=None, N_cells=20, copy_data=False, periodic={}, metric='euclid', load_grid=None):
		
		if load_grid is None:
			if type(data) is not np.ndarray:
				raise TypeError('Argument "data" must be a numpy array.')
			self.data = data.copy() if copy_data else data
			self.dim  = self.data.shape[1]
			self.N_cells = N_cells
			self.metric = metric
			self.set_periodicity(periodic)
			self._build_grid()
		else:
			f = np.load(load_grid).item()
			self.N_cells = f['N_cells']
			self.dim = f['dim']
			self.metric = f['metric']
			self.set_periodicity(f['periodic'])
			self.grid = f['grid']
			self.k_bins = f['k_bins']
			self.time = f['time']
			try:
				self.data = f['data']
			except Exception as e:
				self.data = data
			print 'Succsefully loaded GriSPy grid created on {}'.format(f['time']['datetime'])
		self._empty = np.array([], dtype=int) 	# Useful for empty arrays

	def _build_grid(self, epsilon=1.e-6):
		''' Builds the grid
		'''
		t0 = time.time()
		data_ind = np.arange(len(self.data))
		self.k_bins	= np.zeros((self.N_cells+1, self.dim))
		k_digit= np.zeros(self.data.shape, dtype=int)
		for k in xrange(self.dim):
			k_data = self.data[:,k]
			self.k_bins[:,k]  = np.linspace(k_data.min()-epsilon, k_data.max()+epsilon, self.N_cells+1)
			k_digit[:,k] = np.digitize(k_data, bins=self.k_bins[:,k])-1

		# Check that there is at least one point per cell
		if self.N_cells**self.dim < len(self.data):
			#compact_ind = np.sum(k_digit*(self.N_cells**np.arange(self.dim)), axis=1)	
			compact_ind = np.ravel_multi_index([k_digit[:,i] for i in xrange(self.dim)],(self.N_cells,)*self.dim,order='F')

			compact_ind_sort = np.argsort(compact_ind)
			compact_ind = compact_ind[compact_ind_sort]
			k_digit = k_digit[compact_ind_sort]

			split_ind = np.searchsorted(compact_ind, np.arange(self.N_cells**self.dim))
			if compact_ind[-1]<(self.N_cells**self.dim-1):	# Delete empty cells
				split_ind = np.delete(split_ind, range(compact_ind[-1],len(split_ind)))
			list_ind = np.split(data_ind[compact_ind_sort], split_ind[1:]) 
			k_digit = k_digit[split_ind]

			self.grid = {}
			for i,j in enumerate(k_digit): self.grid[tuple(j)] = list(list_ind[i])
		else:
			self.grid = {}
			for i in xrange(len(self.data)):
				cell_point = tuple(k_digit[i,:])
				if cell_point not in self.grid:
					self.grid[cell_point] = [i]
				else:
					self.grid[cell_point].append(i)

		# Record date and build time
		self.time = {'buildtime': time.time()-t0}
		currentDT = datetime.datetime.now()
		self.time['datetime'] = currentDT.strftime("%Y-%b-%d %H:%M:%S")

	def distance(self, centre_0, centres):
		''' Computes the distance between points
		metric: 'euclid', 'sphere'
		'''
		if len(centres)==0: return self._empty
		if self.metric == 'euclid':
			return np.sqrt(((centres-centre_0)**2).sum(axis=1))
		elif self.metric == 'sphere':
			lon1 = np.deg2rad(centre_0[0])
			lat1 = np.deg2rad(centre_0[1])
			lon2 = np.deg2rad(centres[:,0])
			lat2 = np.deg2rad(centres[:,1])

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
		''' Retrieves the neighbor distances whithin the given cells
		'''
		neighbors_indices = []
		neighbors_distances = []
		for i in xrange(len(centres)):
			if len(neighbor_cells[i])==0:	#no hay celdas vecinas
				neighbors_indices += [self._empty]
				neighbors_distances += [self._empty]
				continue
			# Genera una lista con los vecinos de cada celda
			#print neighbor_cells[i]
			ind_tmp = [self.grid.get(tuple(neighbor_cells[i][j]), []) for j in xrange(len(neighbor_cells[i]))]
			# Une en una sola lista todos sus vecinos
			neighbors_indices += [np.concatenate(ind_tmp).astype(int)]
			neighbors_distances += [self.distance(centres[i], self.data[neighbors_indices[i],:])]
		return neighbors_distances, neighbors_indices

	# Neighbor-cells methods
	def _get_neighbor_cells(self, centres, distance_upper_bound, distance_lower_bound=0, shell_flag=False):
		''' Retrieves the cells touched by the search radius
		'''
		cell_point = np.zeros((len(centres),self.dim), dtype=int)
		out_of_field = np.zeros(len(cell_point), dtype=bool)		
		for k in xrange(self.dim):
			cell_point[:,k] = np.digitize(centres[:,k], bins=self.k_bins[:,k])-1
			out_of_field[(centres[:,k]-distance_upper_bound>self.k_bins[-1,k])] = True
			out_of_field[(centres[:,k]+distance_upper_bound<self.k_bins[0,k])] = True
			
		if np.all(out_of_field): return [self._empty]*len(centres)	# no neighbor cells

		# Armo la caja con celdas a explorar
		k_cell_min = np.zeros((len(centres),self.dim), dtype=int)
		k_cell_max = np.zeros((len(centres),self.dim), dtype=int)
		for k in xrange(self.dim):
			k_cell_min[:,k] = np.digitize(centres[:,k]-distance_upper_bound, bins=self.k_bins[:,k])-1
			k_cell_max[:,k] = np.digitize(centres[:,k]+distance_upper_bound, bins=self.k_bins[:,k])-1

			k_cell_min[k_cell_min[:,k]<0,k] = 0
			k_cell_max[k_cell_max[:,k]<0,k] = 0
			k_cell_min[k_cell_min[:,k]>=self.N_cells,k] = self.N_cells-1
			k_cell_max[k_cell_max[:,k]>=self.N_cells,k] = self.N_cells-1

		cell_size = self.k_bins[1,:]-self.k_bins[0,:]
		cell_radii = 0.5 * np.sum(cell_size**2)**0.5

		neighbor_cells = []
		for i in xrange(len(centres)):
			# Para cada centro i, agrego un arreglo con shape (:,k)
			k_grids = [np.arange(k_cell_min[i,k],k_cell_max[i,k]+1) for k in xrange(self.dim)]
			k_grids = np.meshgrid(*k_grids)
			neighbor_cells += [np.array(map(np.ndarray.flatten, k_grids)).T]

			# Calculo la distancia de cada centro i a sus celdas vecinas,
			# luego descarto las celdas que no toca el circulo definido por la distancia
			cells_physical = np.array([self.k_bins[neighbor_cells[i][:,k], k] + 0.5*cell_size[k] for k in xrange(self.dim)]).T
			mask_cells = self.distance(centres[i], cells_physical) < distance_upper_bound[i]+cell_radii
			
			if shell_flag:
				mask_cells *= self.distance(centres[i], cells_physical) > distance_lower_bound[i]-cell_radii

			if np.any(mask_cells):
				neighbor_cells[i] = neighbor_cells[i][mask_cells]
			else:
				neighbor_cells[i] = self._empty
		return neighbor_cells

	def _mirror_universe(self, centres, distance_upper_bound):
		'''	Generate Terran centres in the Mirror Universe
		'''
		def _near_boundary(centres, distance_upper_bound):
			mask = np.zeros((len(centres), self.dim), dtype=bool)
			for k in xrange(self.dim):
				if self.periodic[k] is None: continue
			
				mask[:,k] = abs(centres[:,k]-self.periodic[k][0]) < distance_upper_bound
				mask[:,k] += abs(centres[:,k]-self.periodic[k][1]) < distance_upper_bound
			return mask.sum(axis=1, dtype=bool)

		# Constructs the indices outside _mirror() to not repeat calculation for every centre
		_ind = np.zeros((2**self.dim, self.dim), dtype=bool)
		for k in xrange(self.dim):
			_i = np.repeat([False,True], 2**k)
			_ind[:,k] = np.concatenate( (_i, )*(2**(self.dim-k-1)) )
		def _mirror(centre, distance_upper_bound):
			mirror_centre = np.repeat(centre[np.newaxis,:], 2**self.dim,axis=0)
			mask = np.ones(2**self.dim, dtype=bool)
			for k in xrange(self.dim):
				_i = _ind[:,k]
				if self.periodic[k] is None:
					mask[_i] = False 
					continue
				if abs(centre[k]-self.periodic[k][0]) < distance_upper_bound:
					mirror_centre[_i, k] += self.periodic[k][1] - self.periodic[k][0]
				elif abs(centre[k]-self.periodic[k][1]) < distance_upper_bound:
					mirror_centre[_i, k] -= self.periodic[k][1] - self.periodic[k][0]
				else:
					mask[_i] = False
			return mirror_centre[mask]

		terran_centres = np.array([[ ]]*self.dim).T
		terran_indices = np.array([], dtype=int)
		near_boundary = _near_boundary(centres, distance_upper_bound)
		if not np.any(near_boundary): return terran_centres, terran_indices

		for i, centre in enumerate(centres):
			if not near_boundary[i]: continue
			mirror_centre = _mirror(centre, distance_upper_bound[i])
			if len(mirror_centre)>0:
				terran_centres = np.concatenate((terran_centres, mirror_centre), axis=0)
				terran_indices = np.concatenate((terran_indices, np.repeat(i, len(mirror_centre))))
		
		return terran_centres, terran_indices

	def _bubble(self, centres, distance_upper_bound=-1., sorted=False, kind='quicksort'):
		''' Find all points within given distances of each centre.
		Different distances for each point can be provided.
		This should not be used. Instead, use shell_neighbors.
		'''
		neighbor_cells = self._get_neighbor_cells(centres, distance_upper_bound)
		neighbors_distances, neighbors_indices = self._get_neighbor_distance(centres, neighbor_cells)

		# We need to generate mirror centres for periodic boundaries...
		if self.periodic_flag:
			terran_centres, terran_indices = self._mirror_universe(centres, distance_upper_bound)
			# terran_centres are the centres in the mirror universe for those near the boundary.
			terran_neighbor_cells = self._get_neighbor_cells(terran_centres, distance_upper_bound[terran_indices])
			terran_neighbors_distances, terran_neighbors_indices = self._get_neighbor_distance(terran_centres, terran_neighbor_cells)

			for i, t in zip( terran_indices, np.arange(len(terran_centres)) ):
				# i runs over normal indices that have a terran counterpart
				# t runs over terran indices, 0 to len(terran_centres)
				neighbors_distances[i] = np.concatenate((neighbors_distances[i], terran_neighbors_distances[t]))
				neighbors_indices[i] = np.concatenate((neighbors_indices[i], terran_neighbors_indices[t]))

		for i in xrange(len(centres)):
			mask_distances = neighbors_distances[i]<=distance_upper_bound[i]
			neighbors_distances[i] = neighbors_distances[i][mask_distances]
			neighbors_indices[i] = neighbors_indices[i][mask_distances]
			if sorted:
				sorted_ind = np.argsort(neighbors_distances[i], kind=kind)
				neighbors_distances[i] = neighbors_distances[i][sorted_ind]
				neighbors_indices[i] = neighbors_indices[i][sorted_ind]
		return neighbors_distances, neighbors_indices

	def _shell(self, centres, distance_lower_bound=-1., distance_upper_bound=-1., sorted=False, kind='quicksort'):
		''' Find all points within given lower and upper distances of each centre.
		Different distances for each point can be provided.
		This should not be used. Instead, use shell_neighbors.
		'''

		neighbor_cells = self._get_neighbor_cells(centres, distance_upper_bound=distance_upper_bound,
						distance_lower_bound=distance_lower_bound, shell_flag=True)
		neighbors_distances, neighbors_indices = self._get_neighbor_distance(centres, neighbor_cells)

		# We need to generate mirror centres for periodic boundaries...
		if self.periodic_flag:
			terran_centres, terran_indices = self._mirror_universe(centres, distance_upper_bound)
			# terran_centres are the centres in the mirror universe for those near the boundary.
			terran_neighbor_cells = self._get_neighbor_cells(terran_centres, distance_upper_bound[terran_indices])
			terran_neighbors_distances, terran_neighbors_indices = self._get_neighbor_distance(terran_centres, terran_neighbor_cells)

			for i, t in zip( terran_indices, np.arange(len(terran_centres)) ):
				# i runs over normal indices that have a terran counterpart
				# t runs over terran indices, 0 to len(terran_centres)
				neighbors_distances[i] = np.concatenate((neighbors_distances[i], terran_neighbors_distances[t]))
				neighbors_indices[i] = np.concatenate((neighbors_indices[i], terran_neighbors_indices[t]))

		for i in xrange(len(centres)):
			mask_distances_upper = neighbors_distances[i]<=distance_upper_bound[i]
			mask_distances_lower = neighbors_distances[i][mask_distances_upper]>distance_lower_bound[i]
			neighbors_distances[i] = neighbors_distances[i][mask_distances_upper][mask_distances_lower]
			neighbors_indices[i] = neighbors_indices[i][mask_distances_upper][mask_distances_lower]
			if sorted:
				sorted_ind = np.argsort(neighbors_distances[i], kind=kind)
				neighbors_distances[i] = neighbors_distances[i][sorted_ind]
				neighbors_indices[i] = neighbors_indices[i][sorted_ind]
		return neighbors_distances, neighbors_indices

	def _nearest(self, centres, n=1, kind='quicksort'):
		'''Find the n nearest-neighbors for each centre.
		This should not be used. Instead, use nearest_neighbors.
		'''
		# Initial definitions
		N_centres = len(centres)
		centres_lookup_ind = np.arange(0,N_centres)
		n_found = np.zeros(N_centres, dtype=bool)
		lower_distance_tmp = np.zeros(N_centres)
		upper_distance_tmp = np.zeros(N_centres)

		# Abro la celda del centro como primer paso
		centre_cell = self._get_neighbor_cells(centres, distance_upper_bound=upper_distance_tmp)
		# crear funcion que regrese vecinos sin calcular distancias
		neighbors_distances, neighbors_indices = self._get_neighbor_distance(centres, centre_cell)

		# Calculo una primera aproximacion con la 'distancia media' = 0.5 * (n/denstiy)**(1/dim)
		# Factor de escala para la distancia inicial 
		mean_distance_factor = 1.
		cell_size = self.k_bins[1,:]-self.k_bins[0,:]
		cell_volume = np.prod(cell_size).astype(float)
		neighbors_number = np.array(map(len, neighbors_indices))
		mean_distance = 0.5 * (n/(neighbors_number/cell_volume))**(1./self.dim)
		mask_mean = np.isinf(mean_distance)
		mean_distance[mask_mean] = cell_size.min()
		upper_distance_tmp = mean_distance_factor * mean_distance

		neighbors_indices = [self._empty]*N_centres
		neighbors_distances = [self._empty]*N_centres
		while not np.all(n_found):	
			neighbors_distances_tmp, neighbors_indices_tmp = self.shell_neighbors(centres[~n_found],
															distance_lower_bound=lower_distance_tmp[~n_found],
															distance_upper_bound=upper_distance_tmp[~n_found])

			for i_tmp, i in enumerate(centres_lookup_ind[~n_found]):
				if n_found[i]: continue
				if n <= len(neighbors_indices_tmp[i_tmp])+len(neighbors_indices[i]):
					n_more = n-len(neighbors_indices[i])
					n_found[i] = True
				else:
					n_more = len(neighbors_indices_tmp[i_tmp])
					lower_distance_tmp[i_tmp] = upper_distance_tmp[i_tmp].copy()
					upper_distance_tmp[i_tmp] += cell_size.min()

				sorted_ind = np.argsort(neighbors_distances_tmp[i_tmp], kind=kind)[:n_more]
				neighbors_distances[i] = np.hstack((neighbors_distances[i],
													neighbors_distances_tmp[i_tmp][sorted_ind]))
				neighbors_indices[i] = np.hstack((neighbors_indices[i],
												  neighbors_indices_tmp[i_tmp][sorted_ind]))

		return neighbors_distances, neighbors_indices

	# User methods
	def save_grid(self, file='grispy.npy', save_data=False):
		'''
		Save all grid attributes in a binary file for future use.

		Note: Given that the memory needed to store the set of k-dimensional
		data points is expected to be large, saving the data in a GriSPy file is
		optional. If this data exists in another form, it is encouraged to use
		that. However, if the data was created dinamically (e.g. a catalog of
		random points) then you should definitely save it.

		Parameters
		----------
		file: string, optional
		    File name where the grid will be saved. The file format is a numpy
		    binary file with extension '.npy'. If the extension is not
		    explicitely given it will be added automatically.
		    Default: grispy.npy
		save_data: bool, optional
		    Indicates if the k-dimensional points should be saved.
		    Default: False
		'''
		dic = {'grid': self.grid, 'N_cells': self.N_cells,
				'dim': self.dim, 'metric': self.metric, 'periodic': self.periodic,
				'k_bins': self.k_bins, 'time': self.time}
		if save_data: dic['data'] = self.data

		np.save(file, dic)
		print 'GriSPy grid attributes saved to: {}'.format(file)

	def set_periodicity(self, periodic={}):
		'''
		Set periodicity conditions.	This allows to define or change the periodicity 
		limits without having to construct the grid again.

		Parameters
		----------
		periodic: dict, optional
		    Dictionary indicating if the data domain is periodic in some or all its
		    dimensions. The key is an integer that corresponds to the number of 
		    dimensions in data,	going from 0 to k-1. The value is a tuple with the
		    domain limits and the data must be contained within these limits. If an
		    axis is not specified, or if its value is None, it will be considered 
		    as non-periodic. Default: all axis set to None.
		    Example, periodic = { 0: (0, 360), 1: None}.
		'''
		if len(periodic)==0:
			self.periodic_flag = False
			self.periodic = {}
		else:
			self.periodic_flag = any(map(lambda x: x is not None, periodic.values()))	
			if self.periodic_flag:
				self.periodic = { k : periodic.get(k, None) for k in xrange(self.dim) }

	def bubble_neighbors(self, centres, distance_upper_bound=-1., sorted=False, kind='quicksort', njobs=1):	
		'''
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
		    Number of jobs for parallel computation. Default, njobs=1.	
		metric: str, optional
		    Metric definition to compute distances.	Not implemented yet.
		    Distances are computed with euclidean metric.

		Returns
		-------
		distances: list, length m
		    Returns a list of m arrays. Each array has the distances to the
		    neighbors of that centre.

		indices: list, length m
		    Returns a list of m arrays. Each array has the indices to the
		    neighbors of that centre.
		'''

		# Match distance_upper_bound shape with centres shape
		if np.isscalar(distance_upper_bound):
			distance_upper_bound *= np.ones(len(centres))
		elif len(centres)!=len(distance_upper_bound):
			raise ValueError("If an array is given in 'distance_upper_bound', \
								its size must be the same as the number of centres.")
		if njobs==1:
			return self._bubble(centres, distance_upper_bound=distance_upper_bound, sorted=sorted, kind=kind)

		task_queue = Queue()
		centres_id = range(len(centres))
		ic = len(centres)/njobs + 1
		for nj in xrange(njobs):
			args = (centres[nj*ic:(nj+1)*ic , :], distance_upper_bound[nj*ic:(nj+1)*ic], sorted, kind)
			task_queue.put( (args, centres_id[nj*ic:(nj+1)*ic]) )
		for nj in xrange(njobs):
			task_queue.put('STOP')

		def _worker(input, output):	################################################
			''' Sets the queue for parallel computation
			input is a queue with args tuples:
			    args are the arguments to pass to search
			output is a queue storing (distances, indices, id) tuples:
			    id is the centre id.
			'''
			for args, centres_id in iter(input.get, 'STOP'):
				neighbors_distances, neighbors_indices = self._bubble(*args)
				output.put( (neighbors_distances, neighbors_indices, centres_id) )

		done_queue = Queue()
		for nj in xrange(njobs):
			_l = (task_queue, done_queue)
			Process(target=_worker, args=(task_queue, done_queue)).start()

		neighbors_distances = []
		neighbors_indices = []
		centres_id = []
		for nj in xrange(njobs):
			dd, ii, cid = done_queue.get()
			neighbors_distances += dd
			neighbors_indices += ii
			centres_id += cid

		# This sorting is made to match the order of input and output centres id
		idsort = np.argsort(centres_id)
		neighbors_distances = np.array(neighbors_distances, dtype=object)[idsort]
		neighbors_indices = np.array(neighbors_indices, dtype=object)[idsort]
		return list(neighbors_distances), list(neighbors_indices)

	def shell_neighbors(self, centres, distance_lower_bound=-1., distance_upper_bound=-1., sorted=False, kind='quicksort', njobs=1):
		'''
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
		    Number of jobs for parallel computation. Default, njobs=1.		
		metric: str, optional
		    Metric definition to compute distances.	Not implemented yet.
		    Distances are computed with euclidean metric.

		Returns
		-------
		distances: list, length m
		    Returns a list of m arrays. Each array has the distances to the
		    neighbors of that centre.

		indices: list, length m
		    Returns a list of m arrays. Each array has the indices to the
		    neighbors of that centre.
		'''
		# Match distance bounds shapes with centres shape
		if np.isscalar(distance_lower_bound):
			distance_lower_bound *= np.ones(len(centres))
		elif len(centres)!=len(distance_lower_bound):
			raise ValueError("If an array is given in 'distance_lower_bound', \
								its size must be the same as the number of centres.")
		if np.isscalar(distance_upper_bound):
			distance_upper_bound *= np.ones(len(centres))
		elif len(centres)!=len(distance_upper_bound):
			raise ValueError("If an array is given in 'distance_upper_bound', \
								its size must be the same as the number of centres.")
		if np.any(distance_lower_bound > distance_upper_bound):
			raise ValueError("One or more values in 'distance_lower_bound' is greater \
								than its 'distance_upper_bound' pair.")

		if njobs==1:
			return self._shell(centres, distance_lower_bound=distance_lower_bound, 
						distance_upper_bound=distance_upper_bound, sorted=sorted, kind=kind)

		task_queue = Queue()
		centres_id = range(len(centres))
		ic = len(centres)/njobs + 1
		for nj in xrange(njobs):
			args = (centres[nj*ic:(nj+1)*ic , :], distance_lower_bound[nj*ic:(nj+1)*ic],
					distance_upper_bound[nj*ic:(nj+1)*ic], sorted, kind)
			task_queue.put( (args, centres_id[nj*ic:(nj+1)*ic]) )
		for nj in xrange(njobs):
			task_queue.put('STOP')

		def _worker(input, output):	################################################
			''' Sets the queue for parallel computation
			input is a queue with args tuples:
			    args are the arguments to pass to search
			output is a queue storing (distances, indices, id) tuples:
			    id is the centre id.
			'''
			for args, centres_id in iter(input.get, 'STOP'):
				neighbors_distances, neighbors_indices = self._shell(*args)
				output.put( (neighbors_distances, neighbors_indices, centres_id) )

		done_queue = Queue()
		for nj in xrange(njobs):
			Process(target=_worker, args=(task_queue, done_queue)).start()

		neighbors_distances = []
		neighbors_indices = []
		centres_id = []
		for nj in xrange(njobs):
			dd, ii, cid = done_queue.get()
			neighbors_distances += dd
			neighbors_indices += ii
			centres_id += cid

		# This sorting is made to match the order of input and output centres id
		idsort = np.argsort(centres_id)
		neighbors_distances = np.array(neighbors_distances, dtype=object)[idsort]
		neighbors_indices = np.array(neighbors_indices, dtype=object)[idsort]
		return list(neighbors_distances), list(neighbors_indices)

	def nearest_neighbors(self, centres, n=1, kind='quicksort', njobs=1):
		'''
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
		    Number of jobs for parallel computation. Default, njobs=1.	

		Returns
		-------
		distances: list, length m
		    Returns a list of m arrays. Each array has the distances to the
		    neighbors of that centre.

		indices: list, length m
		    Returns a list of m arrays. Each array has the indices to the
		    neighbors of that centre.
		'''

		if njobs==1:
			return self._nearest(centres, n=n, kind=kind)

		task_queue = Queue()
		centres_id = range(len(centres))
		ic = len(centres)/njobs + 1
		for nj in xrange(njobs):
			args = (centres[nj*ic:(nj+1)*ic , :], n, kind)
			task_queue.put( (args, centres_id[nj*ic:(nj+1)*ic]) )
		for nj in xrange(njobs):
			task_queue.put('STOP')

		def _worker(input, output):	################################################
			''' Sets the queue for parallel computation
			input is a queue with args tuples:
			    args are the arguments to pass to search
			output is a queue storing (distances, indices, id) tuples:
			    id is the centre id.
			'''
			for args, centres_id in iter(input.get, 'STOP'):
				neighbors_distances, neighbors_indices = self._nearest(*args)
				output.put( (neighbors_distances, neighbors_indices, centres_id) )

		done_queue = Queue()
		for nj in xrange(njobs):
			Process(target=_worker, args=(task_queue, done_queue)).start()

		neighbors_distances = []
		neighbors_indices = []
		centres_id = []
		for nj in xrange(njobs):
			dd, ii, cid = done_queue.get()
			neighbors_distances += dd
			neighbors_indices += ii
			centres_id += cid

		# This sorting is made to match the order of input and output centres id
		idsort = np.argsort(centres_id)
		neighbors_distances = np.array(neighbors_distances, dtype=object)[idsort]
		neighbors_indices = np.array(neighbors_indices, dtype=object)[idsort]
		return list(neighbors_distances), list(neighbors_indices)

		return

	# Plot grid
	def plot_grid(self, proyection='01'):
		x_proy, y_proy = int(proyection[0]),int(proyection[1])
		for i in xrange(len(self.k_bins[:,x_proy])):
			plt.axvline(self.k_bins[i,x_proy])
			plt.axhline(self.k_bins[i,y_proy])
		return plt.show()
######################################################################################################################


