Tutorial
--------

**Example in 2D Uniform Distribution**

This example generates a 2D random uniform distribution, and them uses GriSPy to search neighbors within a given radius and/or the n-nearest neighbors

----------------------------------------------------------------

Import GriSPy and others packages::

	>>> import numpy as np
	>>> import matplotlib.pyplot as plt

	>>> from grispy import GriSPy

Create random points and centres::

	>>> Npoints = 10 ** 3
	>>> Ncentres = 1
	>>> dim = 2
	>>> Lbox = 100.0

	>>> np.random.seed(0)
	>>> data = np.random.uniform(0, Lbox, size=(Npoints, dim))
	>>> centres = np.random.uniform(-Lbox / 2, Lbox / 2, size=(Ncentres, dim))

Build the grid with the data::

	>>> gsp = GriSPy(data)

Set periodicity. Periodic conditions on x-axis (or axis=0) and y-axis (or axis=1)::

	
	>>> periodic = {0: (0, Lbox), 1: (0, Lbox)}
	>>> gsp.set_periodicity(periodic)

Also you can build a periodic grid in the same step::

	>>> gsp = GriSPy(data, periodic=periodic)

Query for neighbors within upper_radii::

	>>> upper_radii = 10.0
	>>> bubble_dist, bubble_ind = gsp.bubble_neighbors(
	...    	data, distance_upper_bound=upper_radii
	... )
	

Query for neighbors in a shell within lower_radii and upper_radii::

	>>> upper_radii = 10.0
	>>> lower_radii = 8.0
	>>> shell_dist, shell_ind = gsp.shell_neighbors(centres,
	... 	distance_lower_bound=lower_radii,
	... 	distance_upper_bound=upper_radii
	... )

Query for nth nearest neighbors::
	
	>>> n_nearest = 10
	>>> near_dist, near_ind = gsp.nearest_neighbors(centres, n=n_nearest)


Plot bubble results::

	>>> plt.figure()
	>>> plt.title("Bubble query")
	>>> plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=3)
	>>> for i in range(Ncentres):
	...	ind_i = bubble_ind[i]
	...	plt.scatter(data[ind_i, 0], data[ind_i, 1], c="C3", marker="o", s=5)
	>>> plt.plot(centres[:,0],centres[:,1],'ro',ms=10)

Plot shell results::

	>>> plt.figure()
	>>> plt.title("Shell query")
	>>> plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=2)
	>>> for i in range(Ncentres):
	...	ind_i = bubble_ind[i]
	...	plt.scatter(data[ind_i, 0], data[ind_i, 1], c="C2", marker="o", s=5)
	>>> plt.plot(centres[:,0],centres[:,1],'ro',ms=10)

Plot nearest results::

	>>> plt.figure()
	>>> plt.title("n-Nearest query")
	>>> plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=2)
	>>> for i in range(Ncentres):
	...	ind_i = near_ind[i]
	...	plt.scatter(data[ind_i, 0], data[ind_i, 1], c="C0", marker="o", s=5)
	>>> plt.plot(centres[:,0],centres[:,1],'ro',ms=10)
