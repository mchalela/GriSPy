# GriSPy
GriSPy (Grid Search in Python) is a regular grid search algorithm for quick nearest-neighbor lookup.

This class indexes a set of k-dimensional points in a regular grid providing a fast aproach for nearest neighbors queries. Optional periodic boundary conditions can be provided for each axis individually.

GriSPy has the following queries implemented:
- **bubble_neighbors**: find neighbors within a given radius. A different radius for each centre can be provided.
- **shell_neighbors**: find neighbors within given lower and upper radius. Different lower and upper radius can be provided for each centre.
- **nearest_neighbors**: find the nth nearest neighbors for each centre.

And the following methods are available:
- **set_periodicity**: define the periodicity conditions.
- **save_grid**: save the grid for future use.
- **load_grid**: load a previously saved grid.
