API Module
==========

| GriSPy has the following queries implemented:
|	- **bubble_neighbors**: find neighbors within a given radius. A different radius for each centre can be provided.
|	- **shell_neighbors**: find neighbors within given lower and upper radius. Different lower and upper radius can be provided for each centre.
|	- **nearest_neighbors**: find the nth nearest neighbors for each centre.

| And the following methods are available:
|	- **set_periodicity**: define the periodicity conditions.
| 	- **save_grid**: save the grid for future use.
| 	- **load_grid**: load a previously saved grid.

-----------------------------------------------------------------

.. toctree::
   :maxdepth: 2

.. automodule:: grispy.core
   :members:
   :show-inheritance:
   :member-order: bysource