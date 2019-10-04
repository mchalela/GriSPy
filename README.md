# GriSPy (Grid Search in Python)


**GriSPy** is a regular grid search algorithm for quick nearest-neighbor lookup.

This class indexes a set of k-dimensional points in a regular grid providing a fast aproach for nearest neighbors queries. Optional periodic boundary conditions can be provided for each axis individually.

GriSPy has the following queries implemented:
- **bubble_neighbors**: find neighbors within a given radius. A different radius for each centre can be provided.
- **shell_neighbors**: find neighbors within given lower and upper radius. Different lower and upper radius can be provided for each centre.
- **nearest_neighbors**: find the nth nearest neighbors for each centre.

And the following methods are available:
- **set_periodicity**: define the periodicity conditions.
- **save_grid**: save the grid for future use.
- **load_grid**: load a previously saved grid.

--------------------------------

## Requirements

You need Python 3.7 or later to run GriSPy. You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems.


## Development Install

1.  Clone this repo and then inside the local
2.  Execute

        $ pip install -e .

## Authors

Martin Chalela (E-mail: tinchochalela@gmail.com),  
Emanuel Sillero, Luis Pereyra and Alejandro Garcia
