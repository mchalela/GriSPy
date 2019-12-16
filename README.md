# GriSPy (Grid Search in Python)
[![Build Status](https://travis-ci.org/mchalela/GriSPy.svg?branch=master)](https://travis-ci.org/mchalela/GriSPy) [![Documentation Status](https://readthedocs.org/projects/grispy/badge/?version=latest)](https://grispy.readthedocs.io/en/latest/?badge=latest) 
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.6](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

![logo](https://github.com/mchalela/GriSPy/raw/master/res/logo.png)

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

1.  Clone this repo and then inside the local directory
2.  Execute

        $ pip install -e .

## Authors

Martin Chalela (E-mail: tinchochalela@gmail.com),  
Emanuel Sillero, Luis Pereyra and Alejandro Garcia
