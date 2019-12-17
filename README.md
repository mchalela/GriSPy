# GriSPy (Grid Search in Python)

![logo](https://github.com/mchalela/GriSPy/raw/master/res/logo_mid.png)


[![PyPi Version](https://badge.fury.io/py/grispy.svg)](https://badge.fury.io/py/grispy)
[![Build Status](https://travis-ci.org/mchalela/GriSPy.svg?branch=master)](https://travis-ci.org/mchalela/GriSPy)
[![Documentation Status](https://readthedocs.org/projects/grispy/badge/?version=latest)](https://grispy.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-370/)


**GriSPy** is a regular grid search algorithm for quick nearest-neighbor lookup.

This class indexes a set of k-dimensional points in a regular grid providing a fast aproach for nearest neighbors queries. Optional periodic boundary conditions can be provided for each axis individually.

GriSPy has the following queries implemented:
- **bubble_neighbors**: find neighbors within a given radius. A different radius for each centre can be provided.
- **shell_neighbors**: find neighbors within given lower and upper radius. Different lower and upper radius can be provided for each centre.
- **nearest_neighbors**: find the nth nearest neighbors for each centre.

And the following method is available:
- **set_periodicity**: define the periodicity conditions.

--------------------------------

## Requirements

You need Python 3.6 or later to run GriSPy. You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems.


## Standard Installation

GriSPy is available at [PyPI](https://pypi.org/project/grispy/). You can install it via the pip command

        $ pip install grispy

## Development Install

Clone this repo and then inside the local directory execute

        $ pip install -e .

## Authors

Martin Chalela (E-mail: tinchochalela@gmail.com),
Emanuel Sillero, Luis Pereyra and Alejandro Garcia
