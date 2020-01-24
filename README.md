# GriSPy (Grid Search in Python)

![logo](https://github.com/mchalela/GriSPy/raw/master/res/logo_mid.png)


[![PyPi Version](https://badge.fury.io/py/grispy.svg)](https://badge.fury.io/py/grispy)
[![Build Status](https://travis-ci.org/mchalela/GriSPy.svg?branch=master)](https://travis-ci.org/mchalela/GriSPy)
[![Documentation Status](https://readthedocs.org/projects/grispy/badge/?version=latest)](https://grispy.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/mchalela/GriSPy/badge.svg?branch=master)](https://coveralls.io/github/mchalela/GriSPy?branch=master) 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![ascl:1912.013](https://img.shields.io/badge/ascl-1912.013-blue.svg?colorB=262255)](http://ascl.net/1912.013)



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
        
## Citation

If you use *GriSPy* in a scientific publication, we would appreciate
citations to the following paper:

> Chalela, M., Sillero, E., Pereyra, L., García, M. A., Cabral, J. B., Lares, M., & Merchán, M. (2019). 
> GriSPy: A Python package for Fixed-Radius Nearest Neighbors Search. arXiv preprint arXiv:1912.09585.

### Bibtex

```bibtex
@article{
  chalela2019grispy,
  title={GriSPy: A Python package for Fixed-Radius Nearest Neighbors Search},
  author={
    Chalela, Martin and Sillero, Emanuel and Pereyra, 
    Luis and Garc{\'\i}a, Mario Alejandro and Cabral, 
    Juan B and Lares, Marcelo and Merch{\'a}n, Manuel},
  journal={arXiv preprint arXiv:1912.09585},
  year={2019}
}
```

Full-text: https://ui.adsabs.harvard.edu/abs/2019arXiv191209585C/abstract


## Authors

Martin Chalela (E-mail: tinchochalela@gmail.com),
Emanuel Sillero, Luis Pereyra and Alejandro Garcia
