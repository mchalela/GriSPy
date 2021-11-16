# GriSPy (Grid Search in Python)

![logo](https://github.com/mchalela/GriSPy/raw/master/res/logo_mid.png)


[![PyPi Version](https://badge.fury.io/py/grispy.svg)](https://badge.fury.io/py/grispy)
[![Build Status](https://github.com/mchalela/GriSPy/actions/workflows/grispy_ci.yml/badge.svg?branch=master)](https://github.com/mchalela/GriSPy/actions/workflows/grispy_ci.yml)
[![Documentation Status](https://readthedocs.org/projects/grispy/badge/?version=latest)](https://grispy.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/mchalela/GriSPy/badge.svg?branch=master)](https://coveralls.io/github/mchalela/GriSPy?branch=master) 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyPI downloads](https://img.shields.io/pypi/dm/grispy)](https://pypistats.org/packages/grispy)

[![ascl:1912.013](https://img.shields.io/badge/ascl-1912.013-blue.svg?colorB=262255)](http://ascl.net/1912.013)
[![arXiv](https://img.shields.io/badge/arXiv-1912.09585-b31b1b.svg)](https://arxiv.org/abs/1912.09585)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)



**GriSPy** is a regular grid search algorithm for quick nearest-neighbor lookup.

This class indexes a set of k-dimensional points in a regular grid providing a fast aproach for nearest neighbors queries. Optional periodic boundary conditions can be provided for each axis individually.

GriSPy has the following queries implemented:
- **bubble_neighbors**: find neighbors within a given radius. A different radius for each centre can be provided.
- **shell_neighbors**: find neighbors within given lower and upper radius. Different lower and upper radius can be provided for each centre.
- **nearest_neighbors**: find the nth nearest neighbors for each centre.

## Usage example

Let's create a 2D random distribution of points as an example:

```python
import numpy as np
import grispy as gsp

data = np.random.uniform(size=(1000, 2))
grid = gsp.GriSPy(data)
```

The `grid` object now has all the data points indexed in a grid. Now let's search for neighbors around new points:
```python
centres = np.random.uniform(size=(10, 2))
dist, ind = grid.bubble_neighbors(centres, distance_upper_bound=0.1)
```

And that's it! The `dist` and `ind` lists contain the distances and indices to `data` neighbors within a 0.1 search radius.

--------------------------------

## Requirements

You will need Python 3.6 or later to run GriSPy.


## Standard Installation

GriSPy is available at [PyPI](https://pypi.org/project/grispy/). You can install it via the pip command

```bash
$ pip install grispy
```

## Development Install

Clone this repo and then inside the local directory execute

```bash
$ pip install -e .
```

## Citation

If you use *GriSPy* in a scientific publication, we would appreciate citations to the following paper:

> Chalela, M., Sillero, E., Pereyra, L., García, M. A., Cabral, J. B., Lares, M., & Merchán, M. (2020). 
> GriSPy: A Python package for fixed-radius nearest neighbors search. 10.1016/j.ascom.2020.100443.

### Bibtex

```bibtex
@ARTICLE{Chalela2021,
       author = {{Chalela}, M. and {Sillero}, E. and {Pereyra}, L. and {Garcia}, M.~A. and {Cabral}, J.~B. and {Lares}, M. and {Merch{\'a}n}, M.},
        title = "{GriSPy: A Python package for fixed-radius nearest neighbors search}",
      journal = {Astronomy and Computing},
     keywords = {Data mining, Nearest-neighbor search, Methods, Data analysis, Astroinformatics, Python package},
         year = 2021,
        month = jan,
       volume = {34},
          eid = {100443},
        pages = {100443},
          doi = {10.1016/j.ascom.2020.100443},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021A&C....3400443C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

Full-text: https://arxiv.org/abs/1912.09585


## Authors

Martin Chalela (E-mail: mchalela@unc.edu.ar),
Emanuel Sillero, Luis Pereyra, Alejandro Garcia, Juan B. Cabral, Marcelo Lares, Manuel Merchán.
