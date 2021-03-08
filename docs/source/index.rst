.. GriSPy documentation master file, created by
   sphinx-quickstart on Sun Dec  1 12:17:55 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


GriSPy documentation
====================

.. only:: html

    .. image:: _static/logo_mid.png
        :align: center
        :scale: 100 %


.. image:: https://badge.fury.io/py/grispy.svg
    :target: https://badge.fury.io/py/grispy
    :alt: PyPi Version

.. image:: https://travis-ci.org/mchalela/grispy.svg?branch=master
    :target: https://travis-ci.org/mchalela/GriSPy
    :alt: Build Status

.. image:: https://img.shields.io/badge/docs-passing-brightgreen.svg
    :target: http://grispy.readthedocs.io
    :alt: ReadTheDocs

.. image:: https://coveralls.io/repos/github/mchalela/GriSPy/badge.svg?branch=master
   :target: https://coveralls.io/github/mchalela/GriSPy?branch=master
   :alt: Coverage

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://tldrlegal.com/license/mit-license
   :alt: License

.. image:: https://img.shields.io/badge/Python-3.6+-blue.svg
   :target: https://www.python.org/downloads/release/python-370/
   :alt: Python 3.6+

.. image:: https://img.shields.io/badge/ascl-1912.013-blue.png?colorB=262255
   :target: http://ascl.net/1912.013
   :alt: ascl:1912.013

.. image:: https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00
   :target: https://github.com/leliel12/diseno_sci_sfw
   :alt: Curso doctoral FAMAF: Diseño de software para cómputo científico

**GriSPy** (Grid Search in Python) is a regular grid search algorithm for quick nearest-neighbor lookup.

This class indexes a set of k-dimensional points in a regular grid providing a fast aproach for nearest neighbors queries. Optional periodic boundary conditions can be provided for each axis individually. Additionally GriSPy provides the posibility of working with individual search radius for each query point in fixed-radius searches and minimum and maximum search radius for shell queries.

| **Authors**
|  Martin Chalela (E-mail: mchalela@unc.edu.ar),
|  Emanuel Sillero, Luis Pereyra, Alejandro Garcia, Juan B. Cabral, Marcelo Lares, Manuel Merchán.


----------------------------------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   licence
   tutorial.ipynb
   api

----------------------------------------------------------------

Repository and Issues
---------------------

https://github.com/mchalela/GriSPy


Citation
--------

If you use *GriSPy* in a scientific publication, we would appreciate citations to the following paper:

  Chalela, M., Sillero, E., Pereyra, L., García, M. A., Cabral, J. B., Lares, M., & Merchán, M. (2020). 
  GriSPy: A Python package for fixed-radius nearest neighbors search. 10.1016/j.ascom.2020.100443.


**Bibtex**

.. code-block:: bibtex

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
   
Full-text: https://arxiv.org/abs/1912.09585


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
