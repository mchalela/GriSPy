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
    :alt: ReadTheDocs.org

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://tldrlegal.com/license/mit-license
   :alt: License

.. image:: https://img.shields.io/badge/Python-3.6+-blue.svg
   :target: https://www.python.org/downloads/release/python-370/
   :alt: Python 3.6+



**GriSPy** (Grid Search in Python) is a regular grid search algorithm for quick nearest-neighbor lookup.

This class indexes a set of k-dimensional points in a regular grid providing a fast aproach for nearest neighbors queries. Optional periodic boundary conditions can be provided for each axis individually. Additionally GriSPy provides the posibility of working with individual search radius for each query point in fixed-radius searches and minimum and maximum search radius for shell queries.

| **Authors**
| Martin Chalela (E-mail: tinchochalela@gmail.com),
| Emanuel Sillero, Luis Pereyra and Alejandro Garcia


Repository and Issues
---------------------

https://github.com/mchalela/GriSPy

----------------------------------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   licence
   tutorial.ipynb
   api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
