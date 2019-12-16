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


**GriSPy** (Grid Search in Python) is a regular grid search algorithm for quick nearest-neighbor lookup.

This class indexes a set of k-dimensional points in a regular grid providing a fast aproach for nearest neighbors queries. Optional periodic boundary conditions can be provided for each axis individually. Additionally GriSPy provides the posibility of working with individual search radius for each query point in fixed-radius searches and minimum and maximum search radius for shell queries.

| **Authors**
| Martin Chalela (E-mail: tinchochalela@gmail.com),
| Emanuel Sillero, Luis Pereyra and Alejandro Garcia

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
