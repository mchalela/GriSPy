#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


import importlib
import numpy as np
import matplotlib.pyplot as plt
import grispy ; importlib.reload(grispy)
GriSPy = grispy.GriSPy
#from grispy import GriSPy

plt.ion()


# Example 1. 2D Uniform Distribution ------------------------------------------
# This example generates a 2D random uniform distribution.
# Periodic conditions on y-axis, or axis=1.
# We search for neighbors within a given radius and n-nearest neighbors.

# Create random points and centres
Npoints = 10 ** 5
Ncentres = 1
dim = 2
Lbox = 10.0

np.random.seed(0)
data = np.random.uniform(-Lbox / 2, Lbox / 2, size=(Npoints, dim))
#centres = np.random.uniform(-Lbox / 2, Lbox / 2, size=(Ncentres, dim))
centres = np.array([[0.,0.]])

# Grispy params
upper_radii = 9.0
lower_radii = 8.9
n_nearest = 100
periodic = {0: (-Lbox / 2, Lbox / 2), 1: (-Lbox / 2, Lbox / 2)}

# Build the grid with the data
gsp = GriSPy(data, periodic=periodic)
# La periodicidad se puede mandar cuando se crea la grid como en la linea
# anterior, o se puede setear despues de crear la grid:
# gsp = GriSPy(data)
# gsp.set_periodicity(periodic)

# Query for neighbors within upper_radii
#bubble_dist, bubble_ind = gsp.bubble_neighbors(
#    centres, distance_upper_bound=upper_radii
#)

# Query for neighbors in a shell within lower_radii and upper_radii
shell_dist, shell_ind = gsp.shell_neighbors(
    centres, distance_lower_bound=lower_radii, distance_upper_bound=upper_radii
)

# Query for nth nearest neighbors
#near_dist, near_ind = gsp.nearest_neighbors(centres, n=n_nearest)

'''
# Plot bubble results
plt.figure()
plt.title("Bubble query")
plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=3)
for i in range(Ncentres):
    ind_i = bubble_ind[i]
    plt.scatter(data[ind_i, 0], data[ind_i, 1], c="C3", marker="o", s=5)
'''

# Plot shell results
plt.figure(1)
plt.cla()
plt.title("Shell query")
plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=2)
for i in range(Ncentres):
    ind_i = shell_ind[i]
    plt.scatter(data[ind_i, 0], data[ind_i, 1], c="C2", marker="o", s=5)
plt.plot(centres[0,0],centres[0,1],'ro',ms=10)

'''
# Plot nearest results
plt.figure()
plt.title("n-Nearest query")
plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=2)
for i in range(Ncentres):
    ind_i = near_ind[i]
    plt.scatter(data[ind_i, 0], data[ind_i, 1], c="C0", marker="o", s=5)
'''