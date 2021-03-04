#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   GriSPy Project (https://github.com/mchalela/GriSPy).
# Copyright (c) 2019, Martin Chalela
# License: MIT
#   Full Text: https://github.com/mchalela/GriSPy/blob/master/LICENSE


import numpy as np
import matplotlib.pyplot as plt
from grispy import GriSPy


# Example 1. 2D Uniform Distribution ------------------------------------------
# This example generates a 2D random uniform distribution.
# Periodic conditions on y-axis, or axis=1.
# We search for neighbors within a given radius and n-nearest neighbors.

# Create random points and centres
Npoints = 10 ** 4
Ncentres = 2
dim = 2
Lbox = 100.0

np.random.seed(2)
data = np.random.uniform(0, Lbox, size=(Npoints, dim))
centres = np.random.uniform(0, Lbox, size=(Ncentres, dim))

# Grispy params
upper_radii = 15.0
lower_radii = 10.0
n_nearest = 100
periodic = {0: (0, Lbox), 1: (0, Lbox)}

# Build the grid with the data
gsp = GriSPy(data, periodic=periodic)

# Query for neighbors within upper_radii
bubble_dist, bubble_ind = gsp.bubble_neighbors(
    centres, distance_upper_bound=upper_radii
)

# Query for neighbors in a shell within lower_radii and upper_radii
shell_dist, shell_ind = gsp.shell_neighbors(
    centres, distance_lower_bound=lower_radii, distance_upper_bound=upper_radii
)

# Query for nth nearest neighbors
near_dist, near_ind = gsp.nearest_neighbors(centres, n=n_nearest)


# Plot results
plt.figure(4, figsize=(10, 3.2))

plt.subplot(1, 3, 1, aspect="equal")
plt.title("Bubble query")
plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=3)
for ind in bubble_ind:
    plt.scatter(data[ind, 0], data[ind, 1], c="C3", marker="o", s=5)
plt.plot(centres[:, 0], centres[:, 1], "ro", ms=10)

plt.subplot(1, 3, 2, aspect="equal")
plt.title("Shell query")
plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=2)
for ind in shell_ind:
    plt.scatter(data[ind, 0], data[ind, 1], c="C2", marker="o", s=5)
plt.plot(centres[:, 0], centres[:, 1], "ro", ms=10)

plt.subplot(1, 3, 3, aspect="equal")
plt.title("n-Nearest query")
plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=2)
for ind in near_ind:
    plt.scatter(data[ind, 0], data[ind, 1], c="C0", marker="o", s=5)
plt.plot(centres[:, 0], centres[:, 1], "ro", ms=10)

plt.tight_layout()
plt.show()
