# import time
import numpy as np
import matplotlib.pyplot as plt
from grispy3 import GriSPy

plt.ion()


# Example 1. 2D Uniform Distribution ------------------------------------------
# This example generates a 2D random uniform distribution.
# Periodic conditions on y-axis, or axis=1.
# We search for neighbors within a given radius and n-nearest neighbors.

# Create random points and centres
Npoints = 10 ** 4
Ncentres = 5
dim = 2
Lbox = 100.0

np.random.seed(0)
data = np.random.uniform(-Lbox / 2, Lbox / 2, size=(Npoints, dim))
centres = np.random.uniform(-Lbox / 2, Lbox / 2, size=(Ncentres, dim))

# Grispy params
upper_radii = 10.0
lower_radii = 7.0
n_nearest = 100
periodic = {1: (-Lbox / 2, Lbox / 2)}

# Build the grid with the data
gsp = GriSPy(data, periodic=periodic)
# La periodicidad se puede mandar cuando se crea la grid como en la linea
# anterior, o se puede setear despues de crear la grid:
# gsp = GriSPy(data)
# gsp.set_periodicity(periodic)

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


# Plot bubble results
plt.figure()
plt.title("Bubble query")
plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=3)
for i in range(Ncentres):
    ind_i = bubble_ind[i]
    plt.scatter(data[ind_i, 0], data[ind_i, 1], c="C3", marker="o", s=5)

# Plot shell results
plt.figure()
plt.title("Shell query")
plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=2)
for i in range(Ncentres):
    ind_i = shell_ind[i]
    plt.scatter(data[ind_i, 0], data[ind_i, 1], c="C2", marker="o", s=5)

# Plot nearest results
plt.figure()
plt.title("n-Nearest query")
plt.scatter(data[:, 0], data[:, 1], c="k", marker=".", s=2)
for i in range(Ncentres):
    ind_i = near_ind[i]
    plt.scatter(data[ind_i, 0], data[ind_i, 1], c="C0", marker="o", s=5)
