import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from utils import *

SIZE_COORD_ARRAY = 128

# generate random seed of polar coordinates
r_bounds = [0, .90]
theta_bounds = [0, 360]

r_s = np.random.uniform(r_bounds[0], r_bounds[1], (SIZE_COORD_ARRAY, 1))
theta_s = np.random.uniform(theta_bounds[0], theta_bounds[1], (SIZE_COORD_ARRAY, 1))

cart_x, cart_y = pol2cart_array(r_s, theta_s)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(cart_x, cart_y)

# plot unit circle
unit_circle = patches.Circle((0, 0), radius=1, fill=False)
ax.add_patch(unit_circle)

plt.show()

# generate corresponding vectors
ones = np.ones((SIZE_COORD_ARRAY, 1))
r_diff = np.subtract(ones, r_s)

cart_u, cart_v = pol2cart_array(r_diff, theta_s)

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.scatter(cart_x, cart_y)

# plot unit circle
unit_circle = patches.Circle((0, 0), radius=1, fill=False)
ax1.add_patch(unit_circle)

ax1.quiver(cart_x, cart_y, cart_u, cart_v, angles='xy', scale_units='xy', scale=1, color='r')
plt.show()

# optimize spread of points

new_r = []
for r, t in zip(r_s, theta_s):
    perturbation = np.random.uniform(0, 0.1)

    while perturbation + r >= 1:
        perturbation = np.random.uniform(0, 0.1)

    new_r.append(perturbation + r)


# calculate distance between points

