import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from utils import *



SIZE_COORD_ARRAY = 128
POP_SIZE = 20

# circle parameters
# shape = 'circle'
# parameters = {'r_min': 0,
#               'r_max': 1,
#               'theta_min': 0,
#               'theta_max': 360}

# polygon parameters
shape = 'arbitrary'
x, y = generate_random_polygon(8, -5, 5, -5, 5)
# parameters = {'x': [0, 1, 2, 1],
#               'y': [0, 1, 0, -1]}
parameters = {'x': x,
              'y': y}

lloyds_rel(200, shape, parameters)

# initialize optimization parameters
# hp = {'num_neighbors': 10,
#       'pop_multiplier': 10,
#       'k': 5}

# evolutionary_algorithm(20, hp)

# initialize optimization parameters
# hp = {'num_neighbors': 3,
#       'c1': 1,
#       'c2': 1}

# iterations = 100
# particle_swarm(iterations, hp)


# generate sunflower
# sun_flower(128, 1)