import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from utils import *



SIZE_COORD_ARRAY = 128
POP_SIZE = 10

# initialize optimization parameters
hp = {'num_neighbors': 5,
      'pop_multiplier': 10,
      'k': 5}

evolutionary_algorithm(100, hp)

# # initialize optimization parameters
# hp = {'num_neighbors': 3,
#       'c1': 1,
#       'c2': 1}

# iterations = 300
# particle_swarm(cart_x, cart_y, iterations, hp)


# generate sunflower
sun_flower(128, 1)
