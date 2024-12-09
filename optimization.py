import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from utils import *
from utils_3d import *


SIZE_COORD_ARRAY = 128
POP_SIZE = 20

# ======================================= 2D ===========================================#
# circle parameters
# shape = 'circle'
# parameters = {'r_min': 0,
#               'r_max': 1,
#               'theta_min': 0,
#               'theta_max': 360}

# # polygon parameters
# shape = 'arbitrary'
# x, y = generate_random_polygon(8, -5, 5, -5, 5)
# # parameters = {'x': [0, 1, 2, 1],
# #               'y': [0, 1, 0, -1]}
# parameters = {'x': x,
#               'y': y}

# lloyds_rel(200, shape, parameters)

# ======================================= 3D ===========================================#

parameters = {'r_min': 0,
              'r_max': .9,
              'theta_min': 0,
              'theta_max': 2*np.pi,
              'phi_min': -1*(np.pi) / 2,
              'phi_max': np.pi / 2}

# lloyds_rel(128, 'semi_circle', parameters)
lloyds_rel_3D(128, 'semi_sphere', parameters)

