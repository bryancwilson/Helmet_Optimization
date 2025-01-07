import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from utils import *
from utils_3d import *


SIZE_COORD_ARRAY = 128
POP_SIZE = 20

# in millimeters
# distance from region of interest to helmet
ELLIPSE_A_DIM = 100
ELLIPSE_B_DIM = 100
ELLIPSE_C_DIM = 50
RADIUS_OF_ROI = 5
ELEMENT_SIZE = 5
HOLE_RADIUS = 10
DEPTH = 75

parameters = {'r_min': 0,
              'r_max': .9,
              'theta_min': 0,
              'theta_max': 2*np.pi,
              'phi_min': 0,
              'phi_max': 2*np.pi}

helmet_parameters = {'radius': 1,
                     'center': (0, 0, 0),
                     'a': (ELLIPSE_A_DIM / RADIUS_OF_ROI) + 1,
                     'b': (ELLIPSE_B_DIM / RADIUS_OF_ROI) + 1,
                     'c': (ELLIPSE_C_DIM / RADIUS_OF_ROI) + 1,
                     'circumference': (4*(np.pi / 2) ** (((ELLIPSE_A_DIM / RADIUS_OF_ROI) + 1) / ((ELLIPSE_B_DIM / RADIUS_OF_ROI) + 1)))*((ELLIPSE_B_DIM / RADIUS_OF_ROI) + 1),
                     'ele_size': ELEMENT_SIZE / RADIUS_OF_ROI,
                     'hole_size': HOLE_RADIUS / RADIUS_OF_ROI}

iterations = 100

# ======================================= 2D ===========================================#
# circle parameters
# shape = 'circle'

# spaced_points = lloyds_rel(iterations, 'circle', parameters, False)

# optimize_angle(shape='ellipse',
#                parameters=parameters,
#                new_points=spaced_points,
#                depth = 15,
#                helmet_parameters=helmet_parameters)

# # polygon parameters
# shape = 'arbitrary'
# x, y = generate_random_polygon(8, -5, 5, -5, 5)
# # parameters = {'x': [0, 1, 2, 1],
# #               'y': [0, 1, 0, -1]}
# parameters = {'x': x,
#               'y': y}
# lloyds_rel(200, shape, parameters)

# ======================================= 3D ===========================================#

spaced_points = lloyds_rel_3D(iterations, 'sphere', parameters, False)

optimize_angle_3d(shape='ellipsoid',
                  parameters=parameters,
                  new_points=spaced_points,
                  depth=DEPTH / RADIUS_OF_ROI,
                  helmet_parameters=helmet_parameters)