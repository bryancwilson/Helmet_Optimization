import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from utils import *
from utils_3d import *

'''
Rebecca's Approach For Helmet Optimization
- Ran two simuations for both deeper and shallow region. Creates a spherical helmet 
  and measured that maximum sound on the element (for both shallow and deeper). Took the element that had the highest for shallow
  and deep

  time reversal sim (emitting from the focus to the transducer and seeing where the maximum sound amplitude)

  software used for simulation
  - Full wave 2 (Gianmarco's Simulation Tool)

  varied the distance between element and focal point (may need to consider the shape of the elements)

  plotting each voxel (plotting all of the voxels within a point vs just allowing matplotlib make a shape)

  quantify the distance from the element 
  - non-linear 
    squeeze or stretch the distance and optimizing the distance to the skull

    the hole is 1cm distance from element to distance, making 
'''

SIZE_COORD_ARRAY = 128
POP_SIZE = 20

ANGLE_DEGREES = 120
ANGLE_RAD = (ANGLE_DEGREES * np.pi) / 180

# in millimeters
# distance from region of interest to helmet
ELLIPSE_A_DIM = 50
ELLIPSE_B_DIM = 50
ELLIPSE_C_DIM = 100
RADIUS_OF_ROI = 5
ELEMENT_SIZE = 5
HOLE_RADIUS = 10
DEPTH = 75

roi_parameters = {'r_min': 0,
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

opt_parameters = {'phi_lower_bound': int(round((-1*ANGLE_RAD)/2)),
                  'phi_upper_bound': int(round((ANGLE_RAD)/2))}

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

spaced_points = lloyds_rel_3D(iterations, 'sphere', roi_parameters, False)

optimize_angle_3d(shape='ellipsoid',
                  opt_parameters=opt_parameters,
                  roi_parameters=roi_parameters,
                  new_points=spaced_points,
                  depth=DEPTH / RADIUS_OF_ROI,
                  helmet_parameters=helmet_parameters,
                  radius_of_roi=RADIUS_OF_ROI)