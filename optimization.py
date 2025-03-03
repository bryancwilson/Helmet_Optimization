import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from utils import *
from utils_3d import *
import pandas as pd

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

'''

'''

SIZE_COORD_ARRAY = 128

POP_SIZE = 20

ANGLE_DEGREES = 140
ANGLE_RAD = (ANGLE_DEGREES * np.pi) / 180

# in millimeters
# distance from region of interest to helmet
ELLIPSE_A_DIM = 50
ELLIPSE_B_DIM = 50
ELLIPSE_C_DIM = 56
RADIUS_OF_ROI = 5
ELEMENT_SIZE = 5
HOLE_RADIUS = 100
DEPTH = 60

# tangent ogive parameters
# L=56
# R=50

roi_parameters = {'r_min': 0,
              'r_max': .9,
              'theta_min': 0,
              'theta_max': 2*np.pi,
              'phi_min': 0,
              'phi_max': 2*np.pi,
              'num_fp': 128}

helmet_parameters = {'shape': 'semi_ellipsoid',
                     'base_angle': ANGLE_RAD,
                     'radius': 1,
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

# output_params = {'base_angle': [ANGLE_DEGREES],
#                 'hole_radius': [HOLE_RADIUS],
#                 'element_size': [ELEMENT_SIZE],
#                 'a': [ELLIPSE_A_DIM],
#                 'b': [ELLIPSE_B_DIM],
#                 'c': [ELLIPSE_C_DIM]}
# output_coords = {'x': list(fib_points[:, 0]),
#                  'y': list(fib_points[:, 1]),
#                  'z': list(fib_points[:, 2])}

# output_coords_df = pd.DataFrame.from_dict(output_coords)
# output_coords_df.to_csv('Helmet_Coords_SemiEllipsoid.csv')
# output_params_df = pd.DataFrame.from_dict(output_params)
# output_params_df.to_csv('Helmet_Params_SemiEllipsoid.csv')

# # plot francisco's helmet =====================================================================================
helmet_points = francisco_bl()

# plot surface of the region of interest
surf_points = surface_points()

# flag the rows of elements
flags = [0] * len(helmet_points)
for i in range(len(helmet_points)):
  if i >= 113 and i < 128:
    flags[i] = 5
  elif i >= 93 and i < 113:
    flags[i] = 4
  elif i >= 67 and i < 93:
    flags[i] = 3
  elif i >= 36 and i < 67:
    flags[i] = 2
  elif i < 36:
    flags[i] = 1

element_focus_points = calculate_normal_vectors(helmet_points=helmet_points,
                                                flags=flags,
                                                vector_sizes= 5,
                                                plot=False)

pd.DataFrame(element_focus_points).to_csv('Element_Focal_Coordinates.csv')
# data = []
# for efp in element_focus_points:
#   diameter = 1
#   height = 7

#   x, y, z = parameterize_cylinder(vector=efp,
#                                   diameter=diameter,
#                                   height=height)
#   data.append(go.Surface(x=x, y=y, z=z, showscale=False))

# plot_cylinder(data)

# initialized set of evenly spaced points in the volume
spaced_points = lloyds_rel_3D(iterations=iterations, 
                              shape='sphere', 
                              parameters=roi_parameters, 
                              plot=False)

# plot element focus points scatter
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
ax1.scatter(element_focus_points[:, 0], element_focus_points[:, 1], element_focus_points[:, 2], color="b")
ax1.scatter(helmet_points[:, 0], helmet_points[:, 1], helmet_points[:, 2], color="k")
ax1.set_title("Element Focal Positions")

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.set_zlim(-10, 10)
ax1.plot_wireframe(x*2.5, y*2.5, z*2.5, color="k")

ax1.scatter(spaced_points[:, 0], spaced_points[:, 1], spaced_points[:, 2], color='r')

plt.show()

# parameters
population_size = 10
# initialize element positions candidates
helmet_params = []
top_cands_ele_pos = []
top_cands_ele_foc_pos = []
neighbors = []

# initialized set of evenly spaced points in the volume
spaced_points = lloyds_rel_3D(iterations=iterations, 
                              shape='sphere', 
                              parameters=roi_parameters, 
                              plot=False)

while len(top_cands_ele_pos) < population_size:

  l = random.uniform(-13500, -13200)
  r = random.uniform(-.01, -.005)


  print("L: ", l, "R: ", r)
  helmet_points = helmet_element_cands_3d(L=l,
                                  R=r,
                                  helmet_parameters=helmet_parameters,
                                  plot=False)
  top_cands_ele_pos.append(helmet_points)

  # inside_sphere = False
  # while not inside_sphere:
  # element focus points0
  vs = np.random.uniform(low=.1, high=20, size=128)
  element_focus_points = calculate_normal_vectors(helmet_points=helmet_points,
                                                  vector_sizes= vs,
                                                  plot=False)
  inside_sphere = in_sphere(element_focal_points=element_focus_points)

  helmet_params.append(tuple([l, r, tuple(vs)]))
  top_cands_ele_foc_pos.append(element_focus_points)

  ns = neighbor_distances(element_focal_points=helmet_points,
                     spaced_focal_points=spaced_points)
  
neighbors = ns

# optimization run
stds = []
means = []
for _ in range(10):

  helmet_param_mse_s = {}
  for i in range(len(helmet_params)):
  # subject helmet parameters to fitness function

    helmet_param_mse_s[tuple(helmet_params[i])] = error_calculation(top_cands_ele_foc_pos[i],
                                                                    helmet_points,
                                                                    surf_points,
                                                                    spaced_points)
    
  sorted_ = sorted(helmet_param_mse_s.items(), key=lambda kv: kv[1])[:5]

  sifted_sorted = []
  for pair in sorted_:
    sifted_sorted.append(pair[0])

  children_cands = crossover(sifted_sorted, 10)

  helmet_params, top_cands_ele_pos, top_cands_ele_foc_pos = build_helmet(children_cands, helmet_parameters)

  # save metrics

  means.append(np.mean(list(helmet_param_mse_s.values())))
  stds.append(np.std(list(helmet_param_mse_s.values())))



# plot converged geometry
print("Param Converged To: ", helmet_params[0])
helmet_points = helmet_element_cands_3d(L=helmet_params[0][0],
                                  R=helmet_params[0][1],
                                  helmet_parameters=helmet_parameters,
                                  plot=False)
element_focus_points = calculate_normal_vectors(helmet_points=helmet_points,
                                                vector_sizes= helmet_params[0][2],
                                                plot=False)

# plot element focus points scatter
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
ax1.scatter(element_focus_points[:, 0], element_focus_points[:, 1], element_focus_points[:, 2], color="b")
ax1.scatter(helmet_points[:, 0], helmet_points[:, 1], helmet_points[:, 2], color="k")
ax1.set_title("Element Focal Positions")

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.set_zlim(-10, 10)
ax1.plot_wireframe(x*2.5, y*2.5, z*2.5, color="k")

ax1.scatter(spaced_points[:, 0], spaced_points[:, 1], spaced_points[:, 2], color='r')

plt.figure()

fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(np.linspace(0, len(means), len(means)), means)
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Mean MSE')
ax1.set_title('EVOL ALG Error (mean)')

ax2.plot(np.linspace(0, len(stds), len(stds)), stds)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('STD DEV MSE')
ax2.set_title('EVOL ALG Error (std dev)')

# check for duplicate element positions in list
neigh = NearestNeighbors(n_neighbors=3)
helmet_points = np.array(helmet_points)
neigh.fit(helmet_points)
distances, _ = neigh.kneighbors(helmet_points)
distances = np.reshape(distances[:, 1:], (128*(3 - 1)))
ax3.hist(distances)
ax3.set_xlabel('Distances (cm)')
ax3.set_ylabel('Frequency')
ax3.set_title('Nearest Element-to-Element Distance Histogram')

plt.show()

# optimize_angle_3d_v2(shape='ellipsoid',
#                   opt_parameters=opt_parameters,
#                   roi_parameters=roi_parameters,
#                   new_points=spaced_points,
#                   depth=DEPTH / RADIUS_OF_ROI,
#                   helmet_parameters=helmet_parameters,
#                   radius_of_roi=RADIUS_OF_ROI)