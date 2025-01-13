import numpy as np
from itertools import combinations
from math import dist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, ArtistAnimation
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting_func import *
from conv_func import *

# generate random polygon
def generate_random_polygon_3d(num_points, min_x, max_x, min_y, max_y):
    """Generates a random polygon with the given number of points."""

    points = []
    for _ in range(num_points):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        points.append(np.array((x, y)))

    points = np.array(points)
    # Ensure the polygon is valid by checking for self-intersection
    while True:
        polygon = Polygon(points)
        if polygon.is_valid:
            return points[:, 0], points[:, 1]
        else:
            np.random.shuffle(points)

# generate cartesian points
def point_generation_3D(size, shape, parameters, plot=False):
    # generate random seed of polar coordinates

    if shape == 'arbitrary':
        # fig, ax = plt.subplots()

        # # vertices
        # x_v = parameters['x']
        # y_v = parameters['y']
        # z_v = parameters['z']

        # vertices = []
        # for x_, y_, z_ in zip(x_v, y_v, z_v):
        #     vertices.append((x_, y_, z_))

        # # generate polygon
        # pg = Poly3DCollection([vertices])

        # # generate random points in 3D volume
        # cart_x = []
        # cart_y = []
        # cart_z = []
        # for _ in range(size):
        #     x = np.random.uniform(np.min(x_v), np.max(x_v))
        #     y = np.random.uniform(np.min(y_v), np.max(y_v))
        #     z = np.random.uniform(np.min(z_v), np.max(z_v))    

        #     cart_x.append(x)
        #     cart_y.append(y)
        #     cart_z.append(z)


        # Create a 3D array of data
        data = np.random.rand(10, 10, 10)

        # Create a figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the voxels
        ax.voxels(data)

        plt.show()

        # generate point in polygon
        # minx, miny, maxx, maxy = pg.bounds
        # cart_x = []
        # cart_y = []
        # while len(cart_x) < size:
        #     x = np.random.uniform(minx, maxx)
        #     y = np.random.uniform(miny, maxy)
        #     point = Point(x, y)
        #     if pg.contains(point):
        #         cart_x.append(x)
        #         cart_y.append(y)


        # if plot:
        #     polygon = patches.Polygon(list(zip(x_v, y_v)), closed=True, alpha=.1)

        #     ax.add_patch(polygon)
        #     ax.scatter(np.array(cart_x), np.array(cart_y), marker='o', color='red', alpha=0.5)
        #     ax.set_xlim([np.min(x_v), np.max(x_v)])
        #     ax.set_ylim([np.min(y_v), np.max(y_v)])

        #     plt.show()

    elif shape == 'sphere' or shape=='semi_sphere':
        r_bounds = [parameters['r_min'], parameters['r_max']]
        theta_bounds = [parameters['theta_min'], parameters['theta_max']]
        phi_bounds = [parameters['phi_min'], parameters['phi_max']]

        r_s = np.random.uniform(r_bounds[0], r_bounds[1], (size, ))
        theta_s = np.random.uniform(theta_bounds[0], theta_bounds[1], (size, ))
        phi_s = np.random.uniform(phi_bounds[0], phi_bounds[1], (size, ))

        cart_x, cart_y, cart_z = pol2cart_array_3d(r_s, theta_s, phi_s)

        if plot:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(projection='3d')

            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = np.cos(u)*np.sin(v)
            y = np.sin(u)*np.sin(v)
            z = np.cos(v)
            ax.plot_wireframe(x, y, z, color="b")

            ax.scatter(np.array(cart_x), np.array(cart_y), np.array(cart_z), marker='o', color='red', alpha=0.5)
            
            plt.show()

        pg = None

    return cart_x, cart_y, cart_z, pg

def lloyds_rel_3D(iterations, shape, parameters, plot):

    # generate points

    x, y, z, pg = point_generation_3D(128, 
                        shape, 
                        parameters, 
                        plot=True)
    
    coords = []
    for i in range(len(x)):
        coords.append([x[i], y[i], z[i]])

    initial_points = np.array(coords)
    points = np.array(coords)

    center = [0, 0, 0]
    radius = .9
    movements = []

    for e in range(iterations):

        # Compute Voronoi diagram
        vor = Voronoi(points)

        # Plot the Voronoi diagram using scipy's built-in function
        # voronoi_plot_2d(vor, show_vertices=False)  # hide vertices for cleaner visualization
        
        # plt.show()

        new_points = []
        
        # For each point, find the centroid of its Voronoi cell
        for i in range(len(points)):
            # Get the vertices of the Voronoi cell
            region = vor.regions[vor.point_region[i]]
            
            if -1 in region:  # Ignore infinite regions
                new_points.append(points[i])
                
            else:
                # Extract the polygon that represents the Voronoi cell
                vertices = np.array([vor.vertices[i] for i in region])
                
                # Calculate the centroid of the Voronoi cell
                centroid = np.mean(vertices, axis=0)
                
                if shape=='sphere':
                    # Clip centroid to make sure it's inside the region
                    dist = np.linalg.norm(centroid - center)
                    if dist > radius:
                        # Project the centroid back onto the circle boundary
                        centroid = center + (centroid - center) * radius / dist

                elif shape=='semi_sphere':

                    # Clip centroid to make sure it's inside the region
                    dist = np.linalg.norm(centroid - center)
                    if dist > radius:
                        # Project the centroid back onto the circle boundary
                        centroid = center + (centroid - center) * radius / dist 


                    if centroid[2] < 0:
                        centroid[2] = 0                  

                elif shape=='arbitrary':

                    point = Point(centroid[0], centroid[1])
                    if not pg.contains(point):
                        centroid = nearest_points(pg, point)[0]

                        centroid = [centroid.x, centroid.y]

                new_points.append(list(centroid))
        
        new_points = np.array(new_points, dtype=np.float64)
        # Update points to the new centroids
        average_mov =  np.mean(np.linalg.norm(new_points - points, axis=1))
        movements.append(average_mov)
        print("EPOCH: ", e, "Average Displacement: ", average_mov)

        points = np.array(new_points)

    if plot:
        plot_3d_final(final_points=new_points)

        plot_3d(initial_points=initial_points,
            final_points=new_points, 
            iterations=iterations, 
            movements=movements, 
            shape=shape, 
            parameters=parameters)
        
    return points
    
def optimize_angle_3d(shape, opt_parameters, roi_parameters, new_points, depth, helmet_parameters, radius_of_roi):

    plot_3d_final(new_points)

    focal_points = new_points
    new_points = list(new_points) 

    center = helmet_parameters['center']

    point_angle_dict = {}
    used_tps = []
    while new_points != []:

        # remove focal point from stack
        np_ = new_points.pop(0)

        best_obj_val = float('inf')

        x_s = []
        y_s = []
        z_s = []

        # iterate through points on helmet surface
        # steps = (helmet_parameters['circumference'] / 2) / helmet_parameters['ele_size'] - 12
        steps = 20
        neighbors = 4
        for t_ in np.linspace(0, 2*np.pi, int(round(steps))):
            for p_ in np.linspace(opt_parameters['phi_lower_bound'], opt_parameters['phi_upper_bound'], int(round(steps))):

                # ignore angles that overlap with helmet opening
                if p_ > -1*np.arctan2(helmet_parameters['hole_size'], helmet_parameters['c']) and p_ < np.arctan2(helmet_parameters['hole_size'], helmet_parameters['c']):
                    continue

                # ignore used angles
                if (t_, p_) in used_tps:
                    continue

                if shape=='ellipsoid':
                    r = tp2rad_ellipsoid(t_,
                                         p_,
                                         helmet_parameters['a'],
                                         helmet_parameters['b'],
                                         helmet_parameters['c'])
                    
                    x_, y_, z_ = pol2cart_3d(r, t_, p_)
                    
                    x_s.append(x_)
                    y_s.append(y_)
                    z_s.append(z_)

                # else:
                #     x_, y_ = pol2cart(helmet_parameters['radius'], t_)
                #     y_+=center[1] # offset due to radius

                dist = np.linalg.norm(np.array([x_, y_, z_]) - np_) # calculate distance between points

                # theta calculation
                angle_pp_t = np.arctan2(y_ - np_[1], x_ - np_[0]) # calculate angle between focal point and element being auditioned
                angle_pc_t = np.arctan2(y_ - center[1], x_ - center[0]) # angle between focal point and the center

                # phi calculation
                angle_pp_p = np.arctan2(z_ - np_[2], x_ - np_[0]) # calculate angle between focal point and element being auditioned
                angle_pc_p = np.arctan2(z_ - center[2], x_ - center[0]) # angle between focal point and the center

                obj_val = 0*np.abs(depth - dist) + 1*np.abs(angle_pp_t - angle_pc_t) + 1*np.abs(angle_pp_p - angle_pc_p)# evaluate objective function

                # assign focal point to element according to objective function
                if obj_val < best_obj_val:
                    point_angle_dict[tuple(np_)] = (x_, y_, z_, t_, p_, dist, np.abs(angle_pp_t - angle_pc_t), np.abs(angle_pp_p - angle_pc_p), obj_val)
                    best_obj_val = obj_val

        # check if new assignment is better than old assignment
        for key, val in point_angle_dict.items():
            if [val[0], val[1], val[2]] == [point_angle_dict[tuple(np_)][0], point_angle_dict[tuple(np_)][1], point_angle_dict[tuple(np_)][2]] and point_angle_dict[key] != (0, 0, 0, 0, 0, 0, 0, 0, 0) and best_obj_val < val[8]:
                new_points.append(np.array(key))
                point_angle_dict[key] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
            # elif val[0] == point_angle_dict[tuple(np_)][0] and val[1] == point_angle_dict[tuple(np_)][1] and val[2] == point_angle_dict[tuple(np_)][2] and best_obj_val >= val[7]:
            #     new_points.append(np.array(tuple(np_)))
            #     point_angle_dict[tuple(np_)] = (0, 0, 0, 0, 0, 0, 0, 0)

        used_tps.append((point_angle_dict[tuple(np_)][3], point_angle_dict[tuple(np_)][4]))


    values = np.array(list(point_angle_dict.values()))

    # check for duplicate element positions in list
    neigh = NearestNeighbors(n_neighbors=neighbors)
    element_locations = np.array(values[:, :3])
    unique_coords = []
    for i in range(len(element_locations)):
        if list(element_locations[i]) not in unique_coords:
            unique_coords.append(list(element_locations[i]))
        else:
            print("REPEAT POINT")
        element_locations[i] = list(element_locations[i])

    neigh.fit(element_locations)

    # calculate the min distances 
    element_locations = np.reshape(values[:, :3], (128, 3))
    max_dists = []
    for v in element_locations:
        dists, nbrs = neigh.kneighbors(np.array([v]))
        max_dist = np.max(dists[0][1:])
        max_dists.append(max_dist)
    
    # plot resulting element positions
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')

    # plot sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    ## plot skeleton of helmet
    # ax1.scatter(x_s, y_s, z_s)

    ax1.scatter(values[:, 0], values[:, 1], values[:, 2], s=280, marker='h') # plot element positions
    ax1.scatter(focal_points[:, 0], focal_points[:, 1], focal_points[:, 2]) # plot focal point positions
    ax1.set_xlim(-20, 20)
    ax1.set_ylim(-20, 20)
    ax1.set_zlim(-20, 20)

    # connecting focal points to element locations
    for i in range(len(focal_points)):
        ax1.plot([focal_points[i][0], values[i][0]], [focal_points[i][1], values[i][1]], [focal_points[i][2], values[i][2]], 'k-')
    
    ax1.plot_wireframe(x, y, z, color="k")

    plt.show()

    # Plotting Histograms
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    ax1.hist(values[:, 4]*radius_of_roi)
    ax1.set_xlabel('Distance From Element to Focal Point (mm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distance Histogram')

    ax2.hist(values[:, 5])
    ax2.set_xlabel('Angle of Element Relative To Helmet Surface (radians)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Angle Histogram (THETA)')

    ax3.hist(values[:, 6])
    ax3.set_xlabel('Angle of Element Relative To Helmet Surface (radians)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Angle Histogram (PHI)')

    ax4.hist(max_dists)
    ax4.set_xlabel('Distances')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Max Distance Per Element Histogram')
    plt.show()