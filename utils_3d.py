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


def pol2cart_3d(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)   
    z = r * np.cos(phi)

    return x, y, z

def cart2pol_3d(x, y, z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / rho)

    return rho, theta, phi

def pol2cart_array_3d(r_s, theta_s, phi_s):
    cart_x = []
    cart_y = []
    cart_z = []

    for r, t, p in zip(r_s, theta_s, phi_s):
        x, y, z = pol2cart_3d(r, t, p)
        cart_x.append(x)
        cart_y.append(y)
        cart_z.append(z)
    
    return cart_x, cart_y, cart_z

def cart2pol_array_3d():
    pass

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

# create base plot
def polt_3d_final(final_points):
    fig = plt.figure()
    ax2 = fig.add_subplot(projection='3d')

    # plot sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    
    # plot vectors
    cs2 = final_points
    direct = []
    for c_ in cs2:
        rho, theta, phi = cart2pol_3d(c_[0], c_[1], c_[2])
        direct.append(pol2cart_3d(1.3 - rho, theta, phi))

    direct = np.array(direct)

    ax2.scatter(cs2[:, 0], cs2[:, 1], cs2[:, 2])
    ax2.quiver(cs2[:, 0], cs2[:, 1], cs2[:, 2],
               direct[:, 0], direct[:, 1], direct[:, 2])
    
    ax2.plot_wireframe(x, y, z, color="k")
    
    plt.show()

def plot_3d(initial_points, final_points, iterations, movements, shape, parameters):

    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, )

    if shape=='sphere' or shape=='semi_sphere':

    # elif shape=='arbitrary':
    #     x_v = parameters['x']
    #     y_v = parameters['y']
    #     shape_1 = patches.Polygon(list(zip(x_v, y_v)), closed=True, alpha=.1)
    #     shape_2 = patches.Polygon(list(zip(x_v, y_v)), closed=True, alpha=.1)

        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)

    cs1 = initial_points
    ax1.scatter(cs1[:, 0], cs1[:, 1], cs1[:, 2])
    ax1.plot_wireframe(x, y, z, color="b")
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_title('Initial Point Positions')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    cs2 = final_points
    direct = []
    for c_ in cs2:
        rho, theta, phi = cart2pol_3d(c_[0], c_[1], c_[2])
        direct.append(pol2cart_3d(1 - rho, theta, phi))

    direct = np.array(direct)

    ax2.scatter(cs2[:, 0], cs2[:, 1], cs2[:, 2])
    ax2.quiver(cs2[:, 0], cs2[:, 1], cs2[:, 2],
               direct[:, 0], direct[:, 1], direct[:, 2])
    
    ax2.plot_wireframe(x, y, z, color="b")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_title('Final Point Positions')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    ax3.plot(np.linspace(1, iterations, iterations), movements)
    ax3.set_title('Average Displacement Over Iterations')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Avg Displacement')

    plt.show()

def lloyds_rel_3D(iterations, shape, parameters):

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

    polt_3d_final(final_points=new_points)

    plot_3d(initial_points=initial_points,
        final_points=new_points, 
        iterations=iterations, 
        movements=movements, 
        shape=shape, 
        parameters=parameters)
