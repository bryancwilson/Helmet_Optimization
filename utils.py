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

# generate random polygon
def generate_random_polygon(num_points, min_x, max_x, min_y, max_y):
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
def point_generation(size, shape, parameters, plot=False):
    # generate random seed of polar coordinates

    if shape == 'arbitrary':
        fig, ax = plt.subplots()

        # vertices
        x_v = parameters['x']
        y_v = parameters['y']

        vertices = []
        for x_, y_ in zip(x_v, y_v):
            vertices.append((x_, y_))

        # generate polygon
        pg = Polygon(vertices)

        # generate point in polygon
        minx, miny, maxx, maxy = pg.bounds
        cart_x = []
        cart_y = []
        while len(cart_x) < size:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            if pg.contains(point):
                cart_x.append(x)
                cart_y.append(y)


        if plot:
            polygon = patches.Polygon(list(zip(x_v, y_v)), closed=True, alpha=.1)

            ax.add_patch(polygon)
            ax.scatter(np.array(cart_x), np.array(cart_y), marker='o', color='red', alpha=0.5)
            ax.set_xlim([np.min(x_v), np.max(x_v)])
            ax.set_ylim([np.min(y_v), np.max(y_v)])

            plt.show()

    elif shape == 'circle' or shape == 'semi_circle':
        r_bounds = [parameters['r_min'], parameters['r_max']]
        theta_bounds = [parameters['theta_min'], parameters['theta_max']]

        r_s = np.random.uniform(r_bounds[0], r_bounds[1], (size, ))
        theta_s = np.random.uniform(theta_bounds[0], theta_bounds[1], (size, ))

        cart_x, cart_y = pol2cart_array(r_s, theta_s)

        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            unit_circle = patches.Circle((0, 0), radius=1, fill=False)
            ax.add_patch(unit_circle)
            ax.scatter(np.array(cart_x), np.array(cart_y), marker='o', color='red', alpha=0.5)
            
            plt.show()

        pg = None

    return cart_x, cart_y, pg

# metric functions
def inter_distance(coords):
    coord_pairs = combinations(coords, 2)

    run_dist_acc = 0
    for p in coord_pairs:
        run_dist_acc+=dist(p)

    avg_dist = run_dist_acc / len(coord_pairs)
    
def _neighbors(points_list, k):

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points_list)
    
    return neigh

# create base plot
def plot_final(final_points, shape, parameters, ellipse=False, vectors=False, helmet_shape=None, helmet_parameters=None):

    fig, ax = plt.subplots()

    if shape=='circle' or shape=='semi_circle' or shape=='ellipse':
        #shape_1 = patches.Circle((0, 0), radius=1, fill=False)
        shape_2 = patches.Circle((0, 0), radius=1, fill=False)

        # if helmet_shape != None:
        #     # shape_1 = patches.Circle((0, -.5), radius=1.3, fill=False)
        #     shape_2_helmet = patches.Circle((0, -.25), radius=helmet_parameters['radius'], fill=False) 
        #     # shape_2_helmet = patches.Ellipse((0, .5), width=3, height=1.5, fill=False) 

        if helmet_shape=='Ellipse':
            shape_2_helmet = patches.Ellipse(helmet_parameters['center'], 
                                             width=helmet_parameters['a'], 
                                             height=helmet_parameters['b'], 
                                             fill=False) 

    elif shape=='arbitrary':
        x_v = parameters['x']
        y_v = parameters['y']
        #shape_1 = patches.Polygon(list(zip(x_v, y_v)), closed=True, alpha=.1)
        shape_2 = patches.Polygon(list(zip(x_v, y_v)), closed=True, alpha=.1)

    assert not (ellipse == True and vectors == True), "parameters 'ellipse' and 'vectors' should not both be True"

    if ellipse:
        for p in final_points:
            theta = int(round(np.random.uniform(0, 360)))
            ell = patches.Ellipse(xy=tuple(p), width=np.cos(theta), height=np.sin(theta), angle=theta, facecolor='red', alpha=0.2)
            ax.add_patch(ell)
        
    if vectors:
    # plot vectors
        cs2 = final_points
        direct = []
        for c_ in cs2:
            rho, theta = cart2pol(c_[0], c_[1])
            direct.append(pol2cart(1.3 - rho, theta))

            ell = patches.Ellipse(xy=tuple(c_), width=np.cos(theta)/10, height=np.sin(theta)/5, angle=theta, facecolor='red', alpha=0.2)
            ax.add_patch(ell)

        direct = np.array(direct)
        # ax.quiver(cs2[:, 0], cs2[:, 1],
        #              direct[:, 0], direct[:, 1])


    cs2 = final_points
    ax.scatter(cs2[:, 0], cs2[:, 1])
    ax.add_patch(shape_2)
    ax.add_patch(shape_2_helmet)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_title('Final Point Positions')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()

    if helmet_shape:
        return shape_2_helmet
  

def plot_all(initial_points, final_points, iterations, movements, shape, parameters, ellipse=False, vectors=False):

    fig, ax = plt.subplots(1, 3)

    if shape=='circle' or shape=='semi_circle':
        shape_1 = patches.Circle((0, 0), radius=1, fill=False)
        shape_2 = patches.Circle((0, 0), radius=1, fill=False)
    elif shape=='arbitrary':
        x_v = parameters['x']
        y_v = parameters['y']
        shape_1 = patches.Polygon(list(zip(x_v, y_v)), closed=True, alpha=.1)
        shape_2 = patches.Polygon(list(zip(x_v, y_v)), closed=True, alpha=.1)

    assert not (ellipse == True and vectors == True), "parameters 'ellipse' and 'vectors' should not both be True"

    if ellipse:
        for p in final_points:
            theta = int(round(np.random.uniform(0, 360)))
            ell = patches.Ellipse(xy=tuple(p), width=.1, height=.05, angle=theta, facecolor='red', alpha=0.2)
            ax[1].add_patch(ell)
        
    if vectors:
    # plot vectors
        cs2 = final_points
        direct = []
        for c_ in cs2:
            rho, theta = cart2pol(c_[0], c_[1])
            direct.append(pol2cart(1.3 - rho, theta))

            ell = patches.Ellipse(xy=tuple(c_), width=.1, height=.05, angle=theta, facecolor='red', alpha=0.2)
            ax[1].add_patch(ell)

        direct = np.array(direct)
        ax[1].quiver(cs2[:, 0], cs2[:, 1],
                     direct[:, 0], direct[:, 1])

    cs1 = initial_points
    ax[0].scatter(cs1[:, 0], cs1[:, 1])
    ax[0].add_patch(shape_1)
    ax[0].set_xlim(-3, 3)
    ax[0].set_ylim(-3, 3)
    ax[0].set_title('Initial Point Positions')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    cs2 = final_points
    ax[1].scatter(cs2[:, 0], cs2[:, 1])
    ax[1].add_patch(shape_2)
    ax[1].set_xlim(-3, 3)
    ax[1].set_ylim(-3, 3)
    ax[1].set_title('Final Point Positions')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    ax[2].plot(np.linspace(1, iterations, iterations), movements)
    ax[2].set_title('Average Displacement Over Iterations')
    ax[2].set_xlabel('Iterations')
    ax[2].set_ylabel('Avg Displacement')

    plt.show()

# sunflower
def radius(k, n, b):
    if k>n-b:
        return 1
    else:
        return math.sqrt(k - (1/2))/math.sqrt(n - (b+1)/2)
    

def sun_flower(n, alpha, ellipse=False):
    points = []
    b = round(alpha*math.sqrt(n))
    phi = (math.sqrt(5) + 1) / 2
    for k in range(1, n+1):

        r = radius(k, n, b)
        theta = (2*math.pi*k)/phi**2

        points.append(pol2cart(r, theta))

    fig, ax = plt.subplots(1)
    # plot ellipses
    if ellipse:
        for p in points:
            theta = int(round(np.random.uniform(0, 360)))
            ell = patches.Ellipse(xy=tuple(p), width=.1, height=.05, angle=theta, facecolor='blue', alpha=0.2)
            ax.add_patch(ell)

    ax.scatter(np.array(points)[:, 0], np.array(points)[:, 1])
    # plot unit circle
    unit_circle = patches.Circle((0, 0), radius=1, fill=False)
    ax.add_patch(unit_circle)
    plt.show()

def KL(a, num_neighbors):
    a = np.asarray(a, dtype=np.float32) / sum(a)
    b = np.ones((1, num_neighbors-1)) / (num_neighbors-1)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def crossover(point_pairs, population):

    coords = []
    ar1 = population[point_pairs[0]]
    ar2 = population[point_pairs[1]]
    for p1, p2 in zip(ar1, ar2):
        spin_wheel = np.random.rand()
        if spin_wheel > .9:
            x = p1[0]
            y = p1[1]
        elif spin_wheel < .1:
            x = p2[0]
            y = p2[1]
        else:
            x = np.random.uniform(p1[0], p2[0])
            y = np.random.uniform(p1[1], p2[1])

        coords.append([x, y])

    return np.array(coords)

def evolutionary_algorithm(iterations, hp):

    cart_x, cart_y = point_generation(128*hp['pop_multiplier'])

    # generate initial population
    population = []
    for i in range(hp['pop_multiplier']):
        population.append(np.transpose(np.vstack((cart_x[i*128:(i+1)*128], cart_y[i*128:(i+1)*128]))))

    inital_pop = population

    # optimize
    for e in range(iterations):

        # apply objective function on population
        distr = {}
        for i, p in enumerate(population):
            neigh = _neighbors(p, hp['num_neighbors'])

            kls = []
            for point in p:

                # collect neighbors from point of interest
                dist, nbrs = neigh.kneighbors(np.reshape(point, (1, 2)))

                # Calculate KL Divergences
                kl_div = KL(dist[0][1:], hp['num_neighbors']) - np.mean(dist[0][1:])

                kls.append(kl_div)

            # # calculate distance from point to wall
            # r, _ = cart2pol(coords_id[i][0], coords_id[i][1])
            # dist_from_wall = 1 - r
            # run_dist += dist_from_wall
            # run_dist = run_dist / len(nbrs[0] - 1)

            distr[i] = np.mean(kls)

        # select k best candidates
        sorted_dict = dict(sorted(distr.items(), key=lambda item: item[1]))
        best_candidates = list(sorted_dict.keys())[:hp['k']]

        # perform crossovers
        pairs = combinations(best_candidates, r=2)

        # generate children
        children = []
        for pa in pairs:
            children.append(crossover(pa, population))

        np.random.shuffle(children)
        population = children[:10]

        print('EPOCH: ', e, 'Best KL Score', list(sorted_dict.values())[0])

    fig, ax = plt.subplots(1, 2)

    cs1 = inital_pop[0]
    ax[0].scatter(cs1[:, 0], cs1[:, 1])

    cs2 = population[0]
    ax[1].scatter(cs2[:, 0], cs2[:, 1])

    # # plot unit circle
    # unit_circle = patches.Circle((0, 0), radius=1, fill=False)
    # ax.add_patch(unit_circle)

    plt.show()

# optimization functions
def particle_swarm(iterations, hp):

    # generate points
    x, y = point_generation(128)

    # initialize dict with ids to positions and velocities
    axs = []
    coords_id = {}
    best_local_coord = {}
    best_local_dist = {}
    coords = []
    vel = {}
    for i in range(len(x)):
        coords_id[i] = [x[i], y[i]]
        vel[i] = [np.random.uniform(-.1, .1), np.random.uniform(-.1, .1)]
        coords.append([x[i], y[i]])

    initial_coords = np.array(list(coords_id.values()))
    np.random.shuffle(coords)

    for i, c in enumerate(coords):
        best_local_coord[i] = c

    # calculate neighbors
    neigh = _neighbors(coords, hp['num_neighbors'])
    best_local_dist = {}

    # # initialize average distance and best distance dictionary
    for i in range(len(coords)):

        # collect neighbors from point of interest
        dist, nbrs = neigh.kneighbors([[best_local_coord[i][0], best_local_coord[i][1]]])

        # calculate KL Divergence
        kl_div = KL(dist[0][1:], hp['num_neighbors'])

        # calculate distance from point to wall
        r, _ = cart2pol(coords_id[i][0], coords_id[i][1])
        dist_from_wall = 1 - r

        # overal metric
        metric = kl_div - 2*dist_from_wall

        best_local_dist[i] = metric

    # calculate neighbors
    neigh = _neighbors(list(coords_id.values()), hp['num_neighbors'])

    # begin optimization
    dfw_ = []
    kls_ = []
    met_ = []
    for n in range(iterations):

        dists_from_wall = []
        kls = []
        for i in range(0, len(x)):

            # update quality of local positions =====================================================================================)
            # calculate new velocity for each point
            s1 = np.random.random()
            s2 = np.random.random()
            # s1 = 1
            # s2 = 1
            vel[i][0] = vel[i][0] + s1*hp['c1']*(best_local_coord[i][0] - coords_id[i][0]) 
            vel[i][1] = vel[i][1] + s2*hp['c1']*(best_local_coord[i][1] - coords_id[i][1])

            # calculate new position
            r, theta = cart2pol(coords_id[i][0] + vel[i][0], coords_id[i][1] + vel[i][1])
            if r > .90:
                coords_id[i][0], coords_id[i][1] = pol2cart(.90, theta)
            else:
                coords_id[i][0] += vel[i][0]
                coords_id[i][1] += vel[i][1]

            # calculate neighbors
            neigh = _neighbors(coords, hp['num_neighbors'])

            # collect neighbors from point of interest
            dist, nbrs = neigh.kneighbors([[coords_id[i][0], coords_id[i][1]]])

            # Calculate KL Divergence
            kl_div = KL(dist[0][1:], hp['num_neighbors'])
            kls.append(kl_div)

            # # calculate distance from point to wall
            r, _ = cart2pol(coords_id[i][0], coords_id[i][1])
            dist_from_wall = 1 - r

            dists_from_wall.append(dist_from_wall)
            # overal metric
            metric = kl_div - 2*dist_from_wall

            # update best local coordinate position
            if best_local_dist[i] > metric:
                best_local_dist[i] = metric
                best_local_coord[i] = [coords_id[i][0], coords_id[i][1]]
            
            # update global position =================================================================================================
            id = max(best_local_dist, key=best_local_dist.get)

            best_global_coord = coords_id[id]

        print("EPOCH: ", n, "Average Metric", np.mean(list(best_local_dist.values())), "KL: ", np.mean(kls), "FW: ", np.mean(dists_from_wall))
        
        dfw_.append(np.mean(dists_from_wall))
        kls_.append(np.mean(kls))
        met_.append(np.mean(list(best_local_dist.values())))

    # fig = plt.figure()
    # ani = ArtistAnimation(fig, axs, interval=10, blit=True)
    # ani.save("animation.mp4", fps=3)

    fig, ax = plt.subplots(1, 3)

    cs1 = initial_coords
    ax[0].scatter(cs1[:, 0], cs1[:, 1])
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(-1, 1)

    cs2 = np.array(list(coords_id.values()))
    ax[1].scatter(cs2[:, 0], cs2[:, 1])
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)

    x = np.linspace(1, iterations, iterations)
    ax[2].plot(x, met_, label='Metric')
    ax[2].plot(x, kls_, label='KL')
    ax[2].plot(x, dfw_, label='DFW')
    ax[2].legend()

    # # plot unit circle
    # unit_circle = patches.Circle((0, 0), radius=1, fill=False)
    # ax.add_patch(unit_circle)

    plt.show()



def lloyds_rel(iterations, shape, parameters, plot=False):

    # generate points

    x, y, pg = point_generation(128,
                    shape,
                    parameters,
                    True)
    
    coords = []
    for i in range(len(x)):
        coords.append([x[i], y[i]])

    initial_points = np.array(coords)
    points = np.array(coords)

    center = [0,0]
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
                
                if shape == 'circle':
                    # Clip centroid to make sure it's inside the region
                    dist = np.linalg.norm(centroid - center)
                    if dist > radius:
                        # Project the centroid back onto the circle boundary
                        centroid = center + (centroid - center) * radius / dist

                elif shape == 'semi_circle':

                    # Clip centroid to make sure it's inside the region
                    dist = np.linalg.norm(centroid - center)
                    if dist > radius:
                        # Project the centroid back onto the circle boundary
                        centroid = center + (centroid - center) * radius / dist

                    if centroid[1] < 0:
                        # Move centroid back inside semi circle
                        centroid[1] = 0

                elif shape == 'arbitrary':

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
        plot_final(new_points, 
                shape, 
                parameters, 
                ellipse=False, 
                vectors=True)

        plot_all(initial_points=initial_points,
            final_points=new_points, 
            iterations=iterations, 
            movements=movements, 
            shape=shape, 
            parameters=parameters,
            ellipse=True,
            vectors=False)
        
    return new_points

def optimize_angle(shape, parameters, new_points, depth, helmet_parameters):

    helmet_shape = plot_final(new_points, 
        shape, 
        parameters, 
        ellipse=False, 
        vectors=True,
        helmet_shape='Ellipse',
        helmet_parameters=helmet_parameters)
    
    thetas = []
    helmet_path = helmet_shape.get_path()
    focal_points = new_points
    new_points = list(new_points)
    # shape_2_helmet = patches.Circle((0, -.25), radius=1.6, fill=False) 

    center = helmet_parameters['center']

    point_angle_dict = {}
    while new_points != []:

        # remove focal point from list
        np_ = new_points.pop(0)

        best_obj_val = float('inf')
        used_ts = []

        # iterate through n points on helmet shape perimeter
        steps = (helmet_parameters['circumference'] / 2) / helmet_parameters['ele_size']
        for t_ in np.linspace(0, np.pi, int(round(steps))):

            # ignore angles that overlap with helmet opening
            if t_ > ((np.pi / 2) - np.arctan2(helmet_parameters['hole_size'], helmet_parameters['b'])) and t_ < ((np.pi / 2) + np.arctan2(helmet_parameters['hole_size'], helmet_parameters['b'])):
                continue

            # if t_ in used_ts:
            #     continue

            if shape=='ellipse':
                r = theta2rad_ellipse(t_,
                                  helmet_parameters['a'],
                                  helmet_parameters['b'])
                
                x_, y_ = pol2cart(r, t_)
                
            else:
                x_, y_ = pol2cart(helmet_parameters['radius'], t_)
                y_+=center[1] # offset due to radius

            dist = np.linalg.norm(np.array(x_, y_) - np_) # calculate distance between points
            angle_pp = np.atan2(y_ - np_[1], x_ - np_[0]) # calculate angle between points
            angle_pc = np.atan2(y_ - center[1], x_ - center[0])

            obj_val = 0*np.abs(depth - dist) + 1*np.abs(angle_pp - angle_pc) # evaluate objective function

            # assign focal point to element according to objective function
            if obj_val < best_obj_val:
                point_angle_dict[tuple(np_)] = (x_, y_, t_, dist, np.abs(angle_pp - angle_pc), obj_val)
                best_obj_val = obj_val

        # check if new assignment is better than old assignment
        for key, val in point_angle_dict.items():
            if val[0] == point_angle_dict[tuple(np_)][0] and val[1] == point_angle_dict[tuple(np_)][1] and best_obj_val < val[5]:
                new_points.append(np.array(key))
                point_angle_dict[key] = (0, 0, 0, 0, 0, 0)

        used_ts.append(t_)

    # plot resulting element positions
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    shape = patches.Circle((0, 0), radius=1, fill=False)
    shape_helmet = patches.Ellipse(helmet_parameters['center'], 
                                             width=helmet_parameters['a'], 
                                             height=helmet_parameters['b'], 
                                             fill=False) 

    values = np.array(list(point_angle_dict.values()))
    ax1.scatter(values[:, 0], values[:, 1])
    ax1.scatter(focal_points[:, 0], focal_points[:, 1])

    for i in range(len(focal_points)):
        ax1.plot([focal_points[i][0], values[i][0]], [focal_points[i][1], values[i][1]], 'k-')

    ax1.add_patch(shape)
    # ax.add_patch(shape_helmet)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)
    ax1.set_title('Element Assignment Plot')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2.hist(values[:, 3])
    ax2.set_xlabel('Distance From Element to Focal Point')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distance Histogram')

    ax3.hist(values[:, 4])
    ax3.set_xlabel('Angle of Element Relative To Helmet Surface (radians)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Angle Histogram')
    plt.show()

    









            













