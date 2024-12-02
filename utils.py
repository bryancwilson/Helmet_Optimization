import numpy as np
from itertools import combinations
from math import dist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, ArtistAnimation
import math

def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart_array(r_s, theta_s):
    cart_x = []
    cart_y = []

    for r, t in zip(r_s, theta_s):
        x, y = pol2cart(r, t)
        cart_x.append(x)
        cart_y.append(y)
    
    return cart_x, cart_y

# generate cartesian points
def point_generation(size):
    # generate random seed of polar coordinates
    r_bounds = [0, .90]
    theta_bounds = [0, 360]

    r_s = np.random.uniform(r_bounds[0], r_bounds[1], (size, ))
    theta_s = np.random.uniform(theta_bounds[0], theta_bounds[1], (size, ))

    cart_x, cart_y = pol2cart_array(r_s, theta_s)
    return cart_x, cart_y

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
def plot(coords, vel):
    fig, ax = plt.subplots(figsize=(8, 6))
    unit_circle = patches.Circle((0, 0), radius=1, fill=False)
    ax.add_patch(unit_circle)
    p_plot = ax.scatter(np.array(coords)[:, 0], np.array(coords)[:, 1], marker='o', color='red', alpha=0.5)
    p_arrow = ax.quiver(np.array(coords)[:, 0], np.array(coords)[:, 1], np.array(list(vel.values()))[:, 0], np.array(list(vel.values()))[:, 1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)


    return p_arrow

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
    for _ in range(iterations):

        # apply objective function on population
        distr = {}
        for i, p in enumerate(population):
            neigh = _neighbors(p, hp['num_neighbors'])

            kls = []
            for point in p:
                # collect neighbors from point of interest
                dist, nbrs = neigh.kneighbors(np.reshape(point, (1, 2)))

                # Calculate KL Divergences
                kl_div = KL(dist[0][1:], hp['num_neighbors'])
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

        print('Best KL Score', list(sorted_dict.values())[0])

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
def particle_swarm(x, y, iterations, hp):

    # initialize dict with ids to positions and velocities
    axs = []
    coords_id = {}
    best_local_coord = {}
    best_local_dist = {}
    coords = []
    vel = {}
    for i in range(len(x)):
        coords_id[i] = [x[i][0], y[i][0]]
        vel[i] = [np.random.uniform(-.1, .1), np.random.uniform(-.1, .1)]
        coords.append([x[i][0], y[i][0]])

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

        best_local_dist[i] = kl_div

    # calculate neighbors
    neigh = _neighbors(list(coords_id.values()), hp['num_neighbors'])

    # begin optimization
    for n in range(iterations):

        for i in range(0, len(x), 2):

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

            # # calculate distance from point to wall
            # r, _ = cart2pol(coords_id[i][0], coords_id[i][1])
            # dist_from_wall = 1 - r
            # run_dist += dist_from_wall
            # run_dist = run_dist / len(nbrs[0] - 1)

            # update best local coordinate position
            if best_local_dist[i] > kl_div:
                best_local_dist[i] = kl_div
                best_local_coord[i] = [coords_id[i][0], coords_id[i][1]]
            
            # update global position =================================================================================================
            id = max(best_local_dist, key=best_local_dist.get)

            best_global_coord = coords_id[id]

        print("Average Distance", np.mean(list(best_local_dist.values())))

    # fig = plt.figure()
    # ani = ArtistAnimation(fig, axs, interval=10, blit=True)
    # ani.save("animation.mp4", fps=3)

    fig = plt.figure()
    ax = fig.add_subplot()
    cs = np.array(list(coords_id.values()))
    ax.scatter(cs[:, 0], cs[:, 1])

    # # plot unit circle
    # unit_circle = patches.Circle((0, 0), radius=1, fill=False)
    # ax.add_patch(unit_circle)

    plt.show()







