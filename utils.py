import numpy as np
from itertools import combinations
from math import dist
from sklearn.neighbors import NearestNeighbors

def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def pol2cart_array(r_s, theta_s):
    cart_x = []
    cart_y = []

    for r, t in zip(r_s, theta_s):
        x, y = pol2cart(r, t)
        cart_x.append(x)
        cart_y.append(y)
    
    return cart_x, cart_y

# metric functions
def _inter_distance(coords):
    coord_pairs = combinations(coords, 2)

    run_dist_acc = 0
    for p in coord_pairs:
        run_dist_acc+=dist(p)

    avg_dist = run_dist_acc / len(coord_pairs)
    
def _neighbors(x, y, k):

    points_list = np.hstack(x, y)
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points_list)
    
    return neigh
    
# optimization functions
def _particle_swarm(x, y, iterations, hp):

    # initialize dict with ids to positions
    coords_id = {}
    for i in range(len(x)):
        coords_id[i] = (x[i], y[i])
        
    # initialize velocity dictionary
    vel = {}
    for _x, _y in zip(x, y):
        vel[(_x, _y)] = 0

    # calculate neighbors
    neigh = _neighbors(x, y, hp['num_neighbors'])

    # initialize average distance and best distance dictionary
    dist = {}
    best_dist = {}
    for _x, _y in zip(x, y):
        distances, _ = neigh.kneighbors([_x, _y])

        avg_dist = 0
        for d in distances:
            avg_dist += d

        dist[(_x, _y)] = d / hp['num_neighbors']
        best_dist[(_x, _y)] = d / hp['num_neighbors']

    for i in range(iterations):
        for _x, _y in zip(x, y):
            # find global best position
            global_best = max(dist)

            # calculate new velocity for each point
            vel[(_x, _y)] = vel[(_x, _y)] + hp['c1']*(best_dist[(_x, _y)] - dist[(_x, _y)]) + hp['c2']*(global_best - dist[(_x, _y)])

            # calculate new position




def _move_coord(x, y):
    pass