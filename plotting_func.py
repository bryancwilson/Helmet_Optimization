import matplotlib.pyplot as plt
import numpy as np


# ====================================== 3D Plotting Functions ======================================
from conv_func import cart2pol_3d, pol2cart_3d

# create base plot
def plot_3d_final(final_points):
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
    # ax2.quiver(cs2[:, 0], cs2[:, 1], cs2[:, 2],
    #            direct[:, 0], direct[:, 1], direct[:, 2])
    
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

