import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def generate_rotated_points_along_vector(vector, start_point, step_size, num_steps, radius, angle_step=10):
    """
    Generates rotated points along a vector.
    
    Parameters:
    - vector: The direction vector to rotate around.
    - start_point: The starting point of the vector.
    - step_size: Distance between consecutive base points.
    - num_steps: Number of base points along the vector.
    - radius: Radius of rotation around the vector.
    - angle_step: Increment of rotation angles (default is 10 degrees).
    
    Returns:
    - List of rotated points in 3D space.
    """
    v = np.array(vector) / np.linalg.norm(vector)
    base_points = [np.array(start_point) + step_size * i * v for i in range(num_steps)]
    
    # Find an arbitrary perpendicular vector
    if v[0] != 0 or v[1] != 0:
        perp = np.array([-v[1], v[0], 0])
    else:
        perp = np.array([0, -v[2], v[1]])
    
    perp = perp / np.linalg.norm(perp) * radius  # Scale to desired radius

    angles = np.arange(0, 360, angle_step)
    
    x_grid, y_grid, z_grid = [], [], []
    
    for base_point in base_points:
        x_row, y_row, z_row = [], [], []
        for angle in angles:
            theta = np.radians(angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            # Rodrigues' rotation formula
            K = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])

            R = np.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)
            
            # Rotate the perpendicular vector
            rotated_offset = R @ perp
            rotated_point = base_point + rotated_offset
            
            # Store in grid lists
            x_row.append(rotated_point[0])
            y_row.append(rotated_point[1])
            z_row.append(rotated_point[2])
        
        x_grid.append(x_row)
        y_grid.append(y_row)
        z_grid.append(z_row)
    
    return np.array(x_grid), np.array(y_grid), np.array(z_grid)
    
# ====================================== 3D Plotting Functions ======================================
from conv_func import cart2pol_3d, pol2cart_3d

def parameterize_cylinder(diameter, height):

    vector = [1, 1, 1]  # Rotation axis
    start_point = [0, 0, 0]  # Where the vector starts
    step_size = height  # Distance between each base point along the vector
    num_steps = 2  # Number of points along the vector
    radius = diameter/2  # Rotation radius

    x, y, z = generate_rotated_points_along_vector(vector, start_point, step_size, num_steps, radius)
    return x, y, z
    # # # Create the figure and axes object
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # start_points = np.array([[0, 0, 0]])
    # end_points = np.array([[1, 1, 1]])

    # # Plot the arrows
    # for i in range(len(start_points)):
    #     x, y, z = start_points[i]
    #     dx, dy, dz = end_points[i] - start_points[i]
    #     ax.quiver(x, y, z, dx, dy, dz, color=['r', 'g'][i], arrow_length_ratio=0.1)
    # # Add a legend
    # ax.legend()

    # ax.scatter(rotated_points[:, 0],
    #            rotated_points[:, 1],
    #            rotated_points[:, 2])
    # # Show the plot
    # plt.show()

    
    # Generate points for the cylinder
    theta = np.linspace(0, 2*np.pi, n_segments)
    z = np.array([0, height])

    # Create meshgrid for x, y, and z coordinates
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)

    x_grid = np.array([rotated_points[..., 0], rotated_points[..., 0]]) 
    y_grid = np.array([rotated_points[..., 1], rotated_points[..., 1]]) 
    z_grid = np.array([rotated_points[..., 2], rotated_points[..., 2]]) 
    # x_grid, y_grid, z_grid = np.meshgrid(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2])
    return x_grid, y_grid, z_grid

def plot_cylinder(x_grid, y_grid, z_grid):

    # Sphere parameters
    radius = 1
    center_x, center_y, center_z = 0, 0, 0

    # Create the meshgrid for the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center_x
    y = radius * np.outer(np.sin(u), np.sin(v)) + center_y
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z

    # Plot the cylinder
    fig = go.Figure(data=[go.Surface(x=x_grid, y=y_grid, z=z_grid), go.Surface(x=x, 
                                                                               y=y, 
                                                                               z=z, 
                                                                               colorscale=[[0, '#ADD8E6'], [1, '#ADD8E6']], # Light blue color
                                                                               opacity=0.2,
                                                                               showscale=False)])

    # Update layout for better visualization (optional)
    # fig.update_layout(
    #     scene=dict(
    #         xaxis_title='X',
    #         yaxis_title='Y',
    #         zaxis_title='Z'
    #     ),
    #     title='Cylinder Plot'
    # )

    fig.show()

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

    ax2.scatter(cs2[:, 0]*2.5, cs2[:, 1]*2.5, cs2[:, 2]*2.5)
    # ax2.quiver(cs2[:, 0], cs2[:, 1], cs2[:, 2],
    #            direct[:, 0], direct[:, 1], direct[:, 2])
    
    ax2.plot_wireframe(x*2.5, y*2.5, z*2.5, color="k")
    ax2.set_xlabel('mm')
    ax2.set_ylabel('mm')
    ax2.set_zlabel('mm')
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
    ax1.scatter(cs1[:, 0]*2.5, cs1[:, 1]*2.5, cs1[:, 2]*2.5)
    ax1.plot_wireframe(x*2.5, y*2.5, z*2.5, color="k")
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_title('Initial Point Positions')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    cs2 = final_points
    direct = []
    for c_ in cs2:
        rho, theta, phi = cart2pol_3d(c_[0], c_[1], c_[2])
        direct.append(pol2cart_3d(1 - rho, theta, phi))

    direct = np.array(direct)

    ax2.scatter(cs2[:, 0]*2.5, cs2[:, 1]*2.5, cs2[:, 2]*2.5)
    # ax2.quiver(cs2[:, 0], cs2[:, 1], cs2[:, 2],
    #            direct[:, 0], direct[:, 1], direct[:, 2])
    
    ax2.plot_wireframe(x*2.5, y*2.5, z*2.5, color="k")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_title('Final Point Positions')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    ax3.plot(np.linspace(1, iterations, iterations), movements)
    ax3.set_title('Average Displacement Over Iterations')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Avg Displacement')

    plt.show()

