import numpy as np

# ============================================= 3D Conversions ======================================
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

def tp2rad_ellipsoid(theta, phi, a, b, c):

    t_1 = ((np.sin(phi)**2) * (np.cos(theta)**2)) / a**2
    t_2 = ((np.sin(phi)**2) * (np.sin(theta)**2)) / b**2
    t_3 = (np.cos(phi)**2) / c**2

    return np.sqrt( 1 / (t_1 + t_2 + t_3))