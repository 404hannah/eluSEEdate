import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

def rot(vec, r):
    # Rotates a vector 'vec' by rotation vector 'r'
    R, _ = cv2.Rodrigues(r)
    return np.matmul(R, vec)

def plot_cam(ax, t, r, rotation_fun=rot):
    # Draws a little 3D arrow representing a camera in 3D space
    x, y, z = t
    x_vec = rotation_fun(np.array([0.3, 0, 0]), r)
    y_vec = rotation_fun(np.array([0, 0.3, 0]), r)
    z_vec = rotation_fun(np.array([0, 0, 1]), r)

    ax.quiver([x]*3, [y]*3, [z]*3, [x_vec[0], y_vec[0], z_vec[0]], 
              [x_vec[1], y_vec[1], z_vec[1]], [x_vec[2], y_vec[2], z_vec[2]], 
              color=["red", "green", "blue"])

def plot_cams(ts, rs, ponts3d=None, rotation_fun=rot):
    """
    Plots multiple camera positions in a 3D matplotlib graph.
    Good for visualizing the trajectory in 3D.
    """
    fig = plt.figure()
    plt.title("Camera Trajectory")
    ax = fig.add_subplot(111, projection='3d')
    
    for t, r in zip(ts, rs):
        plot_cam(ax, t, r, rotation_fun=rotation_fun)

    if ponts3d is not None:
        xs, ys, zs = ponts3d.T
        ax.scatter(xs, ys, zs)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.view_init(elev=-25, azim=-90)
    plt.show()