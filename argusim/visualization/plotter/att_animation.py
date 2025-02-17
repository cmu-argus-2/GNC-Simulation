import matplotlib.pyplot as plt
import numpy as np
from argusim.world.math.quaternions import quatrotation
from matplotlib.animation import FuncAnimation
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import yaml


def update_quiver(num, data_dicts, quiver_nadir, quiver_mag, quiver_sp, quiver_im, sun_quiver, params):
    G_rw_b     = params["G_rw_b"]
    major_axis = params["major_axis"]

    quat = np.array([data_dicts[0]["q_w"][num], data_dicts[0]["q_x"][num], data_dicts[0]["q_y"][num], data_dicts[0]["q_z"][num]])
    
    Re2b = quatrotation(quat)
    nadir = -np.array([data_dicts[0]["r_x ECI [m]"][num], data_dicts[0]["r_y ECI [m]"][num], data_dicts[0]["r_z ECI [m]"][num]])
    nadir = nadir / np.linalg.norm(nadir)
    mag_field = np.array([data_dicts[0]["xMag ECI [T]"][num], data_dicts[0]["yMag ECI [T]"][num], data_dicts[0]["zMag ECI [T]"][num]])
    mag_field = mag_field / np.linalg.norm(mag_field)
    body_z = Re2b @ G_rw_b.flatten()
    # inertia_min = Re2b @ eigenvectors[:, idx[0]]
    # inertia_med = Re2b @ eigenvectors[:, idx[1]]
    inertia_max = Re2b @ major_axis
    sun_vector_eci = np.array([data_dicts[0]["rSun_x ECI [m]"][num], data_dicts[0]["rSun_y ECI [m]"][num], data_dicts[0]["rSun_z ECI [m]"][num]])
    sun_vector_eci = sun_vector_eci / np.linalg.norm(sun_vector_eci)

    quiver_nadir.set_segments([[[0, 0, 0], nadir]])
    quiver_mag.set_segments([[[0, 0, 0], mag_field]])
    quiver_sp.set_segments([[[0, 0, 0], body_z]])
    quiver_im.set_segments([[[0, 0, 0], inertia_max]])
    sun_quiver.set_segments([[[0, 0, 0], sun_vector_eci]])

    return quiver_nadir, quiver_mag, quiver_sp, quiver_im, sun_quiver


def att_animation(pyparams, data_dicts):
    if not pyparams["PlotFlags"]["att_animation"]:
        return
    trials             = pyparams["trials"]
    trials_dir         = pyparams["trials_dir"]
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    J = np.array(pyparams["inertia"]["nominal_inertia"]).reshape((3,3))
    eigenvalues, eigenvectors = np.linalg.eig(J)
    idx = np.argsort(eigenvalues)
    major_axis = eigenvectors[:, idx[2]]
    if major_axis[np.argmax(np.abs(major_axis))] < 0:
        major_axis = -major_axis
    num_RWs = pyparams["reaction_wheels"]["N_rw"]
    G_rw_b = np.array(pyparams["reaction_wheels"]["rw_orientation"]).reshape(3, num_RWs)
    anim_params = {
        "G_rw_b" : G_rw_b,
        "major_axis" : major_axis
    }

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Attitude Animation in ECI with Sun Vector')

    quiver_nadir = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, color='r', label='Nadir')
    quiver_mag   = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, color='g', label='Mag Field')
    quiver_sp    = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, color='b', label='Solar Panels/+Z')
    # quiver_x2 = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, label='Min J Axis')
    # quiver_y2 = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, label='Med J Axis')
    quiver_im = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, color='k', label='Max J Axis')
    sun_quiver = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, color='y', label='Sun Vector')
    # add a cube representation
    # Define the vertices of a cube
    r = [-0.5, 0.5]
    vertices = np.array([[x, y, z] for x in r for y in r for z in r])

    # Define the edges that connect the vertices
    edges = [
        [vertices[j] for j in [0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2]]
    ]

    # Create a Poly3DCollection object for the cube
    cube = Poly3DCollection(edges, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
    ax.add_collection3d(cube)

    def update_cube(num, data_dicts, cube, params):
        quat = np.array([data_dicts[0]["q_w"][num], data_dicts[0]["q_x"][num], data_dicts[0]["q_y"][num], data_dicts[0]["q_z"][num]])
        Re2b = quatrotation(quat)
        rotated_vertices = np.dot(vertices, Re2b.T)
        new_edges = [[rotated_vertices[j] for j in [0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2]]]
        cube.set_verts(new_edges)
        return cube,

    ax.legend([quiver_nadir, quiver_mag, quiver_sp, quiver_im, sun_quiver], ['Nadir', 'Mag Field', 'Solar Panels/+Z', 'Max J Axis', 'Sun Vector'])
    
    fps = 24
    max_duration = 15  # seconds
    max_frames = fps * max_duration
    total_frames = len(data_dicts[0]["Time [s]"])
    step = max(1, total_frames // max_frames)

    ani = FuncAnimation(fig, update_quiver, frames=range(0, total_frames, step), 
                        fargs=(data_dicts, quiver_nadir, quiver_mag, quiver_sp, quiver_im, sun_quiver, anim_params), 
                        interval=1000/fps, blit=False)
    ani.cube = FuncAnimation(fig, update_cube, frames=range(0, total_frames, step), 
                             fargs=(data_dicts, cube, anim_params), 
                             interval=1000/fps, blit=False)
    ani.save(os.path.join(plot_dir, 'att_anim_BF_Sun.gif'), writer='pillow', fps=20)