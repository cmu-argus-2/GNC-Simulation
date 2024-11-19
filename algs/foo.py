from PlaneFitting import fit_plane_RANSAC
from SunSensorAlgs import compute_body_ang_vel_from_sun_rays
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(1)

init_sun_rays = [
    [1.00100000e01, np.array([2.00185352e-01, -3.37144643e-01, 8.58055543e-01])],
    [2.00100000e01, np.array([-2.01763733e-01, -5.83118781e-01, 7.13627272e-01])],
    [3.00100000e01, np.array([-5.34788415e-01, -1.56241069e-01, 7.61308137e-01])],
    [4.00200000e01, np.array([-4.76457362e-01, 3.75988180e-01, 7.22233529e-01])],
    [5.00300000e01, np.array([1.36997588e-01, 5.76722144e-01, 7.33909551e-01])],
    [6.00400000e01, np.array([5.11192336e-01, 1.02898043e-01, 7.86189792e-01])],
    [7.00400000e01, np.array([4.24583418e-01, -2.71140136e-01, 7.97628954e-01])],
    [8.00400000e01, np.array([2.45167447e-02, -7.12742187e-01, 6.17573886e-01])],
    [9.00400000e01, np.array([-5.10836757e-01, -1.57525418e-01, 7.77323324e-01])],
    [1.00040000e02, np.array([-3.39270446e-01, 2.83171723e-01, 8.33492255e-01])],
    [1.10040000e02, np.array([-4.76709358e-03, 4.67968743e-01, 8.19135233e-01])],
    [1.20040000e02, np.array([4.57810344e-01, 3.88329173e-01, 7.27743185e-01])],
    [1.30040000e02, np.array([4.00079708e-01, -1.75741898e-01, 8.36092705e-01])],
    [1.40050000e02, np.array([8.70172436e-02, -5.57065234e-01, 7.56377105e-01])],
    [1.50060000e02, np.array([-4.24344748e-01, -3.45397675e-01, 7.68525849e-01])],
    [1.60070000e02, np.array([-5.80224178e-01, 1.55011277e-01, 7.27537908e-01])],
    [1.70080000e02, np.array([-2.97050773e-01, 4.23416151e-01, 7.88973765e-01])],
    [1.80090000e02, np.array([5.76607109e-01, 2.64173627e-01, 6.98381369e-01])],
    [1.90100000e02, np.array([4.14193484e-01, 2.14924958e-02, 8.47338085e-01])],
    [2.00110000e02, np.array([3.17465027e-01, -5.05988616e-01, 7.30199615e-01])],
    [2.10120000e02, np.array([-3.63971568e-01, -5.49658837e-01, 6.74833209e-01])],
    [2.20130000e02, np.array([-5.16368655e-01, 3.72948223e-02, 7.88652337e-01])],
    [2.30140000e02, np.array([-2.77023958e-01, 4.70617747e-01, 7.69270214e-01])],
    [2.40150000e02, np.array([3.52548155e-01, 4.34018613e-01, 7.59827377e-01])],
    [2.50160000e02, np.array([7.28338759e-02, 2.37220947e-01, 9.10176603e-01])],
    [2.60170000e02, np.array([3.07864486e-01, -3.46072206e-01, 8.21859773e-01])],
    [2.70180000e02, np.array([-9.73951983e-02, -5.85106802e-01, 7.33596760e-01])],
]
sun_rays = np.array([ray for (t, ray) in init_sun_rays])

print()
w, cov = compute_body_ang_vel_from_sun_rays(init_sun_rays)
print("w:\n", w)
print("cov:\n", cov)
print()

plane, inlier_idxs = fit_plane_RANSAC(sun_rays, tolerance=1e-1)
inlier_sunrays = np.array([sun_rays[idx] for idx in inlier_idxs])
rotation_axes = []
for i in range(len(inlier_sunrays)):
    all_but_one_sunray = inlier_sunrays[np.random.choice(range(len(inlier_sunrays)), len(inlier_sunrays) - 1)]
    plane, inlier_idxs = fit_plane_RANSAC(all_but_one_sunray, tolerance=1e-1)
    plane_normal = plane[0:3] / np.linalg.norm(plane[0:3])
    rotation_axis = plane_normal
    rotation_axis *= np.sign(rotation_axis[2])

    print(f"{len(inlier_idxs)}  DIY: ", rotation_axis)
    rotation_axes.append(rotation_axis)
rotation_axes = np.array(rotation_axes)
print(np.cov(rotation_axes.T))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
x = [ray[0] for ray in sun_rays]
y = [ray[1] for ray in sun_rays]
z = [ray[2] for ray in sun_rays]
ax.scatter(x, y, z)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")
plt.show()
