#https://stackoverflow.com/questions/48911643/set-uvc-equivilent-for-a-3d-quiver-plot-in-matplotlib

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

minX = np.inf
maxX = -np.inf

minY = np.inf
maxY = -np.inf

minZ = np.inf
maxZ = -np.inf


timestamps = []
rotations = []
translations  = []
inputFile = "3dAnimationData.txt"
linesOfDataPerTimestep = 8 #number of lines recorded at each timestep in the log
with open(inputFile, 'r') as f:
    lines = f.readlines()
    for i in range(0, len(lines), linesOfDataPerTimestep):
        timestamps.append(float(lines[i].strip()))
        roation_matrix_row_1 = lines[i+1].split()
        roation_matrix_row_2 = lines[i+2].split()
        roation_matrix_row_3 = lines[i+3].split()

        translation_vector_x = float(lines[i+4].strip())
        translation_vector_y = float(lines[i+5].strip())
        translation_vector_z = float(lines[i+6].strip())

        minX = min(minX, translation_vector_x)
        minY = min(minY, translation_vector_y)
        minZ = min(minZ, translation_vector_z)

        maxX = max(maxX, translation_vector_x)
        maxY = max(maxY, translation_vector_y)
        maxZ = max(maxZ, translation_vector_z)


        rotations.append(np.asarray([
            roation_matrix_row_1,
            roation_matrix_row_2,
            roation_matrix_row_3
        ]).astype(np.float))

        translations.append(np.asarray([
            translation_vector_x,
            translation_vector_y,
            translation_vector_z
        ]).astype(np.float))
dt = timestamps[1]-timestamps[0]


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('3D Test')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim3d([minX-1, maxX+1])
ax.set_ylim3d([minY-1, maxY+1])
ax.set_zlim3d([minZ-1, maxZ+1])

pos3d = translations[0]
x0 = pos3d[0]
y0 = pos3d[1]
z0 = pos3d[2]

x_arrow_fixed = ax.quiver(x0,y0,z0,x0+1,y0,z0, color="red")
y_arrow_fixed = ax.quiver(x0,y0,z0,x0,y0+1,z0, color="green")
z_arrow_fixed = ax.quiver(x0,y0,z0,x0,y0,z0+1, color="blue")

arrow_length = 1
x_arrow_moving = ax.quiver(x0,y0,z0,x0+arrow_length,y0,z0, color="red")
y_arrow_moving = ax.quiver(x0,y0,z0,x0,y0+arrow_length,z0, color="green")
z_arrow_moving = ax.quiver(x0,y0,z0,x0,y0,z0+arrow_length, color="blue")
moving_frame = [[x_arrow_moving, y_arrow_moving, z_arrow_moving]]

fps = 30 #frames per second in the animation
timestamps_per_frame = int(1/(fps*dt)) #number of input data entries (or timestamps) to skip between each animation frame
def update_plot(frameNumber, moving_frame):
    idx = frameNumber*timestamps_per_frame #idx is the timestamp number
    pos3d = translations[idx]

    moving_frame[0].set_segments([[pos3d,pos3d+rotations[idx][:,0]]])
    moving_frame[1].set_segments([[pos3d,pos3d+rotations[idx][:,1]]])
    moving_frame[2].set_segments([[pos3d,pos3d+rotations[idx][:,2]]])
    plt.suptitle("{:.2f} seconds".format(idx*dt))
    return moving_frame

totalTimestamps = len(timestamps)
totalFrames = int(totalTimestamps/timestamps_per_frame)
ani = animation.FuncAnimation(fig, update_plot, frames=totalFrames, fargs=(moving_frame), interval=750/fps, blit=False)
plt.show()