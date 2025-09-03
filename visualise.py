import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import os

def read_binary(filename):
    f = open(filename, "rb")
    chunk = f.read(8)
    particle_count = struct.unpack('q', chunk)[0]

    rank_array = np.zeros(particle_count)
    coord_array = np.zeros((particle_count,3))

    for i in range(particle_count):
        chunk = f.read(4)
        rank_array[i] = struct.unpack('i', chunk)[0]
        chunk = f.read(8)
        chunk = f.read(8)
        coord_array[i][0] = struct.unpack('d', chunk)[0]
        chunk = f.read(8)
        coord_array[i][1] = struct.unpack('d', chunk)[0]
        chunk = f.read(8)
        coord_array[i][2] = struct.unpack('d', chunk)[0]

    return particle_count, rank_array, coord_array

files = os.listdir("./build")
par_file = ""
for f in files:
    if f.startswith("particle_file"):
        par_file = os.path.join("./build", f)
        break

particle_count, rank_array, coord_array = read_binary(par_file)
unique_ranks = np.unique(rank_array).astype(int)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(121, projection='3d')
ax.set_box_aspect((np.ptp(coord_array[:,0]), np.ptp(coord_array[:,1]), np.ptp(coord_array[:,2])))

cm = plt.get_cmap('plasma')
scat = ax.scatter(coord_array[:,0], coord_array[:,1], coord_array[:,2],
                  c=rank_array, cmap=cm, alpha=1.0)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

rax = plt.axes([0.75, 0.3, 0.2, 0.5])
labels = [str(r) for r in unique_ranks]
visibility = [True]*len(labels)
check = CheckButtons(rax, labels, visibility)

def update(label):
    selected_ranks = [int(lbl) for lbl, vis in zip(labels, check.get_status()) if vis]
    if len(selected_ranks) == 0:
        mask = np.zeros_like(rank_array, dtype=bool)
    else:
        mask = np.isin(rank_array, selected_ranks)
    scat._offsets3d = (coord_array[mask,0], coord_array[mask,1], coord_array[mask,2])
    scat.set_array(rank_array[mask])
    fig.canvas.draw_idle()

check.on_clicked(update)

plt.show()
