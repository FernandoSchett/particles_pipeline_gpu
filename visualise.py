import numpy as np
import struct
import matplotlib
import matplotlib.pyplot as plt

def read_binary(filename):
    f = open(filename, "rb")
    chunk = f.read(8)
    particle_count = struct.unpack('q', chunk)[0]

    print("Total particles: " + str(particle_count))
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
    

particle_count, rank_array, coord_array = read_binary("./build/particle_file")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

cm = plt.get_cmap('plasma')
p = ax.scatter(coord_array[:,0], coord_array[:,1], coord_array[:,2], c=rank_array, cmap=cm)
ax.set_xlabel('X')
ax.set_ylabel('Y') 
ax.set_zlabel('Z')

fig.colorbar(p)

plt.savefig("output.png")