#ifndef HELPER_H
#define HELPER_H

#include <cstring>
#include <fstream>
#include <math.h>
#include <mpi.h>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>

typedef struct particle{
    int mpi_rank;
    long long int key;
    double coord[3];
} t_particle;

#define NPROPS_PARTICLE 3

extern MPI_Datatype MPI_particle;
extern int register_MPI_Particle(MPI_Datatype *MPI_Particle);
int allocate_particle(t_particle **particle_array, int count);
int box_distribution(t_particle **particle_array, int count, double box_length);
int torus_distribution(t_particle **particle_array, int count, double major_r, double minor_r);
int parallel_write_to_file(t_particle *particle_array, int *count, char *filename);
int serial_write_to_file(t_particle *particle_array, int count, char *filename);
int serial_read_from_file(t_particle **particle_array, int *count, char *filename);

#endif
