#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>
//#include <boost/sort/sort.hpp>

typedef struct particle{
    int mpi_rank;
    long long int key;
    double coord[3];
} t_particle;

typedef enum {DIST_BOX, DIST_TORUS, DIST_UNKNOWN} dist_type_t;

#define NPROPS_PARTICLE 3
#define MAX_DEPTH 15

extern MPI_Datatype MPI_particle;
extern int register_MPI_Particle(MPI_Datatype *MPI_Particle);

int allocate_particle(t_particle **particle_array, int count);
int box_distribution(t_particle **particle_array, int count, double box_length);
int torus_distribution(t_particle **particle_array, int count, double major_r, double minor_r, double box_length);

int parallel_write_to_file(t_particle *particle_array, int *count, char *filename);
int serial_write_to_file(t_particle *particle_array, int count, char *filename);
int serial_read_from_file(t_particle **particle_array, int *count, char *filename);

void radix_sort_particles(t_particle *particles, int n);

void run_oct_tree_recursive(t_particle *particles, int count, int depth, long long key_prefix, double box_length, double origin[3]);
int generate_particles_keys(t_particle *particle_array, int count, double box_length);

int distribute_particles(t_particle **particles, int* particle_vector_size, int nprocs);
int distribute_particles_right(t_particle **particles, int *particle_vector_size, int nprocs);

void print_particles(t_particle *particle_array, int size, int rank);


#endif
