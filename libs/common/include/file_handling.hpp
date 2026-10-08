#ifndef P_SFC_FILE_HANDLING_HPP
#define P_SFC_FILE_HANDLING_HPP

#include <mpi.h>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <ctime>
#include <limits>
#include <iostream>

#include "particle_types.hpp"
#include "logging.hpp"

extern MPI_Datatype MPI_particle;
extern MPI_Datatype MPI_particle_data;
extern MPI_Datatype MPI_particle_results;
extern MPI_Datatype MPI_particle_pepc;

int register_MPI_Particle(MPI_Datatype *MPI_Particle);

int parallel_write_to_file(t_particle *particle_array, int *count, char *filename);
int serial_write_to_file(t_particle *particle_array, int count, char *filename);
int serial_read_from_file(t_particle **particle_array, int *count, char *filename);

int concat_and_serial_write(t_particle **arrays, const int *counts, int nprocs, const char *filename);

int register_MPI_particle_data(MPI_Datatype *MPI_Particle_data);

int register_MPI_particle_results(MPI_Datatype *MPI_Particle_results);

int register_MPI_Particle_pepc(MPI_Datatype *MPI_Particle_pepc);

int parallel_read_pepc_particles(MPI_Comm comm, t_pepc_particle **pepc_array, \
                             char *filename, int64_t *n_total, int64_t *n_local);

void pepc_to_simple_particle_array(int rank, t_pepc_particle **pepc_array, t_particle **particle_array, int64_t n_local);

#endif
