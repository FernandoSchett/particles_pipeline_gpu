#ifndef FILEHANFLING_H
#define FILEHANFLING_H

#include <mpi.h>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <ctime>
#include <limits>

#include "particle_types.hpp"

int parallel_write_to_file(t_particle *particle_array, int *count, char *filename);
int serial_write_to_file(t_particle *particle_array, int count, char *filename);
int serial_read_from_file(t_particle **particle_array, int *count, char *filename);

int concat_and_serial_write(t_particle **arrays, const int *counts, int nprocs, const char *filename);

int allocate_particle(t_particle **particle_array, int count);

#endif