#ifndef PARTICLESCPU_H
#define PARTICLESCPU_H

#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <mpi.h>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>
#include <string.h>

#include <boost/sort/spreadsort/integer_sort.hpp>

#include "particle_types.hpp"
#include "logging.hpp"
#include "file_handling.hpp"

extern MPI_Datatype MPI_particle;
int register_MPI_Particle(MPI_Datatype *MPI_Particle);

int box_distribution(t_particle **particle_array, int count, double box_length, int seed);
int torus_distribution(t_particle **particle_array, int count, double major_r, double minor_r, double box_length, int seed);

void run_oct_tree_recursive(std::vector<t_particle *> &particles, int depth, long long key_prefix, double box_length, const std::array<double, 3> &origin);
int generate_particles_keys(t_particle *particle_array, int count, double box_length);

void discover_splitters_cpu(t_particle *particles, int local_n, std::vector<unsigned long long> &splitters_out);

int redistribute_by_splitters_cpu(t_particle **particles, int *particle_vector_size, const std::vector<unsigned long long> &splitters);

void write_par_cpu(const ExecConfig &cfg, t_particle *rank_array, int *length_vector);
#endif