#ifndef PARTICLESCPU_H
#define PARTICLESCPU_H

#include <array>
#include <vector>
#include "particle_types.hpp"

#include <algorithm>
#include <limits>
#include <cmath>

int box_distribution(t_particle **particle_array, int count, double box_length);
int torus_distribution(t_particle **particle_array, int count, double major_r, double minor_r, double box_length);

void run_oct_tree_recursive(std::vector<t_particle *> &particles, int depth, long long key_prefix, double box_length, const std::array<double, 3> &origin);
int generate_particles_keys(t_particle *particle_array, int count, double box_length);

int distribute_particles(t_particle **particles, int *particle_vector_size, int nprocs);

#endif