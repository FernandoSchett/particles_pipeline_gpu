#ifndef P_SFC_PARTICLES_CPU_HPP
#define P_SFC_PARTICLES_CPU_HPP

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

int box_distribution(t_particle **particle_array, int count, double box_length, int seed);
int triangle_distribution(t_particle **particle_array, int count, double box_length, int seed);
int torus_distribution(t_particle **particle_array, int count, double major_r, double minor_r, double box_length, int seed);

/** @brief Read PEPC coordinates on the CPU and update the shared Morton domain. */
int pepc_distribution_cpu(t_particle **particles, ExecConfig &cfg, MPI_Comm comm);

int allocate_particle(t_particle **particle_array, int count);
int generate_particles_keys(t_particle *particle_array, int count, double box_length,
                            double domain_x = 0.0, double domain_y = 0.0, double domain_z = 0.0);
void sort_particles_by_key_cpu(t_particle *particles, int count);

void discover_splitters_cpu(t_particle *particles, int local_n, std::vector<unsigned long long> &splitters_out);

int redistribute_by_splitters_cpu(t_particle **particles, int *particle_vector_size, const std::vector<unsigned long long> &splitters);

int validate_mpi_key_boundaries(const t_particle *particles,
                                int particle_count,
                                MPI_Comm communicator);

void write_par_cpu(const ExecConfig &cfg, t_particle *rank_array, int *length_vector);
#endif
