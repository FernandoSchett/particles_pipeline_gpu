#ifndef PUTILS_H
#define PUTILS_H

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <cmath>
#include <vector>
#include <array>

#include "particle_types.hpp"

void print_particles(t_particle *particle_array, int size, int rank);

void setup_particles_box_length(int power, int nprocs, int rank, int *length_per_rank, double *box_length, long long *total_particles, double *RAM_GB, int *major_r, int *minor_r);

void log_results(int rank, int power, long long total_particles, int length_per_rank, int nprocs, double box_length, double RAM_GB, double execution_time, const char *device_type);

static inline bool key_less(const t_particle &a, const t_particle &b);

static inline long long count_leq(const t_particle *particles, int n, unsigned long long val);

struct particle_less
{
    inline bool operator()(const t_particle &a, const t_particle &b) const;
};
struct particle_rightshift
{
    inline unsigned long long operator()(const t_particle &p, unsigned offset) const;
};

#endif
