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

void setup_particles_box_length(ExecConfig &cfg);

void log_results(const ExecConfig &cfg, const exec_times &times, const char *results_path);

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
