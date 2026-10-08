#ifndef P_SFC_PARTICLE_TYPES_HPP
#define P_SFC_PARTICLE_TYPES_HPP

#include <cstdint>
#include <cstddef>

#if !defined(__CUDACC__)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

typedef struct particle
{
  int mpi_rank;
  long long int key;
  double coord[3];
} t_particle;

typedef struct
{
  double q;
  double v[3];
  double m;
  double b[3];
  double f_e[3];
  double f_b[3];
  int species;
  int mp_int1;
  double age;
} t_pepc_particle_data;

typedef struct
{
  double e[3];
  double pot;
} t_pepc_particle_results;

typedef struct
{
  double x[3];
  double work;
  int64_t key;
  int64_t node_leaf;
  int64_t label;
  t_pepc_particle_data data;
  t_pepc_particle_results results;
} t_pepc_particle;

typedef enum
{
  DIST_BOX,
  DIST_TORUS,
  DIST_TRIANGLE,
  DIST_PEPC,
  DIST_UNKNOWN
} dist_type_t;

typedef enum
{
  WEAK_SCALING,
  STRONG_SCALING,
} exp_type_t;

typedef enum
{
  GLOBAL_SORTING,
  BUILD_TABLE,
} exp_alg_dist;

typedef struct
{
  int rank;
  int nprocs;
  int power;
  int seed;
  dist_type_t dist_type;
  exp_type_t exp_type;
  exp_alg_dist alg_type;
  const char *device;
  double box_length;
  int major_r;
  int minor_r;
  long long total_particles;
  int length_per_rank;
  double ram_gb;
} ExecConfig;

typedef struct
{
  double alloc_time;
  double gen_time;
  double splitters_time;
  double dist_time;
  double tree_time;
  double total_time;
} exec_times;

#define NPROPS_PARTICLE 3
#define NPROPS_PEPC_PARTICLE_DATA 9
#define NPROPS_PEPC_PARTICLE_RESULTS 2
#define NPROPS_PEPC_PARTICLE 7
#define MAX_DEPTH 15
#define DEFAULT_SEED 24

struct key_less
{
  __host__ __device__ inline bool operator()(const t_particle &a, const t_particle &b) const
  {
    return (unsigned long long)a.key < (unsigned long long)b.key;
  }
};

#endif
