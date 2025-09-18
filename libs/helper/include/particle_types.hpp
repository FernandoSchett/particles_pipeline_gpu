#ifndef PARTICLESTYPES_H
#define PARTICLESTYPES_H

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

typedef enum
{
  DIST_BOX,
  DIST_TORUS,
  DIST_UNKNOWN
} dist_type_t;

#define NPROPS_PARTICLE 3
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
