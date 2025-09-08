#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstring>

#include <cstdlib>
#include <cmath>
#include <ctime>

#include <sys/stat.h>

#include <cuda_runtime.h>
#include <mpi.h>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include <boost/sort/spreadsort/integer_sort.hpp>

#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
        {                                                                                   \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit(cudaStatus);                                                               \
        }                                                                                   \
    }

typedef struct particle
{
    int mpi_rank;
    long long int key;
    double coord[3];
} t_particle;

struct key_less
{
    __host__ __device__ bool operator()(const t_particle &a, const t_particle &b) const
    {
        return (unsigned long long)a.key < (unsigned long long)b.key;
    }
};

typedef enum
{
    DIST_BOX,
    DIST_TORUS,
    DIST_UNKNOWN
} dist_type_t;

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

int distribute_particles(t_particle **particles, int *particle_vector_size, int nprocs);

void print_particles(t_particle *particle_array, int size, int rank);

void setup_particles_box_length(int power, int nprocs, int rank, int *length_per_rank, double *box_length, long long *total_particles, double *RAM_GB, int *major_r, int *minor_r);

void log_results(int rank, int power, long long total_particles, int length_per_rank, int nprocs, double box_length, double RAM_GB, double execution_time, const char *device_type);

// GPU Kernels
__global__ void box_distribution_kernel(t_particle *particles, int N, double L, unsigned long long seed);
__global__ void torus_distribution_kernel(t_particle *particles, int N, double major_r, double minor_r, double box_length, unsigned long long seed);
__global__ void generate_keys_kernel(t_particle *particles, int N, double box_length);
__global__ void set_rank_kernel(t_particle *p, int n, int rank_id);

// GPU Utils
void gpu_barrier(int nprocs, const std::vector<cudaStream_t> &streams);
void enable_p2p_all(int ndev);
int concat_and_serial_write(t_particle **arrays, const int *counts, int nprocs, const char *filename);
int distribute_gpu_particles(std::vector<t_particle *> &d_rank_array, std::vector<int> &lens, std::vector<cudaStream_t> &gpu_streams);
void compute_cuts_for_dev(int dev, t_particle *d_ptr, int n, const std::vector<unsigned long long> &splitters, std::vector<int> &cuts_out, cudaStream_t stream);
long long count_leq_device(int dev, t_particle *d_ptr, int n, unsigned long long mid, cudaStream_t stream);

void distribute_gpu_particles_mpi(t_particle **d_rank_array, int *lens, int *capacity, cudaStream_t stream);

#endif
