#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <mpi.h>

#include "helper.hpp"

#define DEFAULT_POWER 3

void parse_args(int argc, char **argv, int *power, dist_type_t *dist_type)
{
    *power = DEFAULT_POWER;
    *dist_type = DIST_UNKNOWN;

    if (argc > 1)
    {
        if (std::strcmp(argv[1], "box") == 0)
            *dist_type = DIST_BOX;
        else if (std::strcmp(argv[1], "torus") == 0)
            *dist_type = DIST_TORUS;
    }

    if (*dist_type == DIST_UNKNOWN)
    {
        *dist_type = DIST_BOX;
    }

    if (argc > 2)
    {
        *power = std::atoi(argv[2]);
    }
}

void distribute_gpu_particles_mpi(t_particle **d_rank_array, int *lens, int *capacity, cudaStream_t stream);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int gpus = 0;
    cudaGetDeviceCount(&gpus);
    int local_dev = rank % (gpus > 0 ? gpus : 1);
    cudaSetDevice(local_dev);

    if (rank == 0)
        std::cout << "Using " << nprocs << " GPUs\n";
    char filename[128];

    int length_per_rank = 0;
    long long total_particles = 0;

    dist_type_t dist_type;
    int power = DEFAULT_POWER;
    double box_length = 0.0;
    int major_r = 0;
    int minor_r = 0;
    double RAM_GB = 0.0;
    int capacity = 0;

    const int block = 256;
    int sms = 0;

    parse_args(argc, argv, &power, &dist_type);

    t_particle *d_rank_array = nullptr;
    t_particle *h_host_array = nullptr;
    cudaStream_t gpu_stream;
    int lens = 0;

    setup_particles_box_length(power, nprocs, rank, &length_per_rank, &box_length, &total_particles, &RAM_GB, &major_r, &minor_r);
    lens = length_per_rank;
    printf("Before distribution %d:  %d\n", rank, lens);

    cudaStreamCreate(&gpu_stream);

    cudaMallocAsync(&d_rank_array, length_per_rank * sizeof(t_particle), gpu_stream);
    cudaMallocHost(&h_host_array, length_per_rank * sizeof(t_particle));
    capacity = length_per_rank;

    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, local_dev);
    int maxBlocks = sms * 20;
    int grid = (length_per_rank + block - 1) / block;
    int seed = rank;
    if (grid > maxBlocks)
        grid = maxBlocks;

    switch (dist_type)
    {
    case DIST_BOX:
        box_distribution_kernel<<<grid, block, 0, gpu_stream>>>(d_rank_array, length_per_rank, box_length, seed);
        break;
    case DIST_TORUS:
        torus_distribution_kernel<<<grid, block, 0, gpu_stream>>>(d_rank_array, length_per_rank, major_r, minor_r, box_length, seed);
        break;
    }

    cudaStreamSynchronize(gpu_stream);
    MPI_Barrier(MPI_COMM_WORLD);

    auto t0 = std::chrono::steady_clock::now();

    generate_keys_kernel<<<grid, block, 0, gpu_stream>>>(d_rank_array, length_per_rank, box_length);

    cudaStreamSynchronize(gpu_stream);
    MPI_Barrier(MPI_COMM_WORLD);

    distribute_gpu_particles_mpi(&d_rank_array, &length_per_rank, &capacity, gpu_stream);

    cudaStreamSynchronize(gpu_stream);
    MPI_Barrier(MPI_COMM_WORLD);

    auto t1 = std::chrono::steady_clock::now();
    double dist_sec = std::chrono::duration<double>(t1 - t0).count();

    lens = length_per_rank;

    if (power < 4)
    {
        if (h_host_array)
        {
            cudaFreeHost(h_host_array);
            h_host_array = nullptr;
        }

        const size_t bytes = static_cast<size_t>(lens) * sizeof(t_particle);
        if (lens > 0)
        {
            cudaMallocHost(&h_host_array, bytes);
            cudaMemcpyAsync(h_host_array, d_rank_array, bytes, cudaMemcpyDeviceToHost, gpu_stream);
        }

        cudaStreamSynchronize(gpu_stream);
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> recv_lens;
        if (rank == 0)
            recv_lens.resize(nprocs);
        MPI_Gather(&lens, 1, MPI_INT, rank == 0 ? recv_lens.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> recv_counts, recv_displs;
        size_t total_count = 0;
        if (rank == 0)
        {
            recv_counts.resize(nprocs);
            recv_displs.resize(nprocs);
            for (int i = 0; i < nprocs; ++i)
            {
                recv_counts[i] = recv_lens[i] * (int)sizeof(t_particle);
            }
            recv_displs[0] = 0;
            for (int i = 1; i < nprocs; ++i)
                recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
            total_count = (size_t)recv_displs.back() + (size_t)recv_counts.back();
        }

        std::vector<unsigned char> gather_buf(rank == 0 ? total_count : 0);
        MPI_Gatherv(d_rank_array, lens * (int)sizeof(t_particle), MPI_BYTE,
                    rank == 0 ? gather_buf.data() : nullptr,
                    rank == 0 ? recv_counts.data() : nullptr,
                    rank == 0 ? recv_displs.data() : nullptr,
                    MPI_BYTE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            sprintf(filename, "particle_file_gpu_n%d_total%lld.par", nprocs, total_particles);
            std::vector<t_particle *> host_ptrs(nprocs, nullptr);
            for (int i = 0; i < nprocs; ++i)
                host_ptrs[i] = reinterpret_cast<t_particle *>(gather_buf.data() + recv_displs[i]);
            int rc = concat_and_serial_write(host_ptrs.data(), recv_lens.data(), nprocs, filename);
            if (rc != 0)
            {
                std::cerr << "Error at writing file, rc=" << rc << "\n";
            }
        }
    }

    log_results(rank, power, total_particles, length_per_rank, nprocs, box_length, RAM_GB, dist_sec, "gpu");

    if (d_rank_array)
        cudaFreeAsync(d_rank_array, gpu_stream);
    if (h_host_array)
        cudaFreeHost(h_host_array);
    cudaStreamDestroy(gpu_stream);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
