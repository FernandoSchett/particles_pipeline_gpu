#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include <mpi.h>

#include "particle_types.hpp"
#include "particles_gpu.hcu"
#include "file_handling.hpp"
#include "utils.hpp"
#include "logging.hpp"

#define DEFAULT_POWER 3

void parse_args(int argc, char **argv, ExecConfig &cfg)
{
    cfg.power = DEFAULT_POWER;
    cfg.seed = DEFAULT_SEED;
    cfg.dist_type = DIST_UNKNOWN;
    cfg.exp_type = STRONG_SCALING;
    if (argc > 1)
    {
        if (strcmp(argv[1], "box") == 0)
            cfg.dist_type = DIST_BOX;
        else if (strcmp(argv[1], "torus") == 0)
            cfg.dist_type = DIST_TORUS;
    }
    if (cfg.dist_type == DIST_UNKNOWN)
        cfg.dist_type = DIST_BOX;
    if (argc > 2)
        cfg.power = std::atoi(argv[2]);
    if (argc > 3)
        cfg.seed = std::atoi(argv[3]);
    if (argc > 4)
    {
        if (strcmp(argv[4], "weak") == 0)
            cfg.exp_type = WEAK_SCALING;
        else if (strcmp(argv[4], "strong") == 0)
            cfg.exp_type = STRONG_SCALING;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    ExecConfig cfg;
    MPI_Comm_rank(MPI_COMM_WORLD, &cfg.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cfg.nprocs);
    cfg.device = "gpu";
    parse_args(argc, argv, cfg);

    int gpus = 0;
    cudaGetDeviceCount(&gpus);
    int local_dev = cfg.rank % (gpus > 0 ? gpus : 1);
    cudaSetDevice(local_dev);

    DBG_RANK_PRINT(cfg.rank, 0, "Using %d GPUs\n", gpus);
    char filename[128];

    setup_particles_box_length(cfg);
    int capacity = cfg.length_per_rank;

    cudaStream_t gpu_stream;
    cudaStreamCreate(&gpu_stream);

    t_particle *d_rank_array = nullptr;
    t_particle *h_host_array = nullptr;

    cudaMallocAsync(&d_rank_array, (size_t)cfg.length_per_rank * sizeof(t_particle), gpu_stream);

    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, local_dev);
    const int block = 256;
    int maxBlocks = sms * 20;
    int grid = (cfg.length_per_rank + block - 1) / block;
    if (grid > maxBlocks)
        grid = maxBlocks;

    switch (cfg.dist_type)
    {
    case DIST_BOX:
        box_distribution_kernel<<<grid, block, 0, gpu_stream>>>(d_rank_array, cfg.length_per_rank, cfg.box_length, cfg.seed + cfg.rank);
        break;
    case DIST_TORUS:
        torus_distribution_kernel<<<grid, block, 0, gpu_stream>>>(d_rank_array, cfg.length_per_rank, cfg.major_r, cfg.minor_r, cfg.box_length, cfg.seed + cfg.rank);
        break;
    default:
        break;
    }

    cudaStreamSynchronize(gpu_stream);
    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    generate_keys_kernel<<<grid, block, 0, gpu_stream>>>(d_rank_array, cfg.length_per_rank, cfg.box_length);

    cudaStreamSynchronize(gpu_stream);
    MPI_Barrier(MPI_COMM_WORLD);
    double t05 = MPI_Wtime();

    if (cfg.nprocs > 1)
    {
        distribute_gpu_particles_mpi(&d_rank_array, &cfg.length_per_rank, &capacity, gpu_stream);
    }

    cudaStreamSynchronize(gpu_stream);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    int lens = cfg.length_per_rank;

    if (cfg.power < 4)
    {
        cudaMallocHost(&h_host_array, (size_t)cfg.length_per_rank * sizeof(t_particle));
        if (h_host_array)
        {
            cudaFreeHost(h_host_array);
            h_host_array = nullptr;
        }

        const size_t bytes = (size_t)lens * sizeof(t_particle);
        if (lens > 0)
        {
            cudaMallocHost(&h_host_array, bytes);
            cudaMemcpyAsync(h_host_array, d_rank_array, bytes, cudaMemcpyDeviceToHost, gpu_stream);
        }

        cudaStreamSynchronize(gpu_stream);
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> recv_lens;
        if (cfg.rank == 0)
            recv_lens.resize(cfg.nprocs);
        MPI_Gather(&lens, 1, MPI_INT, cfg.rank == 0 ? recv_lens.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> recv_counts, recv_displs;
        size_t total_count = 0;
        if (cfg.rank == 0)
        {
            recv_counts.resize(cfg.nprocs);
            recv_displs.resize(cfg.nprocs);
            for (int i = 0; i < cfg.nprocs; ++i)
            {
                recv_counts[i] = recv_lens[i] * (int)sizeof(t_particle);
            }
            recv_displs[0] = 0;
            for (int i = 1; i < cfg.nprocs; ++i)
                recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
            total_count = (size_t)recv_displs.back() + (size_t)recv_counts.back();
        }

        std::vector<unsigned char> gather_buf(cfg.rank == 0 ? total_count : 0);
        MPI_Gatherv(d_rank_array, lens * (int)sizeof(t_particle), MPI_BYTE,
                    cfg.rank == 0 ? gather_buf.data() : nullptr,
                    cfg.rank == 0 ? recv_counts.data() : nullptr,
                    cfg.rank == 0 ? recv_displs.data() : nullptr,
                    MPI_BYTE, 0, MPI_COMM_WORLD);

        if (cfg.rank == 0)
        {
            if (cfg.power < 4)
            {
                std::sprintf(filename, "particle_file_gpu_n%d_total%lld.par", cfg.nprocs, cfg.total_particles);
                std::vector<t_particle *> host_ptrs(cfg.nprocs, nullptr);
                for (int i = 0; i < cfg.nprocs; ++i)
                    host_ptrs[i] = reinterpret_cast<t_particle *>(gather_buf.data() + recv_displs[i]);
                int rc = concat_and_serial_write(host_ptrs.data(), recv_lens.data(), cfg.nprocs, filename);
                if (rc != 0)
                {
                    std::cerr << "Error at writing file, rc=" << rc << "\n";
                }
            }
        }
    }

    if (cfg.rank == 0)
    {
        const char *mode_str = (cfg.exp_type == WEAK_SCALING) ? "weak" : "strong";
        std::string out = std::string("../results_") + mode_str + ".csv";
        log_results(cfg, t05 - t0, t1 - t05, out.c_str());
    }

    if (d_rank_array)
        cudaFreeAsync(d_rank_array, gpu_stream);
    if (h_host_array)
        cudaFreeHost(h_host_array);
    cudaStreamDestroy(gpu_stream);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
