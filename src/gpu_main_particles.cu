#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <vector>
#include <cuda_runtime.h>

#include "helper.hpp"

#define DEFAULT_POWER 3

void setup_particles_box_length(int power,
                                int nprocs,
                                int rank,
                                int* length_per_rank,
                                double* box_length,
                                long long* total_particles,
                                double* RAM_GB,
                                int* major_r,
                                int* minor_r)
{
    const long long slice = static_cast<long long>(std::pow(10, power) / nprocs);
    *total_particles = ((1 + nprocs) * nprocs / 2) * slice;
    *RAM_GB = (*total_particles * 40.0) / 1e9; // sizeof(t_particle) is 36, but it can be considered as 40.
    *box_length = std::pow(10, power);
    *length_per_rank = (rank + 1) * slice;
    *major_r = 4 * std::pow(10, power - 1);
    *minor_r = 2 * std::pow(10, power - 1);

    if (rank == nprocs - 1) {
        *length_per_rank += *total_particles % nprocs;
    }

    if (rank == 0) {
        std::printf("%lld particles distributed like (rank_number*%lld) between %d processes in a %.1f sized box using %.4f GBs.\n",
                    *total_particles, slice, nprocs, *box_length, *RAM_GB);
    }
}

void log_results(int rank,
                 int power,
                 long long total_particles,
                 int length_per_rank,
                 int nprocs,
                 double box_length,
                 double RAM_GB,
                 double execution_time)
{
    time_t rawtime;
    std::tm* timeinfo;
    char time_str[64];

    std::time(&rawtime);
    timeinfo = std::localtime(&rawtime);
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", timeinfo);

    struct stat buffer;
    const char* results_path = "../../results.csv";
    const int file_exists = (stat(results_path, &buffer) == 0);

    FILE* f = std::fopen(results_path, "a");
    if (!file_exists) {
        std::fprintf(f, "datetime,power,total_particles,length_per_rank,num_procs,box_length,RAM_GB,execution_time,device\n");
    }

    std::fprintf(f, "%s,%d,%lld,%d,%d,%.1f,%.2f,%f,gpu\n",
                 time_str, power, total_particles, length_per_rank, nprocs,
                 box_length, RAM_GB, execution_time);
    std::printf("%s,%d,%lld,%d,%d,%.1f,%.2f,%f,gpu\n",
                time_str, power, total_particles, length_per_rank, nprocs,
                box_length, RAM_GB, execution_time);

    std::fclose(f);
}

void parse_args(int argc, char** argv, int* power, dist_type_t* dist_type)
{
    *power = DEFAULT_POWER;
    *dist_type = DIST_UNKNOWN;

    if (argc > 1) {
        if (std::strcmp(argv[1], "box") == 0)
            *dist_type = DIST_BOX;
        else if (std::strcmp(argv[1], "torus") == 0)
            *dist_type = DIST_TORUS;
    }

    if (*dist_type == DIST_UNKNOWN) {
        *dist_type = DIST_BOX;
    }

    if (argc > 2) {
        *power = std::atoi(argv[2]);
    }
}

int main(int argc, char** argv)
{
    int rank = 0;
    int nprocs = 1;

    int length_per_rank = 0;
    long long total_particles = 0;

    t_particle* rank_array = nullptr;
    t_particle* host_array = nullptr;

    dist_type_t dist_type;
    int power = DEFAULT_POWER;
    double box_length = 0.0;
    int major_r = 0;
    int minor_r = 0;
    double RAM_GB = 0.0;

    float gen_ms = 0.0f;
    double kernel_time_sec = 0.0;

    parse_args(argc, argv, &power, &dist_type);
    setup_particles_box_length(power, nprocs, rank, &length_per_rank, &box_length, &total_particles, &RAM_GB, &major_r, &minor_r);

    // aloca particulas na gpu.
    cudaMalloc(&rank_array, static_cast<size_t>(length_per_rank) * sizeof(t_particle));
    cudaMallocHost(&host_array, static_cast<size_t>(length_per_rank) * sizeof(t_particle));

    const unsigned long long seed =
        static_cast<unsigned long long>(std::time(nullptr)) ^ (0x9E3779B97F4A7C15ull * static_cast<unsigned long long>(rank + 1));

    cudaStream_t queue = 0;
    const int block = 256;
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int maxBlocks = sms * 20;
    int grid = (length_per_rank + block - 1) / block;
    if (grid > maxBlocks) grid = maxBlocks;

    switch (dist_type) {
        case DIST_BOX:
            box_distribution_kernel<<<grid, block, 0, queue>>>(rank_array, length_per_rank, box_length, seed);
            break;
        case DIST_TORUS:
            torus_distribution_kernel<<<grid, block, 0, queue>>>(rank_array, length_per_rank, major_r, minor_r, box_length, seed);
            break;
        default:
            break;
    }

    // cria as keys na gpu.
    //generate_particles_keys(rank_array, length_per_rank, box_length);
    cudaEvent_t kStart, kStop;
    cudaEventCreate(&kStart);
    cudaEventCreate(&kStop);

    cudaEventRecord(kStart, queue);
    generate_keys_kernel<<<grid, block, 0, queue>>>(rank_array, length_per_rank, box_length);
    cudaEventRecord(kStop, queue);
    cudaEventSynchronize(kStop);

    cudaEventElapsedTime(&gen_ms, kStart, kStop);
    kernel_time_sec = gen_ms / 1000.0;

    cudaDeviceSynchronize();
    cudaMemcpy(host_array, rank_array, static_cast<size_t>(length_per_rank) * sizeof(t_particle), cudaMemcpyDeviceToHost);
    //print_particles(host_array, length_per_rank, rank);

    // distribui as keys na gpu.
    //distribute_particles(&rank_array, &length_per_rank, nprocs);

    // cada gpu manda pro host

    // host escreve em paralelo.
    //sprintf(filename, "particle_file_n%d_total%lld", nprocs, total_particles);
    //parallel_write_to_file(rank_array, length_vector, filename);

    if (rank == 0) {
        log_results(rank, power, total_particles, length_per_rank, nprocs, box_length, RAM_GB, kernel_time_sec);
    }

    cudaEventDestroy(kStart);
    cudaEventDestroy(kStop);

    cudaFreeHost(host_array);
    cudaFree(rank_array);

    return 0;
}
