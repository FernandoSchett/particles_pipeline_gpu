#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <set>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <time.h>
#include <sys/stat.h>
#include <cuda_runtime.h> 

#include "helper.hpp"

#define DEFAULT_POWER 3

void setup_particles_box_length(int power, int nprocs, int rank, int *length_per_rank, double *box_length, long long *total_particles, double *RAM_GB, int *major_r, int *minor_r) {
    long long slice = (long long)(pow(10,power) / nprocs);
    *total_particles = ((1 + nprocs) * nprocs / 2) * slice;
    *RAM_GB = (*total_particles * 40.0) / 1e9; // sizeof(t_particle) is 36, but it can be considered as 40.
    *box_length = pow(10, power);
    *length_per_rank = (rank+1) * slice;
    *major_r = 4 * pow(10, power-1);
    *minor_r = 2 * pow(10, power-1);


    if (rank == nprocs - 1) {
        *length_per_rank += *total_particles % nprocs;
    }

    if (rank == 0){ 
        printf("%lld particles distributed like (rank_number*%d) between %d processes in a %.1f sized box using %.4f GBs.\n", *total_particles, slice, nprocs, *box_length, *RAM_GB); 
    }
}

void log_results(int rank, int power, long long total_particles, int length_per_rank, int nprocs, double box_length, double RAM_GB, double execution_time) {
    time_t rawtime;
    struct tm *timeinfo;
    char time_str[64];

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", timeinfo);

    struct stat buffer;
    int file_exists = (stat("../results.csv", &buffer) == 0);

    FILE *f = fopen("../../results.csv", "a");

    if (!file_exists) {
        fprintf(f, "datetime,power,total_particles,length_per_rank,num_procs,box_length,RAM_GB,execution_time,device\n");
    }

    fprintf(f, "%s,%d,%lld,%d,%d,%.1f,%.2f,%f,gpu\n", 
            time_str, power, total_particles, length_per_rank, nprocs,
            box_length, RAM_GB, execution_time);
    printf("%s,%d,%lld,%d,%d,%.1f,%.2f,%f,gpu\n", 
            time_str, power, total_particles, length_per_rank, nprocs,
            box_length, RAM_GB, execution_time);


    fclose(f);
}

void parse_args(int argc, char **argv, int *power, dist_type_t *dist_type) {
    *power = DEFAULT_POWER;
    *dist_type = DIST_UNKNOWN;

    if (argc > 1) {
        if (strcmp(argv[1], "box") == 0)
        *dist_type = DIST_BOX;
        else if (strcmp(argv[1], "torus") == 0)
        *dist_type = DIST_TORUS;
    }
    
    if (*dist_type == DIST_UNKNOWN) {
        *dist_type = DIST_BOX;
    }
    
    if (argc > 2) {
        *power = atoi(argv[2]);
    }
}

int main(int argc, char **argv){
    int rank = 0, nprocs=1;
    int length_per_rank, total_length;
    long long total_particles;
    int *length_vector;
    t_particle *rank_array, *host_array;
    dist_type_t dist_type;
    int power;
    double box_length;
    int major_r, minor_r;
    double start_time, end_time;
    char filename[128];
    double RAM_GB;
    float gen_ms = 0.0f;
    double kernel_time_sec;


    parse_args(argc, argv, &power, &dist_type);
    setup_particles_box_length(power, nprocs, rank, &length_per_rank, &box_length, &total_particles, &RAM_GB, &major_r , &minor_r);


    // aloca particulas na gpu.
    cudaMalloc(&rank_array, length_per_rank*sizeof(t_particle));     
    cudaMallocHost(&host_array, length_per_rank*sizeof(t_particle));

    // allocate_particle(&rank_array, length_per_rank);


    // cria as coordenadas na gpu.
    

    unsigned long long seed = (unsigned long long)time(nullptr) ^ (0x9E3779B97F4A7C15ull * (unsigned long long)(rank + 1));
    cudaStream_t s = 0;
    int block = 256;
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int maxBlocks = sms * 20;
    int grid = (length_per_rank + block - 1) / block;
    if (grid > maxBlocks) grid = maxBlocks;
    
    switch(dist_type){
        case DIST_BOX:
        box_distribution_kernel<<<grid, block, 0, s>>>(rank_array, length_per_rank, box_length, seed);
        break;
        case DIST_TORUS:
        torus_distribution_kernel<<<grid, block, 0, s>>>(rank_array, length_per_rank, major_r, minor_r, box_length, seed);
        break;
    }
    
    // cria as keys na gpu.
    //generate_particles_keys(rank_array, length_per_rank, box_length);
    cudaEvent_t kStart, kStop;


    cudaEventCreate(&kStart);
    cudaEventCreate(&kStop);

    cudaEventRecord(kStart, s);
    generate_keys_kernel<<<grid, block, 0, s>>>(rank_array, length_per_rank, box_length);
    cudaEventRecord(kStop, s);

    cudaEventSynchronize(kStop);

    cudaEventElapsedTime(&gen_ms, kStart, kStop);
    kernel_time_sec = gen_ms / 1000.0;


    cudaDeviceSynchronize();
    cudaMemcpy(host_array, rank_array, length_per_rank * sizeof(t_particle), cudaMemcpyDeviceToHost);
    print_particles(host_array, length_per_rank, rank);
    
    
    // distribui as keys na gpu.
    //distribute_particles(&rank_array, &length_per_rank, nprocs);
    
    // cada gpu manda pro host 

    // host escreve em paralelo.
    //sprintf(filename, "particle_file_n%d_total%lld", nprocs, total_particles);
    //parallel_write_to_file(rank_array, length_vector, filename);
    
    if(rank == 0)
        log_results(rank, power, total_particles, length_per_rank, nprocs, box_length, RAM_GB, kernel_time_sec);    
    
    cudaEventDestroy(kStart);
    cudaEventDestroy(kStop);
    
    cudaFreeHost(host_array);
    cudaFree(rank_array);
    
    return 0;
}
