#include <cstdlib>
#include <list>
#include <set>
#include <vector>
#include <cmath>
#include <time.h>
#include <ctime>
#include <iostream>
#include <fstream>

#include <sys/stat.h>
#include "./helper.hpp"

#define DEFAULT_POWER 3

void setup_particles_box_length(int power, int nprocs, int rank, int *length_per_rank, double *box_length, long long *total_particles, double *RAM_GB) {
    long long slice = (long long)(pow(10,power) / nprocs);
    *total_particles = ((1 + nprocs) * nprocs / 2) * slice;
    *RAM_GB = (*total_particles * 40.0) / 1e9; // sizeof(t_particle) is 36, but it can be considered as 40.
    *box_length = pow(10, power);
    *length_per_rank = (rank+1) * slice;

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

    FILE *f = fopen("../results.csv", "a");
    if (!f) {
        perror("Erro ao abrir arquivo para escrita");
        return;
    }

    if (!file_exists) {
        fprintf(f, "datetime,power,total_particles,length_per_rank,num_procs,box_length,RAM_GB,execution_time\n");
    }

    fprintf(f, "%s,%d,%lld,%d,%d,%.1f,%.2f,%f\n", 
            time_str, power, total_particles, length_per_rank, nprocs,
            box_length, RAM_GB, execution_time);
    printf("%s,%d,%lld,%d,%d,%.1f,%.2f,%f\n", 
            time_str, power, total_particles, length_per_rank, nprocs,
            box_length, RAM_GB, execution_time);


    fclose(f);
}


int main(int argc, char **argv){
    int rank, nprocs;
    int length_per_rank, total_length;
    int *length_vector;
    int power;
    double box_length;
    double start_time, end_time;
    t_particle *rank_array;
    char filename[128];

    long long total_particles;
    double RAM_GB;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // You can now use MPI_particle as an input to MPI_Datatype during MPI calls.
    register_MPI_Particle(&MPI_particle);

    power = DEFAULT_POWER; 
    if (argc > 1) {
        power = atoi(argv[1]);
    }

    length_vector = (int *)malloc(nprocs*sizeof(int));

    setup_particles_box_length(power, nprocs, rank, &length_per_rank, &box_length, &total_particles, &RAM_GB);

    // Everybody need to know howm much much particles each other have. 
    MPI_Allgather(&length_per_rank, 1, MPI_INT, length_vector, 1, MPI_INT, MPI_COMM_WORLD);

    allocate_particle(&rank_array, length_per_rank);
    box_distribution(&rank_array, length_per_rank, box_length);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    generate_particles_keys(rank_array, length_per_rank, box_length);
    distribute_particles(&rank_array, &length_per_rank, nprocs);
    
    // Update length_vector
    MPI_Allgather(&length_per_rank, 1, MPI_INT, length_vector, 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    sprintf(filename, "particle_file_n%d_total%lld", nprocs, total_particles);
    parallel_write_to_file(rank_array, length_vector, filename);
    
    if(rank == 0)
        log_results(rank, power, total_particles, length_per_rank, nprocs, box_length, RAM_GB, end_time - start_time);
    
    total_length = 0;
    for (int i = 0; i < nprocs; i++){
        total_length += length_vector[i];
    }

    free(rank_array);
    free(length_vector);
    MPI_Finalize();
    return 0;
}
