#include <cstdlib>
#include <list>
#include <set>
#include <vector>
#include <cmath>
#include <time.h>
#include <iostream>

#include "./helper.hpp"

#define DEFAULT_POWER 3

void setup_particles_box_length(int power, int nprocs, int rank, int *length_per_rank, double *box_length) {
    long long slice = (int) (pow(10,power) / nprocs);
    int total_particles = ((1 + nprocs)*nprocs/2)*slice;
   long long RAM_particles = total_particles * 40; // sizeof(t_particle) is 36, but it can be considered as 40.
    *box_length       = pow(10, power); 
    *length_per_rank  = (rank+1)*slice;

    if (rank == nprocs - 1) {
        *length_per_rank += total_particles % nprocs;
    }

    if (rank == 0) {
        printf("%d particles distributed like (rank_number*%d) between %d processes in a %.1f sized box using %f GBs.\n",
               total_particles, *length_per_rank, nprocs, *box_length, RAM_particles / 1e9);
    }
}

int main(int argc, char **argv){
    int rank, nprocs;
    int length_per_rank, total_length;
    int *length_vector, *disp;
    int power;
    double box_length;
    double start_time, end_time;
    t_particle *rank_array, *receive_array;
    char filename[128] = "particle_file";
    char filename2[128] = "serial_particle_file";

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
    
    setup_particles_box_length(power, nprocs, rank, &length_per_rank, &box_length);

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
    
    if (rank == 0)
        printf("Time to generate keys + distribute particles: %f\n", end_time - start_time);

    
    parallel_write_to_file(rank_array, length_vector, filename);
    total_length = 0;
    for (int i = 0; i < nprocs; i++){
        total_length += length_vector[i];
    }
    


    free(rank_array);
    free(length_vector);
    MPI_Finalize();
    return 0;
}


