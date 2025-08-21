#include <cstdlib>
#include <list>
#include <set>
#include <vector>
#include <cmath>
#include <time.h>
#include <iostream>

#include "./helper.hpp"

#define DEFAULT_POWER 2

void setup_particles_box_length(int power, int nprocs, int rank, int *length_per_rank, double *box_length) {
    int total_particles  = pow(10, power);
    *box_length       = pow(10, power);
    *length_per_rank  = (rank+1)*(total_particles / nprocs);

    if (rank == nprocs - 1) {
        *length_per_rank += total_particles % nprocs;
    }

    if (rank == 0) {
        printf("%d particles distributed like (rank_number*%d) between %d processes in a %.1f sized box.\n",
               total_particles, *length_per_rank, nprocs, *box_length);
    }
}

int main(int argc, char **argv){
    int rank, nprocs, i;
    int length_per_rank, total_length, total_particles;
    int *length_vector, *disp;
    int power;
    double box_length;
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
    generate_particles_keys(rank_array, length_per_rank, box_length);
    distribute_particles(&rank_array, &length_per_rank, nprocs);

    // Everybody need to know howm much much particles each other have. 
    MPI_Allgather(&length_per_rank, 1, MPI_INT, length_vector, 1, MPI_INT, MPI_COMM_WORLD);

    parallel_write_to_file(rank_array, length_vector, filename);

    total_length = 0;
    for (i = 0; i < nprocs; i++){
        total_length += length_vector[i];
    }
    
    receive_array = (t_particle *)malloc(total_length*sizeof(t_particle));
    disp = (int *)malloc(nprocs*sizeof(int));
    disp[0] = 0;
    for (i = 1; i < nprocs; i++){
        disp[i] = disp[i-1] + length_vector[i-1];
    }
    MPI_Gatherv(rank_array, length_per_rank, MPI_particle, receive_array, length_vector, disp, MPI_particle, 0, MPI_COMM_WORLD); 

    if (rank == 0) serial_write_to_file(receive_array, total_length, filename2);
    total_length = 0;
    if (rank == 0){
        serial_read_from_file(&rank_array, &total_length, filename);
    }

    free(rank_array);
    free(receive_array);
    free(length_vector);
    free(disp);
    MPI_Finalize();
    return 0;
}


