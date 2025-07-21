#include <cstdlib>
#include <list>
#include <set>
#include <vector>
#include <cmath>
#include <time.h>
#include <iostream>

#include "./helper.hpp"

int main(int argc, char **argv){
    int rank, nprocs, i;
    int length_per_rank, total_length;
    int *length_vector, *disp;
    double box_length;
    t_particle *rank_array, *receive_array;
    char filename[128] = "particle_file";
    char filename2[128] = "serial_particle_file";

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    register_MPI_Particle(&MPI_particle);
    // You can now use MPI_particle as an input to MPI_Datatype during MPI calls.

    length_per_rank = (rank+1)*100;
    box_length = 100;

    length_vector = (int *)malloc(nprocs*sizeof(int));
    MPI_Allgather(&length_per_rank, 1, MPI_INT, length_vector, 1, MPI_INT, MPI_COMM_WORLD);

    allocate_particle(&rank_array, length_per_rank);
    box_distribution(&rank_array, length_per_rank, box_length);

/*
    for (int i = 0; i < length_per_rank; i++){
        printf("P_rank: %d, %d, %f, %f, %f\n", rank, rank_array[i].mpi_rank, rank_array[i].coord[0], \
			rank_array[i].coord[1], rank_array[i].coord[2]);
    }
*/

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

    free(rank_array);
    free(receive_array);
    free(length_vector);
    free(disp);

    total_length = 0;
    if (rank == 0){
        serial_read_from_file(&rank_array, &total_length, filename);
        /* for (int i = 0; i < total_length; i++){
            printf("Q_rank: %d, %d, %f, %f, %f\n", rank, rank_array[i].mpi_rank, rank_array[i].coord[0], \
	    		rank_array[i].coord[1], rank_array[i].coord[2]);
        }*/
        free(rank_array);
    }

    MPI_Finalize();
    return 0;
}


