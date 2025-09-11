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

#include "particle_types.hpp"
#include "particles_cpu.hpp"
#include "file_handling.hpp"
#include "utils.hpp"
#include "logging.hpp"

#define DEFAULT_POWER 3

void parse_args(int argc, char **argv, int *power, dist_type_t *dist_type)
{
    *power = DEFAULT_POWER;
    *dist_type = DIST_UNKNOWN;

    if (argc > 1)
    {
        if (strcmp(argv[1], "box") == 0)
            *dist_type = DIST_BOX;
        else if (strcmp(argv[1], "torus") == 0)
            *dist_type = DIST_TORUS;
    }

    if (*dist_type == DIST_UNKNOWN)
    {
        *dist_type = DIST_BOX;
    }

    if (argc > 2)
    {
        *power = atoi(argv[2]);
    }
}

int main(int argc, char **argv)
{
    int rank, nprocs;
    int length_per_rank, total_length;
    long long total_particles;
    int *length_vector;
    t_particle *rank_array;
    dist_type_t dist_type;
    int power;
    double box_length;
    int major_r, minor_r;
    double start_time, end_time;
    char filename[128];
    double RAM_GB;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // You can now use MPI_particle as an input to MPI_Datatype during MPI calls.
    register_MPI_Particle(&MPI_particle);

    parse_args(argc, argv, &power, &dist_type);
    length_vector = (int *)malloc(nprocs * sizeof(int));

    setup_particles_box_length(power, nprocs, rank, &length_per_rank, &box_length, &total_particles, &RAM_GB, &major_r, &minor_r);

    // Everybody need to know howm much much particles each other have.
    MPI_Allgather(&length_per_rank, 1, MPI_INT, length_vector, 1, MPI_INT, MPI_COMM_WORLD);
    allocate_particle(&rank_array, length_per_rank);

    switch (dist_type)
    {
    case DIST_BOX:
        box_distribution(&rank_array, length_per_rank, box_length);
        break;
    case DIST_TORUS:
        torus_distribution(&rank_array, length_per_rank, major_r, minor_r, box_length);
        break;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    generate_particles_keys(rank_array, length_per_rank, box_length);
    distribute_particles(&rank_array, &length_per_rank, nprocs);

    // if(rank == 0){
    //     print_particles(rank_array, length_per_rank, 0);
    // }

    // distribute_particles(&rank_array, &length_per_rank, nprocs);

    // Update length_vector
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    MPI_Allgather(&length_per_rank, 1, MPI_INT, length_vector, 1, MPI_INT, MPI_COMM_WORLD);
    if(power < 4){
        sprintf(filename, "particle_file_cpu_n%d_total%lld.par", nprocs, total_particles); 
        parallel_write_to_file(rank_array, length_vector, filename);
    }

    if (rank == 0)
        log_results(rank, power, total_particles, length_per_rank, nprocs, box_length, RAM_GB, end_time - start_time, "cpu");

    total_length = 0;
    for (int i = 0; i < nprocs; i++)
    {
        total_length += length_vector[i];
    }

    free(rank_array);
    free(length_vector);
    MPI_Finalize();
    return 0;
}
