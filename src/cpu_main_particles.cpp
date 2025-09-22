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
    cfg.device = "cpu";
    parse_args(argc, argv, cfg);

    register_MPI_Particle(&MPI_particle);
    int *length_vector = (int *)std::malloc(cfg.nprocs * sizeof(int));
    t_particle *rank_array = nullptr;

    setup_particles_box_length(cfg);
    MPI_Allgather(&cfg.length_per_rank, 1, MPI_INT, length_vector, 1, MPI_INT, MPI_COMM_WORLD);
    allocate_particle(&rank_array, cfg.length_per_rank);

    switch (cfg.dist_type)
    {
    case DIST_BOX:
        box_distribution(&rank_array, cfg.length_per_rank, cfg.box_length, cfg.seed + cfg.rank);
        break;
    case DIST_TORUS:
        torus_distribution(&rank_array, cfg.length_per_rank, cfg.major_r, cfg.minor_r, cfg.box_length, cfg.seed + cfg.rank);
        break;
    default:
        break;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    generate_particles_keys(rank_array, cfg.length_per_rank, cfg.box_length);
    MPI_Barrier(MPI_COMM_WORLD);
    double t05 = MPI_Wtime();
    distribute_particles(&rank_array, &cfg.length_per_rank, cfg.nprocs);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (cfg.power < 4)
    {
        char filename[128];
        std::sprintf(filename, "particle_file_cpu_n%d_total%lld.par", cfg.nprocs, cfg.total_particles);
        MPI_Allgather(&cfg.length_per_rank, 1, MPI_INT, length_vector, 1, MPI_INT, MPI_COMM_WORLD);
        parallel_write_to_file(rank_array, length_vector, filename);
    }

    if (cfg.rank == 0)
    {
        const char *mode_str = (cfg.exp_type == WEAK_SCALING) ? "weak" : "strong";
        std::string out = std::string("../results_") + mode_str + ".csv";
        log_results(cfg, t05 - t0, t1 - t05, out.c_str());
    }

    std::free(rank_array);
    std::free(length_vector);
    MPI_Type_free(&MPI_particle);
    MPI_Finalize();
    return 0;
}