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
#include <cstring>

#include "particle_types.hpp"
#include "particles_cpu.hpp"
#include "hashed_octree.hpp"
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
    cfg.alg_type = GLOBAL_SORTING;

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

    if (argc > 5)
    {
        if (strcmp(argv[5], "table") == 0)
            cfg.alg_type = BUILD_TABLE;
        else if (strcmp(argv[5], "total") == 0)
            cfg.alg_type = GLOBAL_SORTING;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ExecConfig cfg;
    exec_times times = {};

    MPI_Comm_rank(MPI_COMM_WORLD, &cfg.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cfg.nprocs);
    cfg.device = "cpu";
    parse_args(argc, argv, cfg);

    std::vector<unsigned long long> splitters;
    HashedOctree local_tree;
    register_MPI_Particle(&MPI_particle);
    int *length_vector = (int *)std::malloc(cfg.nprocs * sizeof(int));
    t_particle *rank_array = nullptr;

    setup_particles_box_length(cfg);
    MPI_Allgather(&cfg.length_per_rank, 1, MPI_INT, length_vector, 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double alloc_t0 = MPI_Wtime();
    const int alloc_status = allocate_particle(&rank_array, cfg.length_per_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    const double alloc_t1 = MPI_Wtime();
    if (alloc_status != 0)
    {
        std::fprintf(stderr, "Rank %d failed to allocate particle array\n", cfg.rank);
        MPI_Abort(MPI_COMM_WORLD, alloc_status);
    }

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
    double t1 = MPI_Wtime();
    double t2 = t1, t3 = t1;

    switch (cfg.alg_type)
    {
    case GLOBAL_SORTING:
    case BUILD_TABLE:
    {
        const bool build_table = cfg.alg_type == BUILD_TABLE;
        if (cfg.nprocs > 1 || build_table)
            discover_splitters_cpu(rank_array, cfg.length_per_rank, splitters);

        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();

        if (cfg.nprocs > 1)
            redistribute_by_splitters_cpu(&rank_array, &cfg.length_per_rank, splitters);

        MPI_Barrier(MPI_COMM_WORLD);
        t3 = MPI_Wtime();

        if (build_table)
        {
            sort_particles_by_key_cpu(rank_array, cfg.length_per_rank);
            const int tree_status = build_local_hashed_octree(
                local_tree, rank_array, cfg.length_per_rank, cfg.rank);
            if (tree_status != 0)
            {
                std::fprintf(stderr, "Rank %d failed to build local hashed octree: %d\n",
                             cfg.rank, tree_status);
                MPI_Abort(MPI_COMM_WORLD, tree_status);
            }

            std::size_t leaf_count = 0;
            for (const TreeNode &node : local_tree.nodes)
                leaf_count += node.is_leaf ? 1u : 0u;

            std::printf("[TREE] rank=%d particles=%d nodes=%zu leaves=%zu hash_entries=%zu\n",
                        cfg.rank, cfg.length_per_rank, local_tree.nodes.size(), leaf_count,
                        local_tree.key_to_node.size());
        }
        break;
    }
    }

    if (cfg.power < 4)
    {
        write_par_cpu(cfg, rank_array, length_vector);
        if (cfg.alg_type == BUILD_TABLE)
        {
            char tree_filename[128];
            std::snprintf(tree_filename, sizeof(tree_filename),
                          "tree_file_cpu_n%d_total%lld.tree",
                          cfg.nprocs, cfg.total_particles);
            const int tree_write_status = write_hashed_octree_file(
                local_tree, cfg, tree_filename, MPI_COMM_WORLD);
            if (tree_write_status != 0)
            {
                std::fprintf(stderr, "Rank %d failed to write hashed octree: %d\n",
                             cfg.rank, tree_write_status);
                MPI_Abort(MPI_COMM_WORLD, tree_write_status);
            }
        }
    }

    if (cfg.rank == 0)
    {
        times.alloc_time = alloc_t1 - alloc_t0;
        times.gen_time = t1 - t0;
        times.splitters_time = t2 - t1;
        times.dist_time = t3 - t2;
        times.total_time = t3 - t0;

        const char *mode_str = (cfg.exp_type == WEAK_SCALING) ? "weak" : "strong";
        std::string out = std::string("../results_") + mode_str + ".csv";
        log_results(cfg, times, out.c_str());
    }

    std::free(rank_array);
    std::free(length_vector);
    MPI_Type_free(&MPI_particle);
    MPI_Finalize();
    return 0;
}
