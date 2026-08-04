#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

namespace
{
int generate_particle_coordinates(const ExecConfig &cfg, t_particle **particles)
{
    switch (cfg.dist_type)
    {
    case DIST_BOX:
        return box_distribution(
            particles, cfg.length_per_rank, cfg.box_length, cfg.seed + cfg.rank);
    case DIST_TORUS:
        return torus_distribution(
            particles, cfg.length_per_rank, cfg.major_r, cfg.minor_r,
            cfg.box_length, cfg.seed + cfg.rank);
    default:
        return box_distribution(
            particles, cfg.length_per_rank, cfg.box_length, cfg.seed + cfg.rank);
    }
}

int distribute_particles_by_morton_key(
    t_particle **particles,
    ExecConfig &cfg,
    std::vector<unsigned long long> &splitters,
    double &splitters_end,
    double &distribution_end)
{
    const bool tree_requested = cfg.alg_type == BUILD_TABLE;
    if (cfg.nprocs > 1 || tree_requested)
        discover_splitters_cpu(
            *particles, cfg.length_per_rank, splitters);

    MPI_Barrier(MPI_COMM_WORLD);
    splitters_end = MPI_Wtime();

    int status = 0;
    if (cfg.nprocs > 1)
        status = redistribute_by_splitters_cpu(
            particles, &cfg.length_per_rank, splitters);

    MPI_Barrier(MPI_COMM_WORLD);
    distribution_end = MPI_Wtime();
    return status;
}

void print_distributed_tree_summary(
    const HashedOctree &tree,
    const ExecConfig &cfg)
{
    std::size_t leaf_count = 0;
    std::size_t fill_count = 0;
    std::size_t local_branch_count = 0;
    std::size_t remote_branch_count = 0;
    for (const TreeNode &node : tree.nodes)
    {
        leaf_count += node.is_leaf ? 1u : 0u;
        fill_count += node.owner == -1 && !node.is_branch ? 1u : 0u;
        local_branch_count += node.is_branch && !node.is_remote ? 1u : 0u;
        remote_branch_count += node.is_branch && node.is_remote ? 1u : 0u;
    }

    std::printf("[DISTRIBUTED TREE] rank=%d particles=%d nodes=%zu leaves=%zu "
                "fill=%zu local_branches=%zu remote_branches=%zu hash_entries=%zu\n",
                cfg.rank, cfg.length_per_rank, tree.nodes.size(), leaf_count,
                fill_count, local_branch_count, remote_branch_count,
                tree.key_to_node.size());
}

int write_small_debug_outputs(
    const ExecConfig &cfg,
    t_particle *particles,
    int *length_vector,
    const HashedOctree &distributed_tree)
{
    write_par_cpu(cfg, particles, length_vector);
    if (cfg.alg_type != BUILD_TABLE)
        return 0;

    char tree_filename[128];
    std::snprintf(tree_filename, sizeof(tree_filename),
                  "tree_file_cpu_n%d_total%lld.tree",
                  cfg.nprocs, cfg.total_particles);
    return write_hashed_octree_file(
        distributed_tree, cfg, tree_filename, MPI_COMM_WORLD);
}

void record_execution_times(
    const ExecConfig &cfg,
    exec_times &times,
    double allocation_start,
    double allocation_end,
    double generation_start,
    double generation_end,
    double splitters_end,
    double distribution_end,
    double tree_start,
    double tree_end)
{
    if (cfg.rank != 0)
        return;

    times.alloc_time = allocation_end - allocation_start;
    times.gen_time = generation_end - generation_start;
    times.splitters_time = splitters_end - generation_end;
    times.dist_time = distribution_end - splitters_end;
    times.tree_time = tree_end - tree_start;
    times.total_time = tree_end - generation_start;

    const char *mode = cfg.exp_type == WEAK_SCALING ? "weak" : "strong";
    const std::string output = std::string("../results_") + mode + ".csv";
    log_results(cfg, times, output.c_str());
}
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ExecConfig cfg{};
    exec_times times = {};

    MPI_Comm_rank(MPI_COMM_WORLD, &cfg.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cfg.nprocs);
    cfg.device = "cpu";
    parse_args(argc, argv, cfg);

    register_MPI_Particle(&MPI_particle);
    setup_particles_box_length(cfg);

    std::vector<int> length_vector(cfg.nprocs);
    std::vector<unsigned long long> splitters;
    HashedOctree distributed_tree;
    t_particle *particles = nullptr;

    MPI_Allgather(&cfg.length_per_rank, 1, MPI_INT,
                  length_vector.data(), 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double allocation_start = MPI_Wtime();
    allocate_particle(&particles, cfg.length_per_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    const double allocation_end = MPI_Wtime();

    generate_particle_coordinates(cfg, &particles);

    MPI_Barrier(MPI_COMM_WORLD);
    const double generation_start = MPI_Wtime();
    generate_particles_keys(particles, cfg.length_per_rank, cfg.box_length);

    MPI_Barrier(MPI_COMM_WORLD);
    const double generation_end = MPI_Wtime();
    double splitters_end = generation_end;
    double distribution_end = generation_end;

    distribute_particles_by_morton_key(
        &particles, cfg, splitters, splitters_end, distribution_end);

    double tree_start = distribution_end;
    double tree_end = distribution_end;
    if (cfg.alg_type == BUILD_TABLE)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        tree_start = MPI_Wtime();
        const DistributedTreeBuildParameters tree_parameters{
            particles,
            cfg.length_per_rank,
            splitters,
            cfg,
            MPI_COMM_WORLD};
        construct_distributed_hashed_octree_cpu(
            distributed_tree, tree_parameters);
        MPI_Barrier(MPI_COMM_WORLD);
        tree_end = MPI_Wtime();
        print_distributed_tree_summary(distributed_tree, cfg);
    }

    write_small_debug_outputs(
        cfg, particles, length_vector.data(), distributed_tree);

    record_execution_times(
        cfg, times, allocation_start, allocation_end,
        generation_start, generation_end, splitters_end, distribution_end,
        tree_start, tree_end);

    std::free(particles);
    MPI_Type_free(&MPI_particle);
    MPI_Finalize();
    return 0;
}
