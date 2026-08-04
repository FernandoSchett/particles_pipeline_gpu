#include "hashed_octree.hpp"
#include "particles_cpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
constexpr std::uint64_t key_space_end = 1ull << (3 * MAX_DEPTH);
constexpr std::uint64_t root_octant_width = 1ull << (3 * (MAX_DEPTH - 1));

int global_status(int local_status, MPI_Comm communicator)
{
    int status = 0;
    MPI_Allreduce(&local_status, &status, 1, MPI_INT, MPI_MAX, communicator);
    return status;
}

int run_tree_case(const char *name,
                  const std::vector<std::uint64_t> &local_keys,
                  const std::vector<unsigned long long> &splitters,
                  bool require_multiple_local_branches,
                  MPI_Comm communicator)
{
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &nprocs);

    std::vector<t_particle> particles(local_keys.size());
    for (std::size_t index = 0; index < local_keys.size(); ++index)
    {
        particles[index].mpi_rank = rank;
        particles[index].key = static_cast<long long>(local_keys[index]);
        particles[index].coord[0] = 0;
        particles[index].coord[1] = 0;
        particles[index].coord[2] = 0;
    }
    sort_particles_by_key_cpu(particles.data(), static_cast<int>(particles.size()));

    int status = validate_mpi_key_boundaries(
        particles.data(), static_cast<int>(particles.size()), communicator);
    const KeyInterval interval = get_rank_key_interval(rank, nprocs, splitters);
    HashedOctree temporary_tree;
    if (status == 0)
        status = build_local_hashed_octree(
            temporary_tree, particles.data(), static_cast<int>(particles.size()),
            rank, &interval);

    std::vector<int> local_branch_indices;
    std::vector<BranchSummary> local_branches;
    if (status == 0)
    {
        local_branch_indices = find_local_branch_nodes(temporary_tree, interval);
        if (!particles.empty() && local_branch_indices.empty())
            status = 20;
        if (require_multiple_local_branches && particles.size() > 1 &&
            local_branch_indices.size() < 2)
            status = 21;
        local_branches = pack_local_branches(temporary_tree, local_branch_indices);
        if (local_branches.size() != local_branch_indices.size())
            status = 22;
    }
    status = global_status(status, communicator);
    if (status != 0)
        return status;

    std::vector<BranchSummary> all_branches;
    status = exchange_branch_summaries(local_branches, all_branches, communicator);
    HashedOctree distributed_tree;
    ExecConfig cfg{};
    cfg.rank = rank;
    cfg.nprocs = nprocs;
    cfg.length_per_rank = static_cast<int>(particles.size());
    long long local_count = static_cast<long long>(particles.size());
    MPI_Allreduce(&local_count, &cfg.total_particles, 1,
                  MPI_LONG_LONG_INT, MPI_SUM, communicator);

    if (status == 0)
        status = build_distributed_hashed_octree(
            distributed_tree, temporary_tree, local_branch_indices, all_branches, cfg);
    if (status == 0)
        status = rebuild_tree_links(distributed_tree);
    if (status == 0)
        status = compute_global_particle_counts(distributed_tree);
    status = global_status(status, communicator);
    if (status == 0)
        status = validate_distributed_tree(
            distributed_tree, cfg, all_branches, communicator);
    if (status != 0)
        return status;

    std::size_t local_branch_count = 0;
    std::size_t remote_branch_count = 0;
    for (const TreeNode &node : distributed_tree.nodes)
    {
        local_branch_count += node.is_branch && !node.is_remote ? 1u : 0u;
        remote_branch_count += node.is_branch && node.is_remote ? 1u : 0u;
    }
    if (local_branch_count != local_branches.size() ||
        remote_branch_count != all_branches.size() - local_branches.size())
        status = 30;

    status = global_status(status, communicator);
    if (rank == 0 && status == 0)
        std::printf("[PASS] %s: particles=%lld branches=%zu\n",
                    name, cfg.total_particles, all_branches.size());
    return status;
}

int test_pure_ranges(int rank)
{
    if (rank != 0)
        return 0;
    const MortonRange root = node_morton_range(1, 0);
    if (root.begin != 0 || root.end != key_space_end)
        return 1;

    const int level = 2;
    const std::uint64_t path = (2ull << 3) | 5ull;
    const std::uint64_t node_key = (1ull << (3 * level)) | path;
    const MortonRange node = node_morton_range(node_key, level);
    const int shift = 3 * (MAX_DEPTH - level);
    if (node.begin != (path << shift) || node.end != ((path + 1) << shift))
        return 2;

    const std::vector<unsigned long long> splitters = {99, 199, 299};
    if (get_rank_key_interval(0, 4, splitters).begin != 0 ||
        get_rank_key_interval(0, 4, splitters).end != 100 ||
        get_rank_key_interval(1, 4, splitters).begin != 100 ||
        get_rank_key_interval(1, 4, splitters).end != 200 ||
        get_rank_key_interval(3, 4, splitters).begin != 300 ||
        get_rank_key_interval(3, 4, splitters).end != key_space_end)
        return 3;
    return 0;
}

int test_repeated_key_redistribution(int rank, MPI_Comm communicator)
{
    std::vector<t_particle> initial(2);
    const std::uint64_t keys[2][2] = {{700, 777}, {777, 900}};
    for (int index = 0; index < 2; ++index)
    {
        initial[index].mpi_rank = rank;
        initial[index].key = static_cast<long long>(keys[rank][index]);
        initial[index].coord[0] = 0;
        initial[index].coord[1] = 0;
        initial[index].coord[2] = 0;
    }
    sort_particles_by_key_cpu(initial.data(), static_cast<int>(initial.size()));

    t_particle *particles = static_cast<t_particle *>(
        std::malloc(initial.size() * sizeof(t_particle)));
    const int allocation_status = global_status(particles ? 0 : 1, communicator);
    if (allocation_status != 0)
    {
        std::free(particles);
        return allocation_status;
    }
    std::copy(initial.begin(), initial.end(), particles);
    int particle_count = static_cast<int>(initial.size());
    int status = redistribute_by_splitters_cpu(
        &particles, &particle_count, std::vector<unsigned long long>{777});
    status = global_status(status, communicator);
    sort_particles_by_key_cpu(particles, particle_count);
    if (status == 0)
        status = validate_mpi_key_boundaries(particles, particle_count, communicator);

    int repeated_count = 0;
    for (int index = 0; index < particle_count; ++index)
        repeated_count += static_cast<std::uint64_t>(particles[index].key) == 777 ? 1 : 0;
    int all_repeated[2] = {0, 0};
    MPI_Allgather(&repeated_count, 1, MPI_INT,
                  all_repeated, 1, MPI_INT, communicator);
    if (all_repeated[0] != 2 || all_repeated[1] != 0)
        status = 2;
    std::free(particles);
    return global_status(status, communicator);
}
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    register_MPI_Particle(&MPI_particle);

    int status = global_status(test_pure_ranges(rank), MPI_COMM_WORLD);
    if (status == 0 && nprocs == 1)
    {
        status = run_tree_case("one-rank", {1, root_octant_width + 7}, {}, true,
                               MPI_COMM_WORLD);
        if (status == 0)
            status = run_tree_case("one-particle", {12345}, {}, false,
                                   MPI_COMM_WORLD);
    }
    else if (status == 0 && nprocs == 2)
    {
        const std::vector<unsigned long long> splitters = {
            4 * root_octant_width - 1};
        const std::vector<std::uint64_t> keys = rank == 0
            ? std::vector<std::uint64_t>{1, root_octant_width + 9}
            : std::vector<std::uint64_t>{4 * root_octant_width + 3,
                                         7 * root_octant_width + 11};
        status = run_tree_case("two-ranks-separated-octants", keys, splitters, true,
                               MPI_COMM_WORLD);

        const std::vector<std::uint64_t> singleton = rank == 0
            ? std::vector<std::uint64_t>{11}
            : std::vector<std::uint64_t>{6 * root_octant_width + 11};
        if (status == 0)
            status = run_tree_case("one-particle-per-rank", singleton, splitters, false,
                                   MPI_COMM_WORLD);

        const std::vector<unsigned long long> empty_splitter = {root_octant_width - 1};
        const std::vector<std::uint64_t> empty_rank = rank == 0
            ? std::vector<std::uint64_t>{}
            : std::vector<std::uint64_t>{root_octant_width + 17,
                                         root_octant_width + 17};
        if (status == 0)
            status = run_tree_case("empty-rank-and-repeated-key", empty_rank,
                                   empty_splitter, false, MPI_COMM_WORLD);

        if (status == 0)
            status = test_repeated_key_redistribution(rank, MPI_COMM_WORLD);

        std::vector<t_particle> split_key(1);
        split_key[0].mpi_rank = rank;
        split_key[0].key = 777;
        const int boundary_status = validate_mpi_key_boundaries(
            split_key.data(), 1, MPI_COMM_WORLD);
        if (status == 0 && boundary_status != 3)
            status = 40;
    }
    else if (status == 0 && nprocs == 4)
    {
        const std::vector<unsigned long long> splitters = {
            root_octant_width / 2 - 1,
            root_octant_width + root_octant_width / 2 - 1,
            5 * root_octant_width + root_octant_width / 3 - 1};
        const std::uint64_t begin = rank == 0 ? 0 : splitters[rank - 1] + 1;
        const std::uint64_t end = rank == 3 ? key_space_end : splitters[rank] + 1;
        std::vector<std::uint64_t> keys;
        if (rank != 2)
            keys = {begin + (end - begin) / 4, begin + 3 * (end - begin) / 4};
        status = run_tree_case("splitters-cut-upper-octants-with-empty-rank",
                               keys, splitters, rank != 2, MPI_COMM_WORLD);
    }
    else if (status == 0)
    {
        status = 99;
    }

    status = global_status(status, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("distributed_hashed_octree_tests np=%d: %s (%d)\n",
                    nprocs, status == 0 ? "OK" : "FAIL", status);
    MPI_Type_free(&MPI_particle);
    MPI_Finalize();
    return status == 0 ? 0 : 1;
}
