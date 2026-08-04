#ifndef P_SFC_HASHED_OCTREE_HPP
#define P_SFC_HASHED_OCTREE_HPP

#include <cstdint>
#include <mpi.h>
#include <unordered_map>
#include <vector>

#include "particle_types.hpp"

struct TreeNode
{
    std::uint64_t key = 0;
    int owner = -1;
    int level = 0;

    int parent = -1;
    int first_child = -1;
    int next_sibling = -1;

    std::uint8_t child_mask = 0;
    int particle_begin = -1;
    std::uint64_t particle_count = 0;

    bool is_leaf = false;
    bool children_available = false;
    bool is_branch = false;
    bool is_remote = false;
};

struct KeyInterval
{
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

struct MortonRange
{
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

struct BranchSummary
{
    std::uint64_t key = 0;
    std::uint64_t particle_count = 0;
    std::int32_t owner = -1;
    std::int32_t level = 0;
    std::uint8_t child_mask = 0;
    std::uint8_t is_leaf = 0;
};

struct HashedOctree
{
    std::vector<TreeNode> nodes;
    std::unordered_map<unsigned long long, int> key_to_node;
    int root = -1;

    void clear()
    {
        nodes.clear();
        key_to_node.clear();
        root = -1;
    }
};

struct DistributedTreeBuildParameters
{
    t_particle *particles;
    int particle_count;
    const std::vector<unsigned long long> &splitters;
    const ExecConfig &execution;
    MPI_Comm communicator;
    HashedOctree *temporary_local_tree = nullptr;
};

int build_local_hashed_octree(HashedOctree &tree,
                              const t_particle *particles,
                              int count,
                              int owner_rank,
                              const KeyInterval *rank_interval = nullptr);

KeyInterval get_rank_key_interval(
    int rank,
    int nprocs,
    const std::vector<unsigned long long> &splitters);

MortonRange node_morton_range(std::uint64_t node_key, int node_level);

std::vector<int> find_local_branch_nodes(
    HashedOctree &tree,
    const KeyInterval &rank_interval);

std::vector<BranchSummary> pack_local_branches(
    const HashedOctree &tree,
    const std::vector<int> &branch_indices);

int exchange_branch_summaries(
    const std::vector<BranchSummary> &local_branches,
    std::vector<BranchSummary> &all_branches,
    MPI_Comm communicator);

int copy_local_branch_subtree(
    const HashedOctree &source,
    int source_branch_index,
    HashedOctree &destination);

int build_distributed_hashed_octree(
    HashedOctree &distributed_tree,
    const HashedOctree &temporary_local_tree,
    const std::vector<int> &local_branch_indices,
    const std::vector<BranchSummary> &all_branches,
    const ExecConfig &cfg);

int rebuild_tree_links(HashedOctree &tree);

int compute_global_particle_counts(HashedOctree &tree);

int validate_distributed_tree(
    const HashedOctree &tree,
    const ExecConfig &cfg,
    const std::vector<BranchSummary> &all_branches,
    MPI_Comm communicator);

int construct_distributed_hashed_octree_cpu(
    HashedOctree &distributed_tree,
    const DistributedTreeBuildParameters &parameters);

int write_hashed_octree_file(const HashedOctree &tree,
                             const ExecConfig &cfg,
                             const char *filename,
                             MPI_Comm communicator);

#endif
