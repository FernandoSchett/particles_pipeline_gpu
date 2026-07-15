#ifndef P_SFC_HASHED_OCTREE_HPP
#define P_SFC_HASHED_OCTREE_HPP

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "particle_types.hpp"

struct TreeNode
{
    unsigned long long key = 1;
    int owner = -1;
    int level = 0;

    int parent = -1;
    int first_child = -1;
    int next_sibling = -1;

    std::uint8_t child_mask = 0;
    int particle_begin = -1;
    unsigned long long particle_count = 0;

    bool is_leaf = false;
    bool children_available = false;
    bool is_branch = false;
    bool is_remote = false;
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

int build_local_hashed_octree(HashedOctree &tree,
                              const t_particle *particles,
                              int count,
                              int owner_rank);

#endif
