#include "hashed_octree.hpp"

#include <algorithm>
#include <cstddef>

namespace
{
struct PendingNode
{
    int node_index;
    int begin;
    int end;
};

inline int octant_at_level(const t_particle &particle, int level)
{
    const int shift = 3 * (MAX_DEPTH - level);
    return static_cast<int>((static_cast<unsigned long long>(particle.key) >> shift) & 0x7ull);
}
}

int build_local_hashed_octree(HashedOctree &tree,
                              const t_particle *particles,
                              int count,
                              int owner_rank)
{
    if (count < 0 || (count > 0 && particles == nullptr))
        return 1;

    if (count > 1 && !std::is_sorted(
                         particles,
                         particles + count,
                         [](const t_particle &left, const t_particle &right)
                         {
                             return static_cast<unsigned long long>(left.key) <
                                    static_cast<unsigned long long>(right.key);
                         }))
        return 2;

    tree.clear();
    tree.nodes.reserve(static_cast<std::size_t>(count) * 2 + 1);
    tree.key_to_node.reserve(static_cast<std::size_t>(count) * 2 + 1);

    TreeNode root;
    root.key = 1;
    root.owner = owner_rank;
    root.level = 0;
    root.particle_begin = count > 0 ? 0 : -1;
    root.particle_count = static_cast<unsigned long long>(count);
    root.is_leaf = count <= 1;
    root.children_available = true;

    tree.nodes.push_back(root);
    tree.key_to_node.emplace(root.key, 0);
    tree.root = 0;

    if (count <= 1)
        return 0;

    std::vector<PendingNode> pending;
    pending.push_back({tree.root, 0, count});

    while (!pending.empty())
    {
        const PendingNode current = pending.back();
        pending.pop_back();

        const int child_level = tree.nodes[current.node_index].level + 1;
        if (child_level > MAX_DEPTH)
        {
            tree.nodes[current.node_index].is_leaf = true;
            continue;
        }

        int cursor = current.begin;
        int previous_child = -1;
        std::vector<PendingNode> internal_children;

        for (int octant = 0; octant < 8; ++octant)
        {
            const int child_begin = cursor;
            while (cursor < current.end &&
                   octant_at_level(particles[cursor], child_level) == octant)
                ++cursor;

            if (cursor == child_begin)
                continue;

            TreeNode child;
            child.key = (tree.nodes[current.node_index].key << 3) |
                        static_cast<unsigned long long>(octant);
            child.owner = owner_rank;
            child.level = child_level;
            child.parent = current.node_index;
            child.particle_begin = child_begin;
            child.particle_count = static_cast<unsigned long long>(cursor - child_begin);
            child.is_leaf = child.particle_count == 1 || child_level == MAX_DEPTH;
            child.children_available = true;

            const int child_index = static_cast<int>(tree.nodes.size());
            tree.nodes.push_back(child);
            if (!tree.key_to_node.emplace(child.key, child_index).second)
                return 3;

            TreeNode &parent = tree.nodes[current.node_index];
            parent.child_mask |= static_cast<std::uint8_t>(1u << octant);
            if (parent.first_child < 0)
                parent.first_child = child_index;
            if (previous_child >= 0)
                tree.nodes[previous_child].next_sibling = child_index;
            previous_child = child_index;

            if (!child.is_leaf)
                internal_children.push_back({child_index, child_begin, cursor});
        }

        TreeNode &parent = tree.nodes[current.node_index];
        parent.is_leaf = parent.first_child < 0;
        parent.children_available = true;

        for (auto it = internal_children.rbegin(); it != internal_children.rend(); ++it)
            pending.push_back(*it);
    }

    return 0;
}
