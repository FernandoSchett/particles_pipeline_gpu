#include "hashed_octree.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>

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

template <typename T>
void append_binary(std::vector<unsigned char> &buffer, const T &value)
{
    const std::size_t old_size = buffer.size();
    buffer.resize(old_size + sizeof(T));
    std::memcpy(buffer.data() + old_size, &value, sizeof(T));
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

int write_hashed_octree_file(const HashedOctree &tree,
                             const ExecConfig &cfg,
                             const char *filename,
                             MPI_Comm communicator)
{
    constexpr char magic[8] = {'P', 'S', 'F', 'C', 'T', 'R', 'E', 'E'};
    constexpr std::uint32_t version = 1;
    constexpr MPI_Offset header_size = 24;
    constexpr MPI_Offset section_header_size = 20;
    constexpr MPI_Offset node_record_size = 42;

    if (tree.root != 0 || tree.nodes.empty() || tree.nodes.size() != tree.key_to_node.size())
        return 1;
    if (tree.nodes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return 2;

    const unsigned long long local_node_count = tree.nodes.size();
    std::vector<unsigned long long> node_counts(cfg.nprocs, 0);
    MPI_Allgather(&local_node_count, 1, MPI_UNSIGNED_LONG_LONG,
                  node_counts.data(), 1, MPI_UNSIGNED_LONG_LONG, communicator);

    MPI_Offset rank_offset = header_size;
    for (int rank = 0; rank < cfg.rank; ++rank)
        rank_offset += section_header_size +
                       static_cast<MPI_Offset>(node_counts[rank]) * node_record_size;

    std::vector<unsigned char> section;
    section.reserve(static_cast<std::size_t>(section_header_size +
                                             local_node_count * node_record_size));

    const std::int32_t owner = cfg.rank;
    const std::uint64_t particle_count = static_cast<std::uint64_t>(cfg.length_per_rank);
    const std::uint64_t node_count = static_cast<std::uint64_t>(local_node_count);
    append_binary(section, owner);
    append_binary(section, particle_count);
    append_binary(section, node_count);

    for (const TreeNode &node : tree.nodes)
    {
        const std::uint64_t key = node.key;
        const std::int32_t node_owner = node.owner;
        const std::int32_t level = node.level;
        const std::int32_t parent = node.parent;
        const std::int32_t first_child = node.first_child;
        const std::int32_t next_sibling = node.next_sibling;
        const std::uint8_t child_mask = node.child_mask;
        const std::int32_t particle_begin = node.particle_begin;
        const std::uint64_t node_particle_count = node.particle_count;
        const std::uint8_t flags =
            (node.is_leaf ? 1u : 0u) |
            (node.children_available ? 2u : 0u) |
            (node.is_branch ? 4u : 0u) |
            (node.is_remote ? 8u : 0u);

        append_binary(section, key);
        append_binary(section, node_owner);
        append_binary(section, level);
        append_binary(section, parent);
        append_binary(section, first_child);
        append_binary(section, next_sibling);
        append_binary(section, child_mask);
        append_binary(section, particle_begin);
        append_binary(section, node_particle_count);
        append_binary(section, flags);
    }

    if (section.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return 2;

    MPI_File file;
    int status = MPI_File_open(communicator, filename,
                               MPI_MODE_CREATE | MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &file);
    if (status != MPI_SUCCESS)
        return 3;

    MPI_File_set_size(file, 0);
    MPI_Barrier(communicator);

    if (cfg.rank == 0)
    {
        std::vector<unsigned char> header;
        header.reserve(static_cast<std::size_t>(header_size));
        header.insert(header.end(), magic, magic + sizeof(magic));
        append_binary(header, version);
        append_binary(header, static_cast<std::uint32_t>(cfg.nprocs));
        append_binary(header, static_cast<std::uint64_t>(cfg.total_particles));
        MPI_File_write_at(file, 0, header.data(), static_cast<int>(header.size()),
                          MPI_BYTE, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(communicator);
    status = MPI_File_write_at_all(file, rank_offset, section.data(),
                                   static_cast<int>(section.size()), MPI_BYTE,
                                   MPI_STATUS_IGNORE);
    MPI_File_close(&file);
    return status == MPI_SUCCESS ? 0 : 4;
}
