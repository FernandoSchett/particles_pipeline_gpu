#include "hashed_octree.hpp"
#include "particles_cpu.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace
{
static_assert(3 * MAX_DEPTH < 63, "MAX_DEPTH does not fit in a 64-bit Morton key");

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

bool range_inside(const MortonRange &range, const KeyInterval &interval)
{
    return range.begin < range.end &&
           interval.begin <= range.begin && range.end <= interval.end;
}

int node_level_from_key(std::uint64_t key)
{
    if (key == 0)
        return -1;
    int highest_bit = 0;
    for (std::uint64_t value = key; value >>= 1;)
        ++highest_bit;
    return highest_bit % 3 == 0 ? highest_bit / 3 : -1;
}

TreeNode make_fill_node(std::uint64_t key, int level)
{
    TreeNode node;
    node.key = key;
    node.level = level;
    node.owner = -1;
    node.children_available = true;
    return node;
}

int insert_node(HashedOctree &tree, const TreeNode &node)
{
    if (tree.nodes.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return -1;
    const int index = static_cast<int>(tree.nodes.size());
    if (!tree.key_to_node.emplace(node.key, index).second)
        return -1;
    tree.nodes.push_back(node);
    return index;
}

std::uint64_t checksum_mix(std::uint64_t hash, std::uint64_t value)
{
    constexpr std::uint64_t fnv_prime = 1099511628211ull;
    for (int byte = 0; byte < 8; ++byte)
    {
        hash ^= (value >> (8 * byte)) & 0xffull;
        hash *= fnv_prime;
    }
    return hash;
}
}

int build_local_hashed_octree(HashedOctree &tree,
                              const t_particle *particles,
                              int count,
                              int owner_rank,
                              const KeyInterval *rank_interval)
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
    root.is_leaf = count == 0;
    root.children_available = true;

    tree.nodes.push_back(root);
    tree.key_to_node.emplace(root.key, 0);
    tree.root = 0;

    if (count == 0)
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
            const bool particle_is_alone = child.particle_count == 1;
            const bool cell_belongs_to_rank = !rank_interval || range_inside(
                node_morton_range(child.key, child.level), *rank_interval);
            child.is_leaf = (particle_is_alone && cell_belongs_to_rank) ||
                            child_level == MAX_DEPTH;
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

KeyInterval get_rank_key_interval(
    int rank,
    int nprocs,
    const std::vector<unsigned long long> &splitters)
{
    const std::uint64_t key_space_end = 1ull << (3 * MAX_DEPTH);
    if (rank < 0 || rank >= nprocs || nprocs < 1 ||
        splitters.size() != static_cast<std::size_t>(nprocs - 1) ||
        !std::is_sorted(splitters.begin(), splitters.end()) ||
        (!splitters.empty() && splitters.back() >= key_space_end))
        return {};

    KeyInterval interval;
    // Redistribution uses upper_bound: splitter keys stay in the rank on the
    // left. Adding one converts (..., splitter] to a semi-open interval.
    interval.begin = rank == 0 ? 0 : splitters[rank - 1] + 1;
    interval.end = rank == nprocs - 1 ? key_space_end : splitters[rank] + 1;
    return interval;
}

MortonRange node_morton_range(std::uint64_t node_key, int node_level)
{
    if (node_level < 0 || node_level > MAX_DEPTH)
        return {};

    const int marker_shift = 3 * node_level;
    const std::uint64_t marker = 1ull << marker_shift;
    if (node_key < marker || node_key >= 2 * marker)
        return {};

    const int suffix_shift = 3 * (MAX_DEPTH - node_level);
    const std::uint64_t prefix = node_key - marker;
    return {prefix << suffix_shift, (prefix + 1) << suffix_shift};
}

std::vector<int> find_local_branch_nodes(
    HashedOctree &tree,
    const KeyInterval &rank_interval)
{
    std::vector<int> branches;
    if (tree.root < 0 || tree.root >= static_cast<int>(tree.nodes.size()) ||
        rank_interval.begin > rank_interval.end)
        return branches;

    for (TreeNode &node : tree.nodes)
        node.is_branch = false;

    const std::uint64_t key_space_end = 1ull << (3 * MAX_DEPTH);
    const bool owns_whole_domain =
        rank_interval.begin == 0 && rank_interval.end == key_space_end;

    for (int index = 0; index < static_cast<int>(tree.nodes.size()); ++index)
    {
        TreeNode &node = tree.nodes[index];
        if (index == tree.root || node.particle_count == 0)
            continue;

        const MortonRange node_range = node_morton_range(node.key, node.level);
        if (!range_inside(node_range, rank_interval))
            continue;

        bool parent_inside = false;
        if (node.parent >= 0)
        {
            const TreeNode &parent = tree.nodes[node.parent];
            parent_inside = range_inside(
                node_morton_range(parent.key, parent.level), rank_interval);
        }

        // With one rank the global root stays a fill node; occupied root children
        // delimit the local subtrees attached below it.
        if ((owns_whole_domain && node.parent == tree.root) ||
            (!owns_whole_domain && !parent_inside))
        {
            node.is_branch = true;
            node.is_remote = false;
            node.children_available = true;
            branches.push_back(index);
        }
    }

    std::sort(branches.begin(), branches.end(),
              [&tree](int left, int right)
              {
                  return tree.nodes[left].particle_begin < tree.nodes[right].particle_begin;
              });
    std::uint64_t covered = 0;
    int expected_begin = 0;
    for (const int index : branches)
    {
        const TreeNode &node = tree.nodes[index];
        if (node.particle_begin != expected_begin)
            return {};
        expected_begin += static_cast<int>(node.particle_count);
        covered += node.particle_count;
    }
    if (covered != tree.nodes[tree.root].particle_count)
        return {};
    return branches;
}

std::vector<BranchSummary> pack_local_branches(
    const HashedOctree &tree,
    const std::vector<int> &branch_indices)
{
    std::vector<BranchSummary> summaries;
    summaries.reserve(branch_indices.size());
    for (const int index : branch_indices)
    {
        if (index < 0 || index >= static_cast<int>(tree.nodes.size()))
            return {};
        const TreeNode &node = tree.nodes[index];
        if (!node.is_branch || node.owner < 0)
            return {};
        summaries.push_back({node.key, node.particle_count,
                             static_cast<std::int32_t>(node.owner),
                             static_cast<std::int32_t>(node.level),
                             node.child_mask,
                             static_cast<std::uint8_t>(node.is_leaf ? 1 : 0)});
    }
    return summaries;
}

int exchange_branch_summaries(
    const std::vector<BranchSummary> &local_branches,
    std::vector<BranchSummary> &all_branches,
    MPI_Comm communicator)
{
    if (local_branches.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return 1;

    int nprocs = 1;
    MPI_Comm_size(communicator, &nprocs);
    const int local_count = static_cast<int>(local_branches.size());
    std::vector<int> counts(nprocs), displacements(nprocs, 0);
    if (MPI_Allgather(&local_count, 1, MPI_INT,
                      counts.data(), 1, MPI_INT, communicator) != MPI_SUCCESS)
        return 2;

    long long total_count = 0;
    for (int rank = 0; rank < nprocs; ++rank)
    {
        if (counts[rank] < 0 || total_count > std::numeric_limits<int>::max() - counts[rank])
            return 1;
        displacements[rank] = static_cast<int>(total_count);
        total_count += counts[rank];
    }

    BranchSummary sample;
    MPI_Aint base = 0;
    MPI_Aint offsets[6];
    MPI_Get_address(&sample, &base);
    MPI_Get_address(&sample.key, &offsets[0]);
    MPI_Get_address(&sample.particle_count, &offsets[1]);
    MPI_Get_address(&sample.owner, &offsets[2]);
    MPI_Get_address(&sample.level, &offsets[3]);
    MPI_Get_address(&sample.child_mask, &offsets[4]);
    MPI_Get_address(&sample.is_leaf, &offsets[5]);
    for (MPI_Aint &offset : offsets)
        offset -= base;
    const int block_lengths[6] = {1, 1, 1, 1, 1, 1};
    MPI_Datatype types[6] = {
        MPI_UINT64_T, MPI_UINT64_T, MPI_INT32_T,
        MPI_INT32_T, MPI_UINT8_T, MPI_UINT8_T};
    MPI_Datatype fields_type;
    MPI_Datatype summary_type;
    MPI_Type_create_struct(6, block_lengths, offsets, types, &fields_type);
    MPI_Type_create_resized(fields_type, 0, sizeof(BranchSummary), &summary_type);
    MPI_Type_free(&fields_type);
    MPI_Type_commit(&summary_type);

    all_branches.resize(static_cast<std::size_t>(total_count));
    const int gather_status = MPI_Allgatherv(
        local_branches.data(), local_count, summary_type,
        all_branches.data(), counts.data(), displacements.data(),
        summary_type, communicator);
    MPI_Type_free(&summary_type);
    if (gather_status != MPI_SUCCESS)
        return 2;

    std::sort(all_branches.begin(), all_branches.end(),
              [](const BranchSummary &left, const BranchSummary &right)
              {
                  if (left.key != right.key)
                      return left.key < right.key;
                  return left.owner < right.owner;
              });

    std::vector<std::pair<MortonRange, std::uint64_t>> ranges;
    ranges.reserve(all_branches.size());
    std::uint64_t summary_particle_count = 0;
    for (std::size_t index = 0; index < all_branches.size(); ++index)
    {
        const BranchSummary &summary = all_branches[index];
        if (summary.owner < 0 || summary.owner >= nprocs ||
            summary.level < 1 || summary.level > MAX_DEPTH ||
            node_level_from_key(summary.key) != summary.level ||
            (index > 0 && all_branches[index - 1].key == summary.key))
            return 3;
        const MortonRange range = node_morton_range(summary.key, summary.level);
        if (range.begin >= range.end)
            return 3;
        ranges.push_back({range, summary.key});
        summary_particle_count += summary.particle_count;
    }
    std::sort(ranges.begin(), ranges.end(),
              [](const auto &left, const auto &right)
              {
                  if (left.first.begin != right.first.begin)
                      return left.first.begin < right.first.begin;
                  return left.first.end < right.first.end;
              });
    for (std::size_t index = 1; index < ranges.size(); ++index)
        if (ranges[index - 1].first.end > ranges[index].first.begin)
            return 4;

    std::uint64_t local_particles = 0;
    for (const BranchSummary &summary : local_branches)
        local_particles += summary.particle_count;
    std::uint64_t global_particles = 0;
    MPI_Allreduce(&local_particles, &global_particles, 1,
                  MPI_UINT64_T, MPI_SUM, communicator);
    return summary_particle_count == global_particles ? 0 : 5;
}

int copy_local_branch_subtree(
    const HashedOctree &source,
    int source_branch_index,
    HashedOctree &destination)
{
    if (source_branch_index < 0 ||
        source_branch_index >= static_cast<int>(source.nodes.size()))
        return 1;

    std::vector<int> pending{source_branch_index};
    while (!pending.empty())
    {
        const int source_index = pending.back();
        pending.pop_back();
        const TreeNode &source_node = source.nodes[source_index];
        TreeNode node = source_node;
        node.parent = -1;
        node.first_child = -1;
        node.next_sibling = -1;
        node.is_branch = source_index == source_branch_index;
        node.is_remote = false;
        node.children_available = true;

        const auto existing = destination.key_to_node.find(node.key);
        if (existing != destination.key_to_node.end())
        {
            if (source_index != source_branch_index)
                return 2;
            destination.nodes[existing->second] = node;
        }
        else if (insert_node(destination, node) < 0)
        {
            return 2;
        }

        std::vector<int> children;
        for (int child = source_node.first_child; child >= 0;
             child = source.nodes[child].next_sibling)
            children.push_back(child);
        for (auto child = children.rbegin(); child != children.rend(); ++child)
            pending.push_back(*child);
    }
    return 0;
}

int build_distributed_hashed_octree(
    HashedOctree &distributed_tree,
    const HashedOctree &temporary_local_tree,
    const std::vector<int> &local_branch_indices,
    const std::vector<BranchSummary> &all_branches,
    const ExecConfig &cfg)
{
    distributed_tree.clear();
    TreeNode root = make_fill_node(1, 0);
    if (insert_node(distributed_tree, root) != 0)
        return 1;
    distributed_tree.root = 0;

    for (const BranchSummary &summary : all_branches)
    {
        std::uint64_t parent_key = summary.key >> 3;
        int parent_level = summary.level - 1;
        while (parent_key >= 1)
        {
            if (distributed_tree.key_to_node.find(parent_key) == distributed_tree.key_to_node.end() &&
                insert_node(distributed_tree, make_fill_node(parent_key, parent_level)) < 0)
                return 2;
            if (parent_key == 1)
                break;
            parent_key >>= 3;
            --parent_level;
        }
    }

    for (const BranchSummary &summary : all_branches)
    {
        if (distributed_tree.key_to_node.find(summary.key) != distributed_tree.key_to_node.end())
            return 3;
        TreeNode branch;
        branch.key = summary.key;
        branch.owner = summary.owner;
        branch.level = summary.level;
        branch.child_mask = summary.child_mask;
        branch.particle_count = summary.particle_count;
        branch.is_leaf = summary.is_leaf != 0;
        branch.is_branch = true;
        branch.is_remote = summary.owner != cfg.rank;
        branch.children_available = !branch.is_remote;
        if (insert_node(distributed_tree, branch) < 0)
            return 3;
    }

    std::unordered_map<std::uint64_t, int> local_sources;
    for (const int index : local_branch_indices)
    {
        if (index < 0 || index >= static_cast<int>(temporary_local_tree.nodes.size()))
            return 4;
        local_sources.emplace(temporary_local_tree.nodes[index].key, index);
    }
    for (const BranchSummary &summary : all_branches)
    {
        if (summary.owner != cfg.rank)
            continue;
        const auto source = local_sources.find(summary.key);
        if (source == local_sources.end() ||
            copy_local_branch_subtree(temporary_local_tree, source->second,
                                      distributed_tree) != 0)
            return 4;
        local_sources.erase(source);
    }
    return local_sources.empty() ? 0 : 4;
}

int rebuild_tree_links(HashedOctree &tree)
{
    const auto root_it = tree.key_to_node.find(1);
    if (root_it == tree.key_to_node.end())
        return 1;
    tree.root = root_it->second;

    std::vector<std::vector<int>> children(tree.nodes.size());
    for (TreeNode &node : tree.nodes)
    {
        node.parent = -1;
        node.first_child = -1;
        node.next_sibling = -1;
        if (!(node.is_branch && node.is_remote))
            node.child_mask = 0;
    }

    for (int index = 0; index < static_cast<int>(tree.nodes.size()); ++index)
    {
        TreeNode &node = tree.nodes[index];
        if (index == tree.root)
            continue;
        const auto parent = tree.key_to_node.find(node.key >> 3);
        if (parent == tree.key_to_node.end())
            return 2;
        node.parent = parent->second;
        children[parent->second].push_back(index);
    }

    for (int parent = 0; parent < static_cast<int>(tree.nodes.size()); ++parent)
    {
        auto &siblings = children[parent];
        std::sort(siblings.begin(), siblings.end(),
                  [&tree](int left, int right)
                  {
                      return (tree.nodes[left].key & 7ull) <
                             (tree.nodes[right].key & 7ull);
                  });
        TreeNode &parent_node = tree.nodes[parent];
        if (parent_node.is_branch && parent_node.is_remote && !siblings.empty())
            return 3;
        if (!siblings.empty())
            parent_node.first_child = siblings.front();
        for (std::size_t child = 0; child < siblings.size(); ++child)
        {
            const int child_index = siblings[child];
            parent_node.child_mask |= static_cast<std::uint8_t>(
                1u << (tree.nodes[child_index].key & 7ull));
            tree.nodes[child_index].next_sibling =
                child + 1 < siblings.size() ? siblings[child + 1] : -1;
        }
    }
    return 0;
}

int compute_global_particle_counts(HashedOctree &tree)
{
    std::vector<int> fill_nodes;
    for (int index = 0; index < static_cast<int>(tree.nodes.size()); ++index)
    {
        TreeNode &node = tree.nodes[index];
        if (node.owner == -1 && !node.is_branch)
        {
            node.particle_count = 0;
            fill_nodes.push_back(index);
        }
    }
    std::sort(fill_nodes.begin(), fill_nodes.end(),
              [&tree](int left, int right)
              {
                  return tree.nodes[left].level > tree.nodes[right].level;
              });
    for (const int index : fill_nodes)
    {
        TreeNode &node = tree.nodes[index];
        for (int child = node.first_child; child >= 0;
             child = tree.nodes[child].next_sibling)
            node.particle_count += tree.nodes[child].particle_count;
    }
    return 0;
}

int validate_distributed_tree_local(
    const HashedOctree &tree,
    const ExecConfig &cfg,
    const std::vector<BranchSummary> &all_branches,
    std::uint64_t &checksum)
{
    if (tree.root < 0 || tree.root >= static_cast<int>(tree.nodes.size()))
        return 1;
    const TreeNode &root = tree.nodes[tree.root];
    if (root.key != 1 || root.level != 0 || root.owner != -1 || root.parent != -1 ||
        root.particle_count != static_cast<std::uint64_t>(cfg.total_particles))
        return 1;
    if (tree.nodes.size() != tree.key_to_node.size())
        return 2;

    std::unordered_map<std::uint64_t, const BranchSummary *> summaries;
    for (const BranchSummary &summary : all_branches)
        if (!summaries.emplace(summary.key, &summary).second)
            return 3;

    std::uint64_t local_branch_particles = 0;
    std::vector<std::pair<int, std::uint64_t>> local_ranges;
    std::vector<int> referenced(tree.nodes.size(), 0);
    for (int index = 0; index < static_cast<int>(tree.nodes.size()); ++index)
    {
        const TreeNode &node = tree.nodes[index];
        const auto mapping = tree.key_to_node.find(node.key);
        if (mapping == tree.key_to_node.end() || mapping->second != index ||
            node_level_from_key(node.key) != node.level)
            return 2;
        if (index != tree.root)
        {
            if (node.parent < 0 || node.parent >= static_cast<int>(tree.nodes.size()) ||
                tree.nodes[node.parent].key != (node.key >> 3))
                return 4;
        }

        const auto summary = summaries.find(node.key);
        if (summary != summaries.end())
        {
            const BranchSummary &expected = *summary->second;
            if (!node.is_branch || node.owner != expected.owner ||
                node.level != expected.level ||
                node.particle_count != expected.particle_count ||
                node.child_mask != expected.child_mask ||
                node.is_leaf != (expected.is_leaf != 0))
                return 5;
            if (node.owner == cfg.rank)
            {
                if (node.is_remote || !node.children_available || node.particle_begin < 0)
                    return 5;
                local_branch_particles += node.particle_count;
                local_ranges.push_back({node.particle_begin, node.particle_count});
            }
            else if (!node.is_remote || node.children_available ||
                     node.particle_begin != -1 || node.first_child != -1)
            {
                return 5;
            }
        }
        else if (node.owner == -1)
        {
            if (node.is_branch || node.is_remote || node.particle_begin != -1)
                return 6;
        }
        else if (node.owner != cfg.rank || node.is_branch || node.is_remote ||
                 !node.children_available)
        {
            return 6;
        }

        std::uint8_t materialized_mask = 0;
        std::uint64_t materialized_particles = 0;
        int expected_particle_begin = node.particle_begin;
        int previous_octant = -1;
        std::unordered_set<int> seen;
        for (int child = node.first_child; child >= 0;
             child = tree.nodes[child].next_sibling)
        {
            if (child >= static_cast<int>(tree.nodes.size()) || !seen.insert(child).second ||
                tree.nodes[child].parent != index)
                return 7;
            const int octant = static_cast<int>(tree.nodes[child].key & 7ull);
            if (octant <= previous_octant)
                return 7;
            previous_octant = octant;
            materialized_mask |= static_cast<std::uint8_t>(1u << octant);
            if (node.owner == cfg.rank)
            {
                if (tree.nodes[child].particle_begin != expected_particle_begin)
                    return 7;
                expected_particle_begin += static_cast<int>(tree.nodes[child].particle_count);
            }
            materialized_particles += tree.nodes[child].particle_count;
            referenced[child] += 1;
        }
        if (!(node.is_branch && node.is_remote) && materialized_mask != node.child_mask)
            return 7;
        if (node.owner == -1 && !node.is_branch)
        {
            std::uint64_t count = 0;
            for (int child = node.first_child; child >= 0;
                 child = tree.nodes[child].next_sibling)
                count += tree.nodes[child].particle_count;
            if (count != node.particle_count)
                return 8;
        }
        if (node.owner == cfg.rank)
        {
            if ((node.first_child < 0) != node.is_leaf)
                return 8;
            if (node.is_leaf && node.particle_count > 1 && node.level < MAX_DEPTH)
                return 8;
            if (node.first_child >= 0 && materialized_particles != node.particle_count)
                return 8;
        }
    }
    for (int index = 0; index < static_cast<int>(tree.nodes.size()); ++index)
        if (index != tree.root && referenced[index] != 1)
            return 7;

    std::sort(local_ranges.begin(), local_ranges.end());
    int expected_begin = 0;
    for (const auto &[begin, count] : local_ranges)
    {
        if (begin != expected_begin)
            return 9;
        expected_begin += static_cast<int>(count);
    }
    if (local_branch_particles != static_cast<std::uint64_t>(cfg.length_per_rank) ||
        expected_begin != cfg.length_per_rank)
        return 9;

    std::vector<const TreeNode *> global_nodes;
    for (const TreeNode &node : tree.nodes)
        if (node.owner == -1 || node.is_branch)
            global_nodes.push_back(&node);
    std::sort(global_nodes.begin(), global_nodes.end(),
              [](const TreeNode *left, const TreeNode *right)
              {
                  return left->key < right->key;
              });
    checksum = 1469598103934665603ull;
    for (const TreeNode *node : global_nodes)
    {
        checksum = checksum_mix(checksum, node->key);
        checksum = checksum_mix(checksum, node->particle_count);
        checksum = checksum_mix(checksum, static_cast<std::uint32_t>(node->owner));
        checksum = checksum_mix(checksum, static_cast<std::uint32_t>(node->level));
        checksum = checksum_mix(checksum, node->child_mask);
        checksum = checksum_mix(checksum, node->is_branch ? 1 : 0);
        checksum = checksum_mix(checksum, node->is_leaf ? 1 : 0);
    }
    return 0;
}

int validate_distributed_tree(
    const HashedOctree &tree,
    const ExecConfig &cfg,
    const std::vector<BranchSummary> &all_branches,
    MPI_Comm communicator)
{
    std::uint64_t checksum = 0;
    const int local_status = validate_distributed_tree_local(
        tree, cfg, all_branches, checksum);
    int global_status = 0;
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, communicator);
    if (global_status != 0)
        return global_status;

    std::uint64_t minimum = 0, maximum = 0;
    MPI_Allreduce(&checksum, &minimum, 1, MPI_UINT64_T, MPI_MIN, communicator);
    MPI_Allreduce(&checksum, &maximum, 1, MPI_UINT64_T, MPI_MAX, communicator);
    return minimum == maximum ? 0 : 10;
}

int construct_distributed_hashed_octree_cpu(
    HashedOctree &distributed_tree,
    const DistributedTreeBuildParameters &parameters)
{
    HashedOctree internal_local_tree;
    HashedOctree &temporary_local_tree = parameters.temporary_local_tree
                                                   ? *parameters.temporary_local_tree
                                                   : internal_local_tree;
    t_particle *particles = parameters.particles;
    const int particle_count = parameters.particle_count;
    const ExecConfig &cfg = parameters.execution;
    MPI_Comm communicator = parameters.communicator;

    distributed_tree.clear();
    temporary_local_tree.clear();

    int local_status = particle_count == cfg.length_per_rank ? 0 : 11;
    if (local_status == 0)
        sort_particles_by_key_cpu(particles, particle_count);

    const int boundary_status = validate_mpi_key_boundaries(
        particles, particle_count, communicator);
    if (local_status == 0)
        local_status = boundary_status;
    const KeyInterval rank_interval = get_rank_key_interval(
        cfg.rank, cfg.nprocs, parameters.splitters);
    if (local_status == 0)
        local_status = build_local_hashed_octree(
            temporary_local_tree, particles, particle_count,
            cfg.rank, &rank_interval);
    std::vector<int> local_branch_indices;
    std::vector<BranchSummary> local_branches;
    if (local_status == 0)
    {
        local_branch_indices = find_local_branch_nodes(
            temporary_local_tree, rank_interval);
        if (particle_count > 0 && local_branch_indices.empty())
            local_status = 20;
        else
            local_branches = pack_local_branches(
                temporary_local_tree, local_branch_indices);
        if (local_branches.size() != local_branch_indices.size())
            local_status = 21;
    }

    int global_status = 0;
    MPI_Allreduce(&local_status, &global_status, 1,
                  MPI_INT, MPI_MAX, communicator);
    std::vector<BranchSummary> all_branches;
    if (global_status == 0)
        local_status = exchange_branch_summaries(
            local_branches, all_branches, communicator);
    else
        local_status = global_status;
    if (local_status == 0)
        local_status = build_distributed_hashed_octree(
            distributed_tree, temporary_local_tree, local_branch_indices,
            all_branches, cfg);
    if (local_status == 0)
        local_status = rebuild_tree_links(distributed_tree);
    if (local_status == 0)
        local_status = compute_global_particle_counts(distributed_tree);

    MPI_Allreduce(&local_status, &global_status, 1,
                  MPI_INT, MPI_MAX, communicator);
    if (global_status == 0)
        local_status = validate_distributed_tree(
            distributed_tree, cfg, all_branches, communicator);
    else
        local_status = global_status;

    // Keep legacy local-tree output free from distributed branch metadata.
    for (TreeNode &node : temporary_local_tree.nodes)
        node.is_branch = false;
    return local_status;
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
