#include "distributed_octree.hcu"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace
{
constexpr int block_size = 256;
constexpr unsigned long long invalid_key = std::numeric_limits<unsigned long long>::max();

int grid_size(int count)
{
    return std::max(1, (count + block_size - 1) / block_size);
}

int check_cuda(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return 0;

    std::fprintf(stderr, "CUDA error during %s: %s\n",
                 operation, cudaGetErrorString(status));
    return static_cast<int>(status);
}

__device__ int lower_bound_particle_key(const t_particle *particles,
                                        int count,
                                        unsigned long long key)
{
    int first = 0;
    int length = count;
    while (length > 0)
    {
        const int half = length >> 1;
        const int middle = first + half;
        if (static_cast<unsigned long long>(particles[middle].key) < key)
        {
            first = middle + 1;
            length -= half + 1;
        }
        else
        {
            length = half;
        }
    }
    return first;
}

__global__ void cornerstone_histogram_kernel(const t_particle *particles,
                                              int particle_count,
                                              const unsigned long long *cornerstone,
                                              int leaf_count,
                                              unsigned long long *counts)
{
    for (int leaf = blockIdx.x * blockDim.x + threadIdx.x;
         leaf < leaf_count;
         leaf += blockDim.x * gridDim.x)
    {
        const int first = lower_bound_particle_key(particles, particle_count,
                                                   cornerstone[leaf]);
        const int last = lower_bound_particle_key(particles, particle_count,
                                                  cornerstone[leaf + 1]);
        counts[leaf] = static_cast<unsigned long long>(last - first);
    }
}

__global__ void rebalance_decision_kernel(const unsigned long long *cornerstone,
                                          const unsigned long long *global_counts,
                                          int leaf_count,
                                          unsigned long long ncrit,
                                          int *decisions)
{
    for (int leaf = blockIdx.x * blockDim.x + threadIdx.x;
         leaf < leaf_count;
         leaf += blockDim.x * gridDim.x)
    {
        const unsigned long long range = cornerstone[leaf + 1] - cornerstone[leaf];
        decisions[leaf] = global_counts[leaf] > ncrit && range > 1 ? 8 : 1;
    }
}

__global__ void rebalance_cornerstone_kernel(const unsigned long long *old_cornerstone,
                                             const int *decisions,
                                             const int *offsets,
                                             int old_leaf_count,
                                             unsigned long long *new_cornerstone)
{
    for (int leaf = blockIdx.x * blockDim.x + threadIdx.x;
         leaf < old_leaf_count;
         leaf += blockDim.x * gridDim.x)
    {
        const int output = offsets[leaf];
        if (decisions[leaf] == 1)
        {
            new_cornerstone[output] = old_cornerstone[leaf];
            continue;
        }

        const unsigned long long child_range =
            (old_cornerstone[leaf + 1] - old_cornerstone[leaf]) / 8;
        for (int child = 0; child < 8; ++child)
            new_cornerstone[output + child] =
                old_cornerstone[leaf] + static_cast<unsigned long long>(child) * child_range;
    }
}

__global__ void generate_node_candidates_kernel(const unsigned long long *cornerstone,
                                                int leaf_count,
                                                unsigned long long *candidates)
{
    const int levels_per_leaf = MAX_DEPTH + 1;
    const int candidate_count = leaf_count * levels_per_leaf;
    for (int candidate = blockIdx.x * blockDim.x + threadIdx.x;
         candidate < candidate_count;
         candidate += blockDim.x * gridDim.x)
    {
        const int leaf = candidate / levels_per_leaf;
        const int level = candidate % levels_per_leaf;
        const unsigned long long range = cornerstone[leaf + 1] - cornerstone[leaf];
        const int range_log2 = __ffsll(static_cast<long long>(range)) - 1;
        const int leaf_level = MAX_DEPTH - range_log2 / 3;

        if (level > leaf_level)
        {
            candidates[candidate] = invalid_key;
            continue;
        }

        const int shift = 3 * (MAX_DEPTH - level);
        const unsigned long long prefix = cornerstone[leaf] >> shift;
        candidates[candidate] = (1ull << (3 * level)) | prefix;
    }
}

__device__ int lower_bound_node_key(const unsigned long long *keys,
                                    int first,
                                    int last,
                                    unsigned long long key)
{
    int length = last - first;
    while (length > 0)
    {
        const int half = length >> 1;
        const int middle = first + half;
        if (keys[middle] < key)
        {
            first = middle + 1;
            length -= half + 1;
        }
        else
        {
            length = half;
        }
    }
    return first;
}

__global__ void connectivity_kernel(const unsigned long long *node_keys,
                                    int node_count,
                                    const int *level_offsets,
                                    int *child_offsets)
{
    for (int node = blockIdx.x * blockDim.x + threadIdx.x;
         node < node_count;
         node += blockDim.x * gridDim.x)
    {
        const unsigned long long key = node_keys[node];
        const int level = (63 - __clzll(key)) / 3;
        if (level >= MAX_DEPTH)
        {
            child_offsets[node] = 0;
            continue;
        }

        const int first = level_offsets[level + 1];
        const int last = level_offsets[level + 2];
        const unsigned long long first_child_key = key << 3;
        const int child = lower_bound_node_key(node_keys, first, last, first_child_key);
        child_offsets[node] = child < last && node_keys[child] == first_child_key ? child : 0;
    }
}

int copy_vector_to_raw(const thrust::device_vector<unsigned long long> &source,
                       unsigned long long **destination,
                       cudaStream_t stream)
{
    const std::size_t bytes = source.size() * sizeof(unsigned long long);
    int status = check_cuda(cudaMallocAsync(reinterpret_cast<void **>(destination), bytes, stream),
                            "octree allocation");
    if (status != 0)
        return status;
    return check_cuda(cudaMemcpyAsync(*destination,
                                      thrust::raw_pointer_cast(source.data()),
                                      bytes, cudaMemcpyDeviceToDevice, stream),
                      "octree copy");
}

int copy_vector_to_raw(const thrust::device_vector<int> &source,
                       int **destination,
                       cudaStream_t stream)
{
    const std::size_t bytes = source.size() * sizeof(int);
    int status = check_cuda(cudaMallocAsync(reinterpret_cast<void **>(destination), bytes, stream),
                            "octree allocation");
    if (status != 0)
        return status;
    return check_cuda(cudaMemcpyAsync(*destination,
                                      thrust::raw_pointer_cast(source.data()),
                                      bytes, cudaMemcpyDeviceToDevice, stream),
                      "octree copy");
}

template <typename T>
void append_binary(std::vector<unsigned char> &buffer, const T &value)
{
    const std::size_t offset = buffer.size();
    buffer.resize(offset + sizeof(T));
    std::memcpy(buffer.data() + offset, &value, sizeof(T));
}
}

void free_distributed_octree_gpu(DistributedGpuOctree &tree,
                                 cudaStream_t stream)
{
    if (tree.d_cornerstone)
        cudaFreeAsync(tree.d_cornerstone, stream);
    if (tree.d_node_keys)
        cudaFreeAsync(tree.d_node_keys, stream);
    if (tree.d_child_offsets)
        cudaFreeAsync(tree.d_child_offsets, stream);
    if (tree.d_level_offsets)
        cudaFreeAsync(tree.d_level_offsets, stream);

    tree = {};
}

int build_global_distributed_octree_gpu(DistributedGpuOctree &tree,
                                        t_particle *d_particles,
                                        int local_particle_count,
                                        int ncrit,
                                        cudaStream_t stream,
                                        MPI_Comm communicator)
{
    static_assert(3 * MAX_DEPTH < 63, "MAX_DEPTH does not fit in a 64-bit Morton key");
    if (local_particle_count < 0 || (local_particle_count > 0 && d_particles == nullptr) || ncrit < 1)
        return 1;

    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    const auto policy = thrust::cuda::par.on(stream);

    if (local_particle_count > 1)
    {
        thrust::sort(policy, d_particles, d_particles + local_particle_count, key_less{});
        int status = check_cuda(cudaStreamSynchronize(stream), "particle key sort");
        if (status != 0)
            return status;
    }

    const unsigned long long key_space_end = 1ull << (3 * MAX_DEPTH);
    thrust::device_vector<unsigned long long> cornerstone(2);
    cornerstone[0] = 0;
    cornerstone[1] = key_space_end;

    for (int iteration = 0; iteration <= MAX_DEPTH; ++iteration)
    {
        const int leaf_count = static_cast<int>(cornerstone.size()) - 1;
        thrust::device_vector<unsigned long long> local_counts(leaf_count);
        cornerstone_histogram_kernel<<<grid_size(leaf_count), block_size, 0, stream>>>(
            d_particles, local_particle_count,
            thrust::raw_pointer_cast(cornerstone.data()), leaf_count,
            thrust::raw_pointer_cast(local_counts.data()));

        int status = check_cuda(cudaGetLastError(), "cornerstone histogram launch");
        if (status != 0)
            return status;

        std::vector<unsigned long long> host_local_counts(leaf_count);
        std::vector<unsigned long long> host_global_counts(leaf_count);
        status = check_cuda(cudaMemcpyAsync(host_local_counts.data(),
                                            thrust::raw_pointer_cast(local_counts.data()),
                                            static_cast<std::size_t>(leaf_count) * sizeof(unsigned long long),
                                            cudaMemcpyDeviceToHost, stream),
                            "cornerstone histogram copy");
        if (status != 0)
            return status;
        status = check_cuda(cudaStreamSynchronize(stream), "cornerstone histogram synchronization");
        if (status != 0)
            return status;

        const int mpi_status = MPI_Allreduce(host_local_counts.data(), host_global_counts.data(),
                                             leaf_count, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                                             communicator);
        if (mpi_status != MPI_SUCCESS)
            return 2;

        thrust::device_vector<unsigned long long> global_counts(host_global_counts);
        thrust::device_vector<int> decisions(leaf_count);
        rebalance_decision_kernel<<<grid_size(leaf_count), block_size, 0, stream>>>(
            thrust::raw_pointer_cast(cornerstone.data()),
            thrust::raw_pointer_cast(global_counts.data()), leaf_count,
            static_cast<unsigned long long>(ncrit),
            thrust::raw_pointer_cast(decisions.data()));
        status = check_cuda(cudaGetLastError(), "cornerstone rebalance decision launch");
        if (status != 0)
            return status;

        const int split_count = static_cast<int>(thrust::count(policy,
                                                                decisions.begin(), decisions.end(), 8));
        if (split_count == 0)
            break;

        thrust::device_vector<int> offsets(leaf_count);
        thrust::exclusive_scan(policy, decisions.begin(), decisions.end(), offsets.begin());
        const int new_leaf_count = thrust::reduce(policy, decisions.begin(), decisions.end(), 0);
        thrust::device_vector<unsigned long long> updated_cornerstone(
            static_cast<std::size_t>(new_leaf_count) + 1);

        rebalance_cornerstone_kernel<<<grid_size(leaf_count), block_size, 0, stream>>>(
            thrust::raw_pointer_cast(cornerstone.data()),
            thrust::raw_pointer_cast(decisions.data()),
            thrust::raw_pointer_cast(offsets.data()), leaf_count,
            thrust::raw_pointer_cast(updated_cornerstone.data()));
        status = check_cuda(cudaGetLastError(), "cornerstone rebalance launch");
        if (status != 0)
            return status;
        status = check_cuda(cudaMemcpyAsync(
                                thrust::raw_pointer_cast(updated_cornerstone.data()) + new_leaf_count,
                                &key_space_end, sizeof(key_space_end), cudaMemcpyHostToDevice, stream),
                            "cornerstone endpoint copy");
        if (status != 0)
            return status;

        cornerstone.swap(updated_cornerstone);
    }

    const int leaf_count = static_cast<int>(cornerstone.size()) - 1;
    if (leaf_count > INT_MAX / (MAX_DEPTH + 1))
        return 3;
    const int candidate_count = leaf_count * (MAX_DEPTH + 1);
    thrust::device_vector<unsigned long long> node_keys(candidate_count);
    generate_node_candidates_kernel<<<grid_size(candidate_count), block_size, 0, stream>>>(
        thrust::raw_pointer_cast(cornerstone.data()), leaf_count,
        thrust::raw_pointer_cast(node_keys.data()));
    int status = check_cuda(cudaGetLastError(), "node key generation launch");
    if (status != 0)
        return status;

    thrust::sort(policy, node_keys.begin(), node_keys.end());
    auto unique_end = thrust::unique(policy, node_keys.begin(), node_keys.end());
    auto valid_end = thrust::lower_bound(policy, node_keys.begin(), unique_end, invalid_key);
    node_keys.resize(static_cast<std::size_t>(valid_end - node_keys.begin()));
    const int node_count = static_cast<int>(node_keys.size());

    std::vector<unsigned long long> level_markers(MAX_DEPTH + 1);
    for (int level = 0; level <= MAX_DEPTH; ++level)
        level_markers[level] = 1ull << (3 * level);
    thrust::device_vector<unsigned long long> device_level_markers(level_markers);
    thrust::device_vector<int> level_offsets(MAX_DEPTH + 2);
    thrust::lower_bound(policy,
                        node_keys.begin(), node_keys.end(),
                        device_level_markers.begin(), device_level_markers.end(),
                        level_offsets.begin());
    level_offsets[MAX_DEPTH + 1] = node_count;

    thrust::device_vector<int> child_offsets(node_count, 0);
    connectivity_kernel<<<grid_size(node_count), block_size, 0, stream>>>(
        thrust::raw_pointer_cast(node_keys.data()), node_count,
        thrust::raw_pointer_cast(level_offsets.data()),
        thrust::raw_pointer_cast(child_offsets.data()));
    status = check_cuda(cudaGetLastError(), "octree connectivity launch");
    if (status != 0)
        return status;
    status = check_cuda(cudaStreamSynchronize(stream), "global octree construction");
    if (status != 0)
        return status;

    free_distributed_octree_gpu(tree, stream);
    status = copy_vector_to_raw(cornerstone, &tree.d_cornerstone, stream);
    if (status == 0)
        status = copy_vector_to_raw(node_keys, &tree.d_node_keys, stream);
    if (status == 0)
        status = copy_vector_to_raw(child_offsets, &tree.d_child_offsets, stream);
    if (status == 0)
        status = copy_vector_to_raw(level_offsets, &tree.d_level_offsets, stream);
    if (status != 0)
    {
        free_distributed_octree_gpu(tree, stream);
        return status;
    }

    tree.leaf_count = leaf_count;
    tree.node_count = node_count;
    tree.max_depth = MAX_DEPTH;
    tree.ncrit = ncrit;
    status = check_cuda(cudaStreamSynchronize(stream), "global octree final copy");
    if (status != 0)
        return status;

    if (rank == 0)
        std::printf("[GLOBAL GPU TREE] leaves=%d nodes=%d ncrit=%d depth=%d\n",
                    leaf_count, node_count, ncrit, MAX_DEPTH);
    return 0;
}

int write_global_distributed_octree_gpu(const DistributedGpuOctree &tree,
                                         const t_particle *d_particles,
                                         int local_particle_count,
                                         const char *filename,
                                         cudaStream_t stream,
                                         MPI_Comm communicator)
{
    constexpr char magic[8] = {'P', 'S', 'F', 'C', 'G', 'T', 'R', 'E'};
    constexpr std::uint32_t version = 1;

    if (!filename || tree.leaf_count < 1 || tree.node_count < 1 ||
        !tree.d_cornerstone || !tree.d_node_keys || !tree.d_child_offsets ||
        !tree.d_level_offsets || local_particle_count < 0 ||
        (local_particle_count > 0 && !d_particles))
        return 1;

    int rank = 0;
    MPI_Comm_rank(communicator, &rank);

    thrust::device_vector<unsigned long long> local_counts(tree.leaf_count);
    cornerstone_histogram_kernel<<<grid_size(tree.leaf_count), block_size, 0, stream>>>(
        d_particles, local_particle_count, tree.d_cornerstone, tree.leaf_count,
        thrust::raw_pointer_cast(local_counts.data()));
    int status = check_cuda(cudaGetLastError(), "global tree dump histogram launch");
    if (status != 0)
        return status;

    std::vector<unsigned long long> host_local_counts(tree.leaf_count);
    std::vector<unsigned long long> global_counts(tree.leaf_count);
    status = check_cuda(cudaMemcpyAsync(
                            host_local_counts.data(),
                            thrust::raw_pointer_cast(local_counts.data()),
                            static_cast<std::size_t>(tree.leaf_count) * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost, stream),
                        "global tree dump histogram copy");
    if (status != 0)
        return status;
    status = check_cuda(cudaStreamSynchronize(stream), "global tree dump synchronization");
    if (status != 0)
        return status;

    if (MPI_Allreduce(host_local_counts.data(), global_counts.data(), tree.leaf_count,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator) != MPI_SUCCESS)
        return 2;

    unsigned long long local_total = static_cast<unsigned long long>(local_particle_count);
    unsigned long long global_total = 0;
    if (MPI_Allreduce(&local_total, &global_total, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, communicator) != MPI_SUCCESS)
        return 2;

    int write_status = 0;
    if (rank == 0)
    {
        std::vector<unsigned long long> cornerstone(
            static_cast<std::size_t>(tree.leaf_count) + 1);
        std::vector<unsigned long long> node_keys(tree.node_count);
        std::vector<int> child_offsets(tree.node_count);
        std::vector<int> level_offsets(static_cast<std::size_t>(tree.max_depth) + 2);

        status = check_cuda(cudaMemcpyAsync(
                                cornerstone.data(), tree.d_cornerstone,
                                cornerstone.size() * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost, stream),
                            "global tree dump cornerstone copy");
        if (status == 0)
            status = check_cuda(cudaMemcpyAsync(
                                    node_keys.data(), tree.d_node_keys,
                                    node_keys.size() * sizeof(unsigned long long),
                                    cudaMemcpyDeviceToHost, stream),
                                "global tree dump node key copy");
        if (status == 0)
            status = check_cuda(cudaMemcpyAsync(
                                    child_offsets.data(), tree.d_child_offsets,
                                    child_offsets.size() * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream),
                                "global tree dump child offset copy");
        if (status == 0)
            status = check_cuda(cudaMemcpyAsync(
                                    level_offsets.data(), tree.d_level_offsets,
                                    level_offsets.size() * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream),
                                "global tree dump level offset copy");
        if (status == 0)
            status = check_cuda(cudaStreamSynchronize(stream), "global tree dump copies");
        if (status != 0)
        {
            write_status = status;
        }
        else
        {
            std::vector<unsigned char> output;
            output.reserve(48 + cornerstone.size() * sizeof(unsigned long long) +
                           global_counts.size() * sizeof(unsigned long long) +
                           node_keys.size() * sizeof(unsigned long long) +
                           child_offsets.size() * sizeof(int) +
                           level_offsets.size() * sizeof(int));
            output.insert(output.end(), magic, magic + sizeof(magic));
            append_binary(output, version);
            append_binary(output, static_cast<std::uint32_t>(tree.max_depth));
            append_binary(output, static_cast<std::uint32_t>(tree.ncrit));
            append_binary(output, std::uint32_t{0});
            append_binary(output, static_cast<std::uint64_t>(global_total));
            append_binary(output, static_cast<std::uint64_t>(tree.leaf_count));
            append_binary(output, static_cast<std::uint64_t>(tree.node_count));
            for (const auto value : cornerstone)
                append_binary(output, static_cast<std::uint64_t>(value));
            for (const auto value : global_counts)
                append_binary(output, static_cast<std::uint64_t>(value));
            for (const auto value : node_keys)
                append_binary(output, static_cast<std::uint64_t>(value));
            for (const auto value : child_offsets)
                append_binary(output, static_cast<std::int32_t>(value));
            for (const auto value : level_offsets)
                append_binary(output, static_cast<std::int32_t>(value));

            FILE *file = std::fopen(filename, "wb");
            if (!file || std::fwrite(output.data(), 1, output.size(), file) != output.size())
                write_status = 3;
            if (file && std::fclose(file) != 0)
                write_status = 3;
        }
    }

    MPI_Bcast(&write_status, 1, MPI_INT, 0, communicator);
    MPI_Barrier(communicator);
    return write_status;
}
