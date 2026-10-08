/**
 * @file particles_gpu.cu
 * @brief GPU particle generation, Morton keys, and MPI redistribution.
 * @details The executable orchestrates this sequence:
 * 1. box_distribution_kernel(), box_x_gradient_kernel(), or torus_distribution_kernel(): generate coordinates.
 * 2. generate_keys_kernel(): calculate Morton keys along the Z-order curve.
 * 3. discover_splitters_gpu(): sort local keys and find global partition boundaries.
 * 4. redistribute_by_splitters_gpu(): exchange particles between MPI ranks when more than one rank participates.
 * 5. build_global_distributed_octree_gpu(): optionally build the tree in distributed_octree.cu.
 * 6. write_par_gpu(): optionally export small particle sets for visualization.
 *
 * The executable skips splitter discovery for a single rank when no tree is requested.
 */
#include "particles_gpu.hcu"

// GPU particle generation and redistribution.

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
        {                                                                                   \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit(cudaStatus);                                                               \
        }                                                                                   \
    }

/**
 * @brief Generate independent uniform coordinates in a cube using Philox.
 *
 * @param particles Device array whose coordinate fields are written.
 * @param N Number of local particles.
 * @param L Cube side length.
 * @param seed Philox seed for this rank.
 */
__global__ void box_distribution_kernel(t_particle *particles, int N, double L, unsigned long long seed)
{
    using RNG = r123::Philox4x32;
    RNG::key_type key = {{(uint32_t)seed, (uint32_t)(seed >> 32)}};

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        RNG::ctr_type ctr = {{(uint32_t)i, 0u, 0u, 0u}};
        RNG::ctr_type r = RNG()(ctr, key);

        particles[i].coord[0] = r123::u01<double>(r.v[0]) * L;
        particles[i].coord[1] = r123::u01<double>(r.v[1]) * L;
        particles[i].coord[2] = r123::u01<double>(r.v[2]) * L;
    }
}

/**
 * @brief Generate a triangular prism by swapping x and z whenever x < z.
 * @details The support is 0 <= z <= x <= L, with 0 <= y <= L.
 * The density is uniform within this wedge; the marginal density increases along x.
 *
 * @param particles Device array whose coordinate fields are written.
 * @param N Number of local particles.
 * @param L Cube side length.
 * @param seed Philox seed for this rank.
 */
__global__ void box_x_gradient_kernel(t_particle *particles, int N, double L, unsigned long long seed)
{
    using RNG = r123::Philox4x32;
    RNG::key_type key = {{(uint32_t)seed, (uint32_t)(seed >> 32)}};
    double x, z, temp;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        RNG::ctr_type ctr = {{(uint32_t)i, 0u, 0u, 0u}};
        RNG::ctr_type r = RNG()(ctr, key);

        x = r123::u01<double>(r.v[0]) * L;
        z = r123::u01<double>(r.v[2]) * L;

        if (x < z){
            temp = x;
            x = z;
            z = temp;
        }
        particles[i].coord[0] = x; //r123::u01<double>(r.v[0]) * L;
        particles[i].coord[1] = r123::u01<double>(r.v[1]) * L;
        particles[i].coord[2] = z; //r123::u01<double>(r.v[2]) * L;
    }
}

/**
 * @brief Generate torus coordinates from a uniform angle and a sampled circular cross section.
 *
 * @param particles Device array whose coordinate fields are written.
 * @param N Number of local particles.
 * @param major_r Distance from the torus axis to the tube center.
 * @param minor_r Tube radius.
 * @param box_length Box side length; the torus is centered at half this value.
 * @param seed Philox seed for this rank.
 */
__global__ void torus_distribution_kernel(t_particle *particles, int N, double major_r, double minor_r, double box_length, unsigned long long seed)
{
    using RNG = r123::Philox4x32;
    RNG::key_type key = {{(uint32_t)seed, (uint32_t)(seed >> 32)}};
    const double TWO_PI = 6.283185307179586476925286766559;
    const double center = box_length * 0.5;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        RNG::ctr_type ctr = {{(uint32_t)i, 0u, 0u, 0u}};
        RNG::ctr_type rnum = RNG()(ctr, key);

        double u0 = r123::u01<double>(rnum.v[0]);
        double u1 = r123::u01<double>(rnum.v[1]);
        double u2 = r123::u01<double>(rnum.v[2]);

        double theta = TWO_PI * u0;
        double phi = TWO_PI * u1;
        double r = minor_r * sqrt(u2);

        double cphi = cos(phi);
        double sphi = sin(phi);
        double cth = cos(theta);
        double sth = sin(theta);

        double Rplus = major_r + r * cphi;

        particles[i].coord[0] = center + Rplus * cth;
        particles[i].coord[1] = center + Rplus * sth;
        particles[i].coord[2] = center + r * sphi;
    }
}

/**
 * @brief Calculate Morton keys (Z-order curve) from particle coordinates.
 * @details Each of MAX_DEPTH levels contributes three bits: x, y, then z.
 * @pre Coordinates lie inside the domain starting at the origin.
 *
 * @param particles Device array; coordinates are read and key fields are written.
 * @param N Number of local particles.
 * @param box_length Side length of the coordinate domain.
 */
__global__ void generate_keys_kernel(t_particle *particles, int N, double box_length)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        double x = particles[i].coord[0];
        double y = particles[i].coord[1];
        double z = particles[i].coord[2];

        double ox = 0.0, oy = 0.0, oz = 0.0;
        double len = box_length;

        unsigned long long key = 0ull;

#pragma unroll 1
        for (int d = 0; d < MAX_DEPTH; ++d)
        {
            len *= 0.5;
            int oct = 0;

            double cx = ox + len;
            double cy = oy + len;
            double cz = oz + len;

            if (x >= cx)
            {
                oct |= 1;
                ox += len;
            }
            if (y >= cy)
            {
                oct |= 2;
                oy += len;
            }
            if (z >= cz)
            {
                oct |= 4;
                oz += len;
            }

            key = (key << 3) | (unsigned long long)oct;
        }

        particles[i].key = (long long)key;
    }
}

/**
 * @brief Assign an MPI owner rank to each particle covered by the launch.
 * @pre The launch provides at least n threads.
 *
 * @param p Device particle array to update.
 * @param n Number of particles.
 * @param rank_id Owner rank to store.
 */
__global__ void set_rank_kernel(t_particle *p, int n, int rank_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i].mpi_rank = rank_id;
}

/**
 * @brief Prints particle coords from device-allocated t_particle array.
 *
 * @param p Particle array on device to print (in).
 * @param n Number of particles (in).
 * @param rank_id Owner rank (in).
 */
__global__ void print_particle_gpu(t_particle *p, int n, int rank_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //if (i == 0) printf("gridDim.x: %d, blockDim.x: %d\n", gridDim.x, blockDim.x);
    if (i < n)
        printf("Rank: %d p[%d].coord: %f, %f, %f\n", rank_id, i, p[i].coord[0], p[i].coord[1], p[i].coord[2]);
}

/**
 * @brief Opens provided PEPC particles file, converts PEPC particle type to t_particle, then populates device buffer.
 * @details device_array is reallocated via cudaMalloc within this function, and filled with particle coordinates from PEPC file.
 *
 * @param comm MPI communicator of participating MPI processes (in).
 * @param filename Filename of the PEPC particles file (in).
 * @param device_array Array that will be populated with PEPC particle coordinates (inout).
 * @param n_total Global particles count (out).
 * @param n_total Particles count per device (out).
 * @param cfg Simulation configurations (in).
 * @param stream CUDA gpu stream for potential async operations(in).
 */
void pepc_distribution(MPI_Comm comm, char *filename, t_particle **device_array, int64_t *n_total, int64_t *n_local, ExecConfig cfg, cudaStream_t stream)
{
    t_pepc_particle *pepc_array;
    t_particle *particle_array;
    int64_t temp_total, temp_local;

    cudaFree(*device_array);
    parallel_read_pepc_particles(comm, &pepc_array, filename, &temp_total, &temp_local);

    pepc_to_simple_particle_array(cfg.rank, &pepc_array, &particle_array, temp_local);
    free(pepc_array);

    //printf("temp_local %lld\n ", temp_local);
    //printf("rank %d p_cpu[0].coords: %f %f %f\n", cfg.rank, particle_array[0].coord[0], particle_array[0].coord[1], particle_array[0].coord[2]);
    cudaMalloc(device_array, (size_t)temp_local*sizeof(t_particle));
    cudaMemcpy(*device_array, particle_array, (size_t)temp_local*sizeof(t_particle), cudaMemcpyHostToDevice);
    free(particle_array);

    *n_total = temp_total;
    *n_local = temp_local;
}

/**
 * @brief Synchronize one stream on each selected local GPU.
 * @note This helper changes the current device and performs no MPI barrier.
 *
 * @param nprocs Number of local devices, indexed from zero; not the MPI process count.
 * @param streams Host vector of streams indexed by device.
 */
void gpu_barrier(int nprocs, const std::vector<cudaStream_t> &streams)
{
    for (int d = 0; d < nprocs; ++d)
    {
        cudaSetDevice(d);
        cudaStreamSynchronize(streams[d]);
    }
}

/**
 * @brief Enable supported peer access between all selected local GPUs.
 * @note This helper changes the current device.
 *
 * @param ndev Number of local devices indexed from zero.
 */
void enable_p2p_all(int ndev)
{
    for (int i = 0; i < ndev; ++i)
    {
        cudaSetDevice(i);
        for (int j = 0; j < ndev; ++j)
        {
            if (i == j)
                continue;

            int can = 0;
            cudaDeviceCanAccessPeer(&can, i, j);

            if (can)
            {
                auto err = cudaDeviceEnablePeerAccess(j, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
                {
                    cudaGetLastError();
                }
            }
        }
    }
}

/**
 * @brief Find particle offsets that partition sorted keys at the splitters.
 * @details Uses upper_bound, so keys equal to a splitter stay in the preceding partition.
 *
 * @param dev Local CUDA device ordinal.
 * @param d_ptr Device array sorted by Morton key.
 * @param n Number of particles.
 * @param splitters Host vector of ordered inclusive upper bounds.
 * @param cuts Output host offsets, including zero and n.
 * @param stream CUDA stream on the selected device.
 */
inline void compute_cuts_for_dev(int dev, t_particle *d_ptr, int n, const std::vector<unsigned long long> &splitters, std::vector<int> &cuts, cudaStream_t stream)
{
    cudaSetDevice(dev);

    cuts.assign(splitters.size() + 2, 0);
    cuts[0] = 0;

    if (n <= 0)
    {
        cuts.back() = 0;
        return;
    }

    thrust::device_ptr<t_particle> first(d_ptr), last(d_ptr + n);
    auto pol = thrust::cuda::par.on(stream);

    for (size_t b = 0; b < splitters.size(); ++b)
    {
        t_particle probe;
        probe.key = (long long)splitters[b];
        auto it = thrust::upper_bound(pol, first, last, probe, key_less{});
        cuts[b + 1] = static_cast<int>(it - first);
    }

    cuts.back() = n;
}

/**
 * @brief Count keys less than or equal to a threshold on a selected device.
 * @return Number of matching particles, or zero for an empty array.
 *
 * @param dev Local CUDA device ordinal.
 * @param d_ptr Device array sorted by Morton key.
 * @param n Number of particles.
 * @param mid Inclusive key threshold.
 * @param stream CUDA stream on the selected device.
 */
long long count_leq_device(int dev, t_particle *d_ptr, int n, unsigned long long mid, cudaStream_t stream)
{
    if (n <= 0)
        return 0;
    cudaSetDevice(dev);
    t_particle probe;
    probe.key = (long long)mid;
    auto pol = thrust::cuda::par.on(stream);
    thrust::device_ptr<t_particle> first(d_ptr), last(d_ptr + n);
    auto it = thrust::upper_bound(pol, first, last, probe, key_less{});
    return static_cast<long long>(it - first);
}

/**
 * @brief Count keys less than or equal to a threshold on the current device.
 * @return Number of matching particles, or zero for an empty array.
 *
 * @param d_ptr Device array sorted by Morton key.
 * @param n Number of particles.
 * @param key Inclusive key threshold.
 * @param stream CUDA stream for the search.
 */
long long count_leq_device2(const t_particle *d_ptr, int n,
                            unsigned long long key, cudaStream_t stream)
{
    if (n <= 0)
        return 0;
    t_particle probe;
    probe.key = (long long)key;
    auto pol = thrust::cuda::par.on(stream);
    thrust::device_ptr<const t_particle> first(d_ptr), last(d_ptr + n);
    auto it = thrust::upper_bound(pol, first, last, probe, key_less{});
    return static_cast<long long>(it - first);
}

struct ExtractKey
{
    /**
     * @brief Read a particle key as an unsigned Morton key.
     * @param p Particle whose key is read.
     * @return The unsigned particle key.
     */
    __host__ __device__ unsigned long long operator()(const t_particle &p) const
    {
        return static_cast<unsigned long long>(p.key);
    }
};

/**
 * @brief Print free and total memory on the current CUDA device.
 *
 * @param tag Label included in the diagnostic.
 */
static inline void dbg_mem(const char *tag)
{
    size_t f, t;
    cudaMemGetInfo(&f, &t);
    fprintf(stderr, "[MEM] %s: free=%.2f GB total=%.2f GB\n", tag, f / 1e9, t / 1e9);
}

/**
 * @brief Sort local particles and find global key quantiles for MPI partitioning.
 * @details Binary searches combine count_leq_device2() results with MPI_Allreduce.
 * Repeated keys can prevent exact particle balance.
 * @pre All ranks in MPI_COMM_WORLD call this routine. The underlying array is writable.
 *
 * @param d_rank_array Device particle array, sorted in place despite its const-qualified pointer.
 * @param lens Number of local particles.
 * @param stream CUDA stream for sorting and searches.
 * @param splitters_out Output host vector of nprocs - 1 splitters; empty if no particles exist.
 */
void discover_splitters_gpu(const t_particle *d_rank_array, int lens, cudaStream_t stream, std::vector<unsigned long long> &splitters_out)
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    {
        auto pol = thrust::cuda::par.on(stream);
        thrust::device_ptr<const t_particle> first(d_rank_array);
        thrust::device_ptr<const t_particle> last(d_rank_array + lens);
        thrust::sort(pol,
                     thrust::device_pointer_cast(const_cast<t_particle *>(d_rank_array)),
                     thrust::device_pointer_cast(const_cast<t_particle *>(d_rank_array) + lens),
                     key_less{});
    }

    unsigned long long local_min = std::numeric_limits<unsigned long long>::max();
    unsigned long long local_max = 0ull;
    if (lens > 0)
    {
        t_particle a, b;
        cudaMemcpyAsync(&a, d_rank_array, sizeof(t_particle), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&b, d_rank_array + (lens - 1), sizeof(t_particle), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        local_min = (unsigned long long)a.key;
        local_max = (unsigned long long)b.key;
    }

    unsigned long long gmin = 0ull, gmax = 0ull;
    MPI_Allreduce(&local_min, &gmin, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &gmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    long long N_local = lens, N_global = 0;
    MPI_Allreduce(&N_local, &N_global, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    if (N_global == 0)
    {
        splitters_out.clear();
        return;
    }

    splitters_out.clear();
    splitters_out.reserve(nprocs ? nprocs - 1 : 0);

    unsigned long long lo_base = gmin;
    for (int i = 1; i < nprocs; ++i)
    {
        const long long target = (N_global * i + nprocs - 1) / nprocs;
        unsigned long long lo = lo_base, hi = gmax;
        while (lo < hi)
        {
            const unsigned long long mid = lo + ((hi - lo) >> 1);
            long long cnt_local = count_leq_device2(d_rank_array, lens, mid, stream);
            long long cnt_global = 0;
            MPI_Allreduce(&cnt_local, &cnt_global, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            if (cnt_global >= target)
                hi = mid;
            else
                lo = mid + 1;
        }
        splitters_out.push_back(lo);
        lo_base = lo;
    }
}

/**
 * @brief Exchange particles so each MPI rank owns its assigned Morton interval.
 * @details MPI_Alltoall exchanges counts; chunked MPI_Sendrecv exchanges device buffers.
 * Received blocks are concatenated and are not necessarily locally sorted.
 * @pre Input particles are sorted. All ranks participate using CUDA-aware MPI.
 *
 * @param d_rank_array Device allocation to update; may be replaced if capacity is insufficient.
 * @param lens Input local count, replaced with the received count.
 * @param capacity Allocated particle capacity, updated when storage grows.
 * @param splitters Host vector of nprocs - 1 ordered inclusive partition bounds.
 * @param stream CUDA stream for device operations.
 */
void redistribute_by_splitters_gpu(t_particle **d_rank_array, int *lens, int *capacity,
                                   const std::vector<unsigned long long> &splitters, cudaStream_t stream)
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::vector<long long> pos(nprocs + 1, 0);
    {
        auto pol = thrust::cuda::par.on(stream);
        thrust::device_ptr<t_particle> first(*d_rank_array), last(*d_rank_array + *lens);
        for (int i = 0; i < nprocs - 1; ++i)
        {
            t_particle probe;
            probe.key = (long long)splitters[i];
            auto it = thrust::upper_bound(pol, first, last, probe, key_less{});
            pos[i + 1] = static_cast<long long>(it - first);
        }
        pos[nprocs] = static_cast<long long>(*lens);
    }

    std::vector<long long> send_counts64(nprocs, 0);
    for (int r = 0; r < nprocs; ++r)
    {
        long long start = pos[r];
        long long end = pos[r + 1];
        long long c = end - start;
        send_counts64[r] = (c > 0) ? c : 0;
    }

    std::vector<long long> recv_counts64(nprocs, 0);
    MPI_Alltoall(send_counts64.data(), 1, MPI_LONG_LONG,
                 recv_counts64.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);

    std::vector<long long> send_displs64(nprocs, 0), recv_displs64(nprocs, 0);
    for (int i = 1; i < nprocs; ++i)
    {
        send_displs64[i] = send_displs64[i - 1] + send_counts64[i - 1];
        recv_displs64[i] = recv_displs64[i - 1] + recv_counts64[i - 1];
    }
    const long long total_recv64 = (nprocs ? recv_displs64.back() + recv_counts64.back() : 0);

    const size_t elem_size = sizeof(t_particle);
    t_particle *d_tmp = nullptr;
    size_t tmp_bytes = (total_recv64 > 0) ? (size_t)total_recv64 * elem_size : 1;
    CUDA_RT_CALL(cudaMallocAsync((void **)&d_tmp, tmp_bytes, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    const long long CHUNK_MAX_BYTES = (long long)INT_MAX - ((long long)INT_MAX % (long long)elem_size);

    for (int p = 0; p < nprocs; ++p)
    {
        long long to_send_e = send_counts64[p];
        long long to_recv_e = recv_counts64[p];
        long long s_off_e = send_displs64[p];
        long long r_off_e = recv_displs64[p];

        long long sent_e = 0, recvd_e = 0;
        while (sent_e < to_send_e || recvd_e < to_recv_e)
        {
            long long send_chunk_e = std::min<long long>(to_send_e - sent_e, CHUNK_MAX_BYTES / (long long)elem_size);
            long long recv_chunk_e = std::min<long long>(to_recv_e - recvd_e, CHUNK_MAX_BYTES / (long long)elem_size);

            const void *sb = static_cast<const void *>((const char *)(*d_rank_array) + (s_off_e + sent_e) * elem_size);
            void *rb = static_cast<void *>((char *)d_tmp + (r_off_e + recvd_e) * elem_size);

            int send_chunk_b = (int)(send_chunk_e * (long long)elem_size);
            int recv_chunk_b = (int)(recv_chunk_e * (long long)elem_size);

            MPI_Sendrecv(sb, send_chunk_b, MPI_BYTE, p, 0,
                         rb, recv_chunk_b, MPI_BYTE, p, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            sent_e += send_chunk_e;
            recvd_e += recv_chunk_e;
        }
    }

    if (total_recv64 > (long long)*capacity)
    {
        if (total_recv64 > (long long)std::numeric_limits<int>::max())
        {
            fprintf(stderr, "ERROR: total_recv64 (%lld) exceeds INT_MAX; increase type of 'capacity'/'lens'.\n",
                    (long long)total_recv64);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (*d_rank_array)
            cudaFree(*d_rank_array);
        CUDA_RT_CALL(cudaMalloc((void **)d_rank_array, (size_t)total_recv64 * elem_size));
        *capacity = (int)total_recv64;
    }

    if (total_recv64 > 0)
        CUDA_RT_CALL(cudaMemcpyAsync(*d_rank_array, d_tmp, (size_t)total_recv64 * elem_size,
                                     cudaMemcpyDeviceToDevice, stream));
    *lens = (int)total_recv64;

    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    CUDA_RT_CALL(cudaFreeAsync(d_tmp, stream));

    DBG_IF({
        if (*lens > 0)
        {
            const int block = 256;
            const int grid = (*lens + block - 1) / block;
            set_rank_kernel<<<grid, block, 0, stream>>>(*d_rank_array, *lens, rank);
            cudaStreamSynchronize(stream);
        }
    });

    DBG_PRINT("After disrank %d: %d\n", rank, *lens);
}

/**
 * @brief Gather particle records and write a small debug .par file on rank zero.
 * @details Output is written in the working directory only when cfg.power < 4.
 * @pre All ranks in MPI_COMM_WORLD participate. MPI must accept CUDA device buffers.
 *
 * @param cfg Execution configuration, including rank count and output size.
 * @param d_rank_array Local device particle array.
 * @param lens Number of local particles.
 * @param gpu_stream CUDA stream synchronized before gathering.
 */
void write_par_gpu(const ExecConfig &cfg,
                   t_particle *d_rank_array,
                   int lens,
                   cudaStream_t gpu_stream)
{
    t_particle *h_host_array = nullptr;

    cudaMallocHost(&h_host_array, (size_t)cfg.length_per_rank * sizeof(t_particle));
    if (h_host_array)
    {
        cudaFreeHost(h_host_array);
        h_host_array = nullptr;
    }

    const size_t bytes = (size_t)lens * sizeof(t_particle);
    if (lens > 0)
    {
        cudaMallocHost(&h_host_array, bytes);
        cudaMemcpyAsync(h_host_array, d_rank_array, bytes, cudaMemcpyDeviceToHost, gpu_stream);
    }

    cudaStreamSynchronize(gpu_stream);
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> recv_lens;
    if (cfg.rank == 0)
        recv_lens.resize(cfg.nprocs);
    MPI_Gather(&lens, 1, MPI_INT, cfg.rank == 0 ? recv_lens.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> recv_counts, recv_displs;
    size_t total_count = 0;
    if (cfg.rank == 0)
    {
        recv_counts.resize(cfg.nprocs);
        recv_displs.resize(cfg.nprocs);
        for (int i = 0; i < cfg.nprocs; ++i)
        {
            recv_counts[i] = recv_lens[i] * (int)sizeof(t_particle);
        }
        recv_displs[0] = 0;
        for (int i = 1; i < cfg.nprocs; ++i)
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        total_count = (size_t)recv_displs.back() + (size_t)recv_counts.back();
    }

    std::vector<unsigned char> gather_buf(cfg.rank == 0 ? total_count : 0);
    MPI_Gatherv(d_rank_array, lens * (int)sizeof(t_particle), MPI_BYTE,
                cfg.rank == 0 ? gather_buf.data() : nullptr,
                cfg.rank == 0 ? recv_counts.data() : nullptr,
                cfg.rank == 0 ? recv_displs.data() : nullptr,
                MPI_BYTE, 0, MPI_COMM_WORLD);

    if (cfg.rank == 0)
    {
        if (cfg.power < 4)
        {
            char filename[128];
            std::sprintf(filename, "particle_file_gpu_n%d_total%lld.par", cfg.nprocs, cfg.total_particles);
            std::vector<t_particle *> host_ptrs(cfg.nprocs, nullptr);
            for (int i = 0; i < cfg.nprocs; ++i)
                host_ptrs[i] = reinterpret_cast<t_particle *>(gather_buf.data() + recv_displs[i]);
            int rc = concat_and_serial_write(host_ptrs.data(), recv_lens.data(), cfg.nprocs, filename);
            if (rc != 0)
            {
                std::cerr << "Error at writing file, rc=" << rc << "\n";
            }
        }
    }

    if (h_host_array)
        cudaFreeHost(h_host_array);
}
