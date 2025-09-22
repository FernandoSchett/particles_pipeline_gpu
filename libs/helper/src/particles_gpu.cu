#include "particles_gpu.hcu"

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

__global__ void set_rank_kernel(t_particle *p, int n, int rank_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i].mpi_rank = rank_id;
}

void gpu_barrier(int nprocs, const std::vector<cudaStream_t> &streams)
{
    for (int d = 0; d < nprocs; ++d)
    {
        cudaSetDevice(d);
        cudaStreamSynchronize(streams[d]);
    }
}

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
    __host__ __device__ unsigned long long operator()(const t_particle &p) const
    {
        return static_cast<unsigned long long>(p.key);
    }
};

static inline void dbg_mem(const char *tag)
{
    size_t f, t;
    cudaMemGetInfo(&f, &t);
    fprintf(stderr, "[MEM] %s: free=%.2f GB total=%.2f GB\n", tag, f / 1e9, t / 1e9);
}

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
        pos[nprocs] = *lens;
    }

    std::vector<int> send_counts(nprocs, 0);
    for (int r = 0; r < nprocs; ++r)
    {
        long long start = pos[r];
        long long end = pos[r + 1];
        long long c = end - start;
        send_counts[r] = (c > 0) ? static_cast<int>(c) : 0;
    }

    std::vector<int> recv_counts(nprocs, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> send_displs(nprocs, 0), recv_displs(nprocs, 0);
    for (int i = 1; i < nprocs; ++i)
    {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }
    const int total_send = send_displs.back() + send_counts.back();
    const int total_recv = recv_displs.back() + recv_counts.back();

    t_particle *d_tmp = nullptr;
    if (total_recv > 0)
        cudaMallocAsync(&d_tmp, (size_t)total_recv * sizeof(t_particle), stream);
    else
        cudaMallocAsync(&d_tmp, 1, stream);

    std::vector<int> send_counts_bytes(nprocs), recv_counts_bytes(nprocs),
        send_displs_bytes(nprocs), recv_displs_bytes(nprocs);
    for (int i = 0; i < nprocs; ++i)
    {
        send_counts_bytes[i] = send_counts[i] * (int)sizeof(t_particle);
        recv_counts_bytes[i] = recv_counts[i] * (int)sizeof(t_particle);
        send_displs_bytes[i] = send_displs[i] * (int)sizeof(t_particle);
        recv_displs_bytes[i] = recv_displs[i] * (int)sizeof(t_particle);
    }

    cudaStreamSynchronize(stream);
    MPI_Alltoallv(
        *d_rank_array, send_counts_bytes.data(), send_displs_bytes.data(), MPI_BYTE,
        d_tmp, recv_counts_bytes.data(), recv_displs_bytes.data(), MPI_BYTE,
        MPI_COMM_WORLD);

    if (total_recv > *capacity)
    {
        if (*d_rank_array)
            cudaFree(*d_rank_array);
        cudaMalloc(d_rank_array, (size_t)total_recv * sizeof(t_particle));
        *capacity = total_recv;
    }

    if (total_recv > 0)
        cudaMemcpyAsync(*d_rank_array, d_tmp, (size_t)total_recv * sizeof(t_particle),
                        cudaMemcpyDeviceToDevice, stream);
    *lens = total_recv;

    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_tmp, stream);

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