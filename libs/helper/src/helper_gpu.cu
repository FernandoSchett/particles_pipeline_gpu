#include "./helper.hpp"

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

int concat_and_serial_write(t_particle **arrays, const int *counts, int nprocs, const char *filename)
{
    long long total_ll = 0;
    for (int d = 0; d < nprocs; ++d)
    {
        if (counts[d] < 0)
            return 1;
        total_ll += (long long)counts[d];
    }

    if (total_ll > std::numeric_limits<int>::max())
    {
        std::fprintf(stderr, "[E] total particles > INT_MAX (%lld)\n", total_ll);
        return 2;
    }
    const int total = (int)total_ll;

    std::vector<t_particle> tmp;
    tmp.reserve((size_t)total);

    for (int d = 0; d < nprocs; ++d)
    {
        const int n = counts[d];
        if (n <= 0)
            continue;
        if (!arrays[d])
            return 3;

        tmp.insert(tmp.end(), arrays[d], arrays[d] + n);
    }
    return serial_write_to_file(tmp.data(), total, const_cast<char *>(filename));
}

inline void compute_cuts_for_dev(
    int dev,
    t_particle *d_ptr,
    int n,
    const std::vector<unsigned long long> &splitters,
    std::vector<int> &cuts,
    cudaStream_t stream)
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

long long count_leq_device(
    int dev, t_particle *d_ptr, int n, unsigned long long mid, cudaStream_t stream)
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

int distribute_gpu_particles(std::vector<t_particle *> &d_rank_array, std::vector<int> &lens, std::vector<cudaStream_t> &gpu_streams)
{
    const int nprocs = (int)d_rank_array.size();
    for (int dev = 0; dev < nprocs; ++dev)
    {
        cudaSetDevice(dev);
        int n = lens[dev];
        if (n <= 0)
            continue;
        auto pol = thrust::cuda::par.on(gpu_streams[dev]);
        thrust::device_ptr<t_particle> first(d_rank_array[dev]), last(d_rank_array[dev] + n);
        thrust::sort(pol, first, last, key_less{});
    }

    gpu_barrier(nprocs, gpu_streams);

    std::vector<unsigned long long> local_min(nprocs, std::numeric_limits<unsigned long long>::max());
    std::vector<unsigned long long> local_max(nprocs, 0ull);
    for (int dev = 0; dev < nprocs; ++dev)
    {
        int n = lens[dev];
        if (n <= 0)
            continue;
        t_particle first_h{}, last_h{};
        cudaMemcpy(&first_h, d_rank_array[dev], sizeof(t_particle), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_h, d_rank_array[dev] + (n - 1), sizeof(t_particle), cudaMemcpyDeviceToHost);
        local_min[dev] = (unsigned long long)first_h.key;
        local_max[dev] = (unsigned long long)last_h.key;
    }

    unsigned long long gmin = std::numeric_limits<unsigned long long>::max();
    unsigned long long gmax = 0ull;
    long long N_global = 0;
    for (int dev = 0; dev < nprocs; ++dev)
    {
        if (lens[dev] > 0)
        {
            gmin = std::min(gmin, local_min[dev]);
            gmax = std::max(gmax, local_max[dev]);
            N_global += lens[dev];
        }
    }
    if (N_global == 0)
        return 0;

    std::vector<unsigned long long> splitters;
    splitters.reserve(nprocs ? nprocs - 1 : 0);
    unsigned long long lo_base = gmin;
    for (int i = 1; i < nprocs; ++i)
    {
        const long long target = (N_global * i + nprocs - 1) / nprocs;
        unsigned long long lo = lo_base, hi = gmax;
        long long c_global = 0;
        while (lo < hi)
        {
            unsigned long long mid = lo + ((hi - lo) >> 1);
            c_global = 0;
            for (int dev = 0; dev < nprocs; ++dev)
            {
                if (lens[dev] == 0)
                    continue;
                c_global += count_leq_device(dev, d_rank_array[dev], lens[dev], mid, gpu_streams[dev]);
            }
            if (c_global >= target)
                hi = mid;
            else
                lo = mid + 1;
        }
        // printf("C GLOBAL: %lld\n", c_global);
        // printf("TARGET: %lld\n", target);
        // printf("lo: %lu\n", lo);
        // printf("high: %lu\n", hi);

        splitters.push_back(lo);
        lo_base = lo;
    }

    std::vector<std::vector<int>> sendcounts(nprocs, std::vector<int>(nprocs, 0));
    std::vector<std::vector<int>> cuts(nprocs);
    for (int src = 0; src < nprocs; ++src)
    {
        if (lens[src] == 0)
        {
            cuts[src] = std::vector<int>(nprocs + 1, 0);
            continue;
        }
        compute_cuts_for_dev(src, d_rank_array[src], lens[src], splitters, cuts[src], gpu_streams[src]);
    }

    gpu_barrier(nprocs, gpu_streams);

    for (int src = 0; src < nprocs; ++src)
        for (int b = 0; b < nprocs; ++b)
        {
            int begin = cuts[src][b];
            int end = cuts[src][b + 1];
            sendcounts[src][b] = std::max(0, end - begin);
        }

    std::vector<std::vector<int>> recvcounts(nprocs, std::vector<int>(nprocs, 0));
    for (int dst = 0; dst < nprocs; ++dst)
        for (int src = 0; src < nprocs; ++src)
            recvcounts[dst][src] = sendcounts[src][dst];

    auto prefix = [&](const std::vector<int> &v)
    {
        std::vector<int> p(v.size(), 0);
        for (size_t i = 1; i < v.size(); ++i)
            p[i] = p[i - 1] + v[i - 1];
        return p;
    };

    std::vector<std::vector<int>> sdispls(nprocs), rdispls(nprocs);
    for (int src = 0; src < nprocs; ++src)
        sdispls[src] = prefix(sendcounts[src]);
    for (int dst = 0; dst < nprocs; ++dst)
        rdispls[dst] = prefix(recvcounts[dst]);

    std::vector<int> recv_tot(nprocs, 0);
    for (int dst = 0; dst < nprocs; ++dst)
        recv_tot[dst] = std::accumulate(recvcounts[dst].begin(), recvcounts[dst].end(), 0);

    std::vector<t_particle *> d_new(nprocs, nullptr);
    for (int dst = 0; dst < nprocs; ++dst)
    {
        cudaSetDevice(dst);
        if (recv_tot[dst] > 0)
            cudaMallocAsync(&d_new[dst], (size_t)recv_tot[dst] * sizeof(t_particle), gpu_streams[dst]);
    }

    for (int src = 0; src < nprocs; ++src)
        for (int dst = 0; dst < nprocs; ++dst)
        {
            int cnt = sendcounts[src][dst];
            if (cnt <= 0)
                continue;
            const int begin_src = cuts[src][dst];
            const size_t bytes = (size_t)cnt * sizeof(t_particle);
            t_particle *src_ptr = d_rank_array[src] + begin_src;
            t_particle *dst_ptr = d_new[dst] + rdispls[dst][src];
            cudaSetDevice(dst);
            cudaMemcpyPeerAsync(dst_ptr, dst, src_ptr, src, bytes, gpu_streams[dst]);
        }

    gpu_barrier(nprocs, gpu_streams);

    for (int dev = 0; dev < nprocs; ++dev)
    {
        cudaSetDevice(dev);
        if (d_rank_array[dev])
            cudaFreeAsync(d_rank_array[dev], gpu_streams[dev]);
        d_rank_array[dev] = d_new[dev];
        lens[dev] = recv_tot[dev];
    }

    const int block = 256;
    for (int dev = 0; dev < nprocs; ++dev)
    {
        cudaSetDevice(dev);
        int n = lens[dev];
        printf("After distribution %d:  %d\n", dev, lens[dev]);
        if (n == 0)
            continue;
        int grid = (n + block - 1) / block;
        set_rank_kernel<<<grid, block, 0, gpu_streams[dev]>>>(d_rank_array[dev], n, dev);
    }

    gpu_barrier(nprocs, gpu_streams);
    return 0;
}

static long long count_leq_device2(t_particle *d_ptr, int n, unsigned long long key, cudaStream_t stream)
{
    if (n <= 0)
        return 0;
    t_particle probe;
    probe.key = (long long)key;
    auto pol = thrust::cuda::par.on(stream);
    thrust::device_ptr<t_particle> first(d_ptr), last(d_ptr + n);
    auto it = thrust::upper_bound(pol, first, last, probe, key_less{});
    return static_cast<long long>(it - first);
}

void distribute_gpu_particles_mpi(t_particle **d_rank_array, int *lens, int *capacity, cudaStream_t stream)
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    {
        auto pol = thrust::cuda::par.on(stream);
        thrust::device_ptr<t_particle> first(*d_rank_array), last(*d_rank_array + *lens);
        thrust::sort(pol, first, last, key_less{});
    }

    unsigned long long local_min = std::numeric_limits<unsigned long long>::max();
    unsigned long long local_max = 0ull;
    if (*lens > 0)
    {
        t_particle a, b;
        cudaMemcpyAsync(&a, *d_rank_array, sizeof(t_particle), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&b, *d_rank_array + (*lens - 1), sizeof(t_particle), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        local_min = (unsigned long long)a.key;
        local_max = (unsigned long long)b.key;
    }

    unsigned long long gmin = 0ull, gmax = 0ull;
    MPI_Allreduce(&local_min, &gmin, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &gmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    long long N_local = *lens, N_global = 0;
    MPI_Allreduce(&N_local, &N_global, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    if (N_global == 0)
        return;

    std::vector<unsigned long long> splitters;
    splitters.reserve(nprocs ? nprocs - 1 : 0);
    unsigned long long lo_base = gmin;
    for (int i = 1; i < nprocs; ++i)
    {
        const long long target = (N_global * i + nprocs - 1) / nprocs;
        unsigned long long lo = lo_base, hi = gmax;
        while (lo < hi)
        {
            const unsigned long long mid = lo + ((hi - lo) >> 1);
            long long cnt_local = count_leq_device2(*d_rank_array, *lens, mid, stream);
            long long cnt_global = 0;
            MPI_Allreduce(&cnt_local, &cnt_global, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            if (cnt_global >= target)
                hi = mid;
            else
                lo = mid + 1;
        }
        splitters.push_back(lo);
        lo_base = lo;
    }

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

    std::vector<int> send_counts_bytes(nprocs), recv_counts_bytes(nprocs), send_displs_bytes(nprocs), recv_displs_bytes(nprocs);
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
    {
        cudaMemcpyAsync(*d_rank_array, d_tmp, (size_t)total_recv * sizeof(t_particle), cudaMemcpyDeviceToDevice, stream);
    }
    *lens = total_recv;

    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_tmp, stream);

    if (*lens > 0)
    {
        const int block = 256;
        const int grid = (*lens + block - 1) / block;
        set_rank_kernel<<<grid, block, 0, stream>>>(*d_rank_array, *lens, rank);
        cudaStreamSynchronize(stream);
    }

    printf("rank %d: %d\n", rank, *lens);
}
