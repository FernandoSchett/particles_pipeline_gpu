#include "particles_cpu.hpp"

#include <cstdlib>

// CPU particle generation and redistribution.

int allocate_particle(t_particle **particle_array, int count)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    *particle_array = static_cast<t_particle *>(std::malloc(count * sizeof(t_particle)));
    if (count > 0 && *particle_array == nullptr)
        return 1;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
    {
        (*particle_array)[i].mpi_rank = rank;
        (*particle_array)[i].key = 0;
        (*particle_array)[i].coord[0] = 0.0;
        (*particle_array)[i].coord[1] = 0.0;
        (*particle_array)[i].coord[2] = 0.0;
    }

    return 0;
}

int box_distribution(t_particle **particle_array, int count, double box_length, int seed)
{
    using RNG = r123::Philox4x32;
    RNG::key_type key = {{(uint32_t)seed, 0u}};

#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
    {
        RNG::ctr_type ctr = {{(uint32_t)i, 0u, 0u, 0u}};
        RNG::ctr_type rnum = RNG()(ctr, key);

        double ux = r123::u01<double>(rnum.v[0]);
        double uy = r123::u01<double>(rnum.v[1]);
        double uz = r123::u01<double>(rnum.v[2]);

        (*particle_array)[i].coord[0] = ux * box_length;
        (*particle_array)[i].coord[1] = uy * box_length;
        (*particle_array)[i].coord[2] = uz * box_length;
    }
    return 0;
}

int torus_distribution(t_particle **particle_array, int count, double major_r, double minor_r, double box_length, int seed)
{
    using RNG = r123::Philox4x32;
    RNG::key_type key = {{(uint32_t)seed, 0u}};
    const double TWO_PI = 6.283185307179586476925286766559;
    const double center = box_length * 0.5;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
    {
        RNG::ctr_type ctr = {{(uint32_t)i, 0u, 0u, 0u}};
        RNG::ctr_type rnum = RNG()(ctr, key);

        double u0 = r123::u01<double>(rnum.v[0]);
        double u1 = r123::u01<double>(rnum.v[1]);
        double u2 = r123::u01<double>(rnum.v[2]);

        double theta = TWO_PI * u0;
        double phi = TWO_PI * u1;
        double r = minor_r * std::sqrt(u2);

        double cphi = std::cos(phi);
        double sphi = std::sin(phi);
        double cth = std::cos(theta);
        double sth = std::sin(theta);

        double Rplus = major_r + r * cphi;

        (*particle_array)[i].coord[0] = center + Rplus * cth;
        (*particle_array)[i].coord[1] = center + Rplus * sth;
        (*particle_array)[i].coord[2] = center + r * sphi;
    }
    return 0;
}

int generate_particles_keys(t_particle *particle_array, int count, double box_length)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
    {
        t_particle &particle = particle_array[i];
        double origin_x = 0.0;
        double origin_y = 0.0;
        double origin_z = 0.0;
        double cell_size = box_length;
        unsigned long long key = 0;

        for (int depth = 0; depth < MAX_DEPTH; ++depth)
        {
            const double half = cell_size * 0.5;
            int octant = 0;

            if (particle.coord[0] >= origin_x + half)
                octant |= 1;
            if (particle.coord[1] >= origin_y + half)
                octant |= 2;
            if (particle.coord[2] >= origin_z + half)
                octant |= 4;

            key = (key << 3) | static_cast<unsigned long long>(octant);
            if (octant & 1)
                origin_x += half;
            if (octant & 2)
                origin_y += half;
            if (octant & 4)
                origin_z += half;
            cell_size = half;
        }

        particle.key = static_cast<long long>(key);
    }

    return 0;
}

static inline bool key_less(const t_particle &a, const t_particle &b)
{
    return (unsigned long long)a.key < (unsigned long long)b.key;
}

static inline long long count_leq(const t_particle *particles, int n, unsigned long long val)
{
    if (n <= 0)
        return 0;
    t_particle probe;
    probe.key = (long long)val;

    const t_particle *first = particles;
    const t_particle *last = particles + n;

    auto it = std::upper_bound(first, last, probe, key_less);
    return (long long)(it - first);
}

struct particle_less
{
    inline bool operator()(const t_particle &a, const t_particle &b) const
    {
        return (unsigned long long)a.key < (unsigned long long)b.key;
    }
};

struct particle_rightshift
{
    inline unsigned long long operator()(const t_particle &p, unsigned offset) const
    {
        return ((unsigned long long)p.key) >> offset;
    }
};

void sort_particles_by_key_cpu(t_particle *particles, int count)
{
    if (count < 2)
        return;

    boost::sort::spreadsort::integer_sort(
        particles, particles + count, particle_rightshift{}, particle_less{});
}

void discover_splitters_cpu(t_particle *particles, int local_n,
                            std::vector<unsigned long long> &splitters_out)
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    sort_particles_by_key_cpu(particles, local_n);

    long long N_local = local_n, N_global = 0;
    MPI_Allreduce(&N_local, &N_global, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    if (N_global == 0)
    {
        splitters_out.clear();
        return;
    }

    unsigned long long local_min = std::numeric_limits<unsigned long long>::max();
    unsigned long long local_max = 0ull;
    if (local_n > 0)
    {
        local_min = (unsigned long long)particles[0].key;
        local_max = (unsigned long long)particles[local_n - 1].key;
    }
    unsigned long long gmin = 0ull, gmax = 0ull;
    MPI_Allreduce(&local_min, &gmin, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &gmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    splitters_out.clear();
    splitters_out.reserve(nprocs > 0 ? nprocs - 1 : 0);

    unsigned long long lo_base = gmin;
    for (int i = 1; i < nprocs; ++i)
    {
        const long long target = (N_global * i + nprocs - 1) / nprocs;
        unsigned long long lo = lo_base, hi = gmax;
        while (lo < hi)
        {
            const unsigned long long mid = lo + ((hi - lo) >> 1);
            long long c_local = count_leq(particles, local_n, mid);
            long long c_global = 0;
            MPI_Allreduce(&c_local, &c_global, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            if (c_global >= target)
                hi = mid;
            else
                lo = mid + 1;
        }
        splitters_out.push_back(lo);
        lo_base = lo;
    }
}

int redistribute_by_splitters_cpu(t_particle **particles,
                                  int *particle_vector_size,
                                  const std::vector<unsigned long long> &splitters)
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int local_n = *particle_vector_size;

    std::vector<int> sendcounts(nprocs, 0), sdispls(nprocs, 0);
    if (nprocs == 1)
    {
        sendcounts[0] = local_n;
    }
    else
    {
        std::vector<int> cuts;
        cuts.reserve(nprocs + 1);
        cuts.push_back(0);
        for (unsigned long long s : splitters)
        {
            t_particle probe;
            probe.key = (long long)s;
            const t_particle *first = *particles;
            const t_particle *last = *particles + local_n;
            auto it = std::upper_bound(first, last, probe, key_less);
            cuts.push_back((int)(it - first));
        }
        cuts.push_back(local_n);
        for (int b = 0; b < nprocs; ++b)
        {
            int begin = cuts[b], end = cuts[b + 1];
            sendcounts[b] = std::max(0, end - begin);
        }
    }
    for (int i = 1; i < nprocs; ++i)
        sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];

    std::vector<int> recvcounts(nprocs, 0), rdispls(nprocs, 0);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 1; i < nprocs; ++i)
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];

    int recv_total = 0;
    for (int x : recvcounts)
        recv_total += x;

    std::vector<t_particle> sendbuf(sdispls.back() + sendcounts.back());
    if (local_n > 0)
    {
        if (nprocs == 1)
        {
            std::memcpy(sendbuf.data(), *particles, local_n * sizeof(t_particle));
        }
        else
        {
            std::vector<int> cuts;
            cuts.reserve(nprocs + 1);
            cuts.push_back(0);
            for (unsigned long long s : splitters)
            {
                t_particle probe;
                probe.key = (long long)s;
                auto it = std::upper_bound(*particles, *particles + local_n, probe, key_less);
                cuts.push_back((int)(it - *particles));
            }
            cuts.push_back(local_n);

            for (int b = 0; b < nprocs; ++b)
            {
                int begin = cuts[b], end = cuts[b + 1];
                int amt = end - begin;
                if (amt > 0)
                {
                    std::memcpy(sendbuf.data() + sdispls[b],
                                *particles + begin,
                                amt * sizeof(t_particle));
                }
            }
        }
    }

    std::vector<t_particle> recvbuf(recv_total);
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_particle,
                  recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_particle,
                  MPI_COMM_WORLD);

    for (auto &p : recvbuf)
        p.mpi_rank = rank;

    free(*particles);
    t_particle *newbuf = (t_particle *)malloc(recvbuf.size() * sizeof(t_particle));
    std::memcpy(newbuf, recvbuf.data(), recvbuf.size() * sizeof(t_particle));
    *particles = newbuf;
    *particle_vector_size = (int)recvbuf.size();

    DBG_PRINT("Rank %d, Number Particles: %d\n", rank, *particle_vector_size);
    return 0;
}

void write_par_cpu(const ExecConfig &cfg,
                   t_particle *rank_array,
                   int *length_vector)
{

    char filename[128];
    std::sprintf(filename, "particle_file_cpu_n%d_total%lld.par", cfg.nprocs, cfg.total_particles);
    MPI_Allgather(&cfg.length_per_rank, 1, MPI_INT, length_vector, 1, MPI_INT, MPI_COMM_WORLD);
    parallel_write_to_file(rank_array, length_vector, filename);
}
