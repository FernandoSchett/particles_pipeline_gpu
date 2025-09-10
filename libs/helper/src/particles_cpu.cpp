#include "particles_cpu.hpp"

MPI_Datatype MPI_particle;
int register_MPI_Particle(MPI_Datatype *MPI_Particle)
{
    int blocklengths[NPROPS_PARTICLE] = {1, 1, 3};
    MPI_Datatype array_types[NPROPS_PARTICLE] = {MPI_INT, MPI_LONG_LONG_INT, MPI_DOUBLE};
    t_particle dummy_particle[2];
    MPI_Aint address[NPROPS_PARTICLE + 1];
    MPI_Aint displacements[NPROPS_PARTICLE];
    MPI_Aint extent_add;
    int i, type_size;

    MPI_Get_address(&dummy_particle[0], &address[0]);
    MPI_Get_address(&dummy_particle[0].mpi_rank, &address[1]);
    MPI_Get_address(&dummy_particle[0].key, &address[2]);
    MPI_Get_address(&dummy_particle[0].coord, &address[3]);

    for (int i = 0; i < NPROPS_PARTICLE; i++)
    {
        displacements[i] = address[i + 1] - address[0];
    }

    MPI_Type_create_struct(NPROPS_PARTICLE, blocklengths, displacements, array_types, MPI_Particle);

    MPI_Get_address(&dummy_particle[1], &extent_add);
    extent_add = extent_add - address[0];
    MPI_Type_create_resized(*MPI_Particle, 0, extent_add, MPI_Particle);
    MPI_Type_size(*MPI_Particle, &type_size);

    MPI_Type_commit(MPI_Particle);
    return 0;
}

int allocate_particle(t_particle **particle_array, int count)
{
    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    (*particle_array) = (t_particle *)malloc(count * sizeof(t_particle));

    for (int i = 0; i < count; i++)
    {
        (*particle_array)[i].mpi_rank = p_rank;
        (*particle_array)[i].key = 0;
        (*particle_array)[i].coord[0] = 0.0;
        (*particle_array)[i].coord[1] = 0.0;
        (*particle_array)[i].coord[2] = 0.0;
    }

    return 0;
}

int box_distribution(t_particle **particle_array, int count, double box_length)
{
    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);

    typedef r123::Philox4x32 RNG;
    RNG rng;
    RNG::ctr_type c = {{}};
    RNG::ukey_type uk = {{}};
    uk[0] = p_rank; // some user_supplied_seed
    RNG::key_type k = uk;
    RNG::ctr_type r;

    c[0] = 07072025;
    c[1] = 31106712;

    for (int i = 0; i < count; i++)
    {
        c[0] += 1;
        c[1] += 1;
        r = rng(c, k);
        (*particle_array)[i].coord[0] = r123::u01<double>(r.v[0]) * box_length;

        c[0] += 1;
        c[1] += 1;
        r = rng(c, k);
        (*particle_array)[i].coord[1] = r123::u01<double>(r.v[0]) * box_length;

        c[0] += 1;
        c[1] += 1;
        r = rng(c, k);
        (*particle_array)[i].coord[2] = r123::u01<double>(r.v[0]) * box_length;
    }
    return 0;
}

int torus_distribution(t_particle **particle_array, int count, double major_r, double minor_r, double box_length)
{
    int p_rank;
    double theta, phi, r;
    double center = box_length / 2.0;

    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);

    typedef r123::Philox4x32 RNG;
    RNG rng;
    RNG::ctr_type c = {{}};
    RNG::ukey_type uk = {{}};
    uk[0] = p_rank;
    RNG::key_type k = uk;
    RNG::ctr_type rnum;

    c[0] = 25082025;
    c[1] = 85712394;

    for (int i = 0; i < count; i++)
    {
        c[0] += 1;
        c[1] += 1;
        rnum = rng(c, k);
        theta = 2.0 * M_PI * r123::u01<double>(rnum.v[0]);

        c[0] += 1;
        c[1] += 1;
        rnum = rng(c, k);
        phi = 2.0 * M_PI * r123::u01<double>(rnum.v[0]);

        c[0] += 1;
        c[1] += 1;
        rnum = rng(c, k);
        r = minor_r * sqrt(r123::u01<double>(rnum.v[0]));

        (*particle_array)[i].coord[0] = center + (major_r + r * cos(phi)) * cos(theta);
        (*particle_array)[i].coord[1] = center + (major_r + r * cos(phi)) * sin(theta);
        (*particle_array)[i].coord[2] = center + r * sin(phi);
    }

    return 0;
}

int generate_particles_keys(t_particle *particle_array, int count, double box_length)
{
    std::vector<t_particle *> particles;
    particles.reserve(count);

    for (int i = 0; i < count; i++)
    {
        particles.push_back(&particle_array[i]);
    }

    std::array<double, 3> origin = {0.0, 0.0, 0.0};
    run_oct_tree_recursive(particles, 0, 0, box_length, origin);

    return 0;
}

void run_oct_tree_recursive(std::vector<t_particle *> &particles, int depth, long long key_prefix, double box_length, const std::array<double, 3> &origin)
{

    // std::cout << "Call: count=" << particles.size() << " depth=" << depth << " prefix=" << key_prefix << "\n";

    if (particles.empty())
        return;

    if (depth >= MAX_DEPTH)
    {
        for (auto *p : particles)
        {
            p->key = key_prefix;
        }
        // std::cout << "MAX_DEPTH\n";
        return;
    }

    double half = box_length / 2.0;
    std::array<double, 3> center = {
        origin[0] + half,
        origin[1] + half,
        origin[2] + half};

    std::vector<t_particle *> octants[8];

    for (auto *p : particles)
    {
        int oct = 0;
        if (p->coord[0] >= center[0])
            oct |= 1;
        if (p->coord[1] >= center[1])
            oct |= 2;
        if (p->coord[2] >= center[2])
            oct |= 4;
        octants[oct].push_back(p);
    }

    for (int i = 0; i < 8; i++)
    {
        if (!octants[i].empty())
        {
            long long new_key = (key_prefix << 3) | i;

            std::array<double, 3> new_origin = {
                origin[0] + (i & 1 ? half : 0),
                origin[1] + (i & 2 ? half : 0),
                origin[2] + (i & 4 ? half : 0)};

            run_oct_tree_recursive(octants[i], depth + 1, new_key, half, new_origin);
        }
    }
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

int distribute_particles(t_particle **particles, int *particle_vector_size, int nprocs)
{
    boost::sort::spreadsort::integer_sort(*particles, *particles + *particle_vector_size, particle_rightshift{}, particle_less{});

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int local_n = *particle_vector_size;

    long long N_local = local_n, N_global = 0;
    MPI_Allreduce(&N_local, &N_global, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    unsigned long long local_min = std::numeric_limits<unsigned long long>::max();
    unsigned long long local_max = 0;
    if (local_n > 0)
    {
        local_min = (unsigned long long)((*particles)[0].key);
        local_max = (unsigned long long)((*particles)[local_n - 1].key);
    }
    unsigned long long gmin, gmax;
    MPI_Allreduce(&local_min, &gmin, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &gmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    std::vector<unsigned long long> splitters;
    splitters.reserve(nprocs > 0 ? nprocs - 1 : 0);

    unsigned long long lo_base = gmin;
    for (int i = 1; i < nprocs; ++i)
    {
        long long target = (N_global * i + nprocs - 1) / nprocs;

        unsigned long long lo = lo_base;
        unsigned long long hi = gmax;

        while (lo < hi)
        {
            unsigned long long mid = lo + ((hi - lo) >> 1);
            long long c_local = count_leq(*particles, local_n, mid);
            long long c_global = 0;
            MPI_Allreduce(&c_local, &c_global, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            if (c_global >= target)
                hi = mid;
            else
                lo = mid + 1;
        }
        splitters.push_back(lo);
        lo_base = lo;
    }

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
                auto it = std::upper_bound(
                    *particles, *particles + local_n, probe, key_less);
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

    printf("Rank %d, Number Particles: %d\n", rank, *particle_vector_size);
    return 0;
}
