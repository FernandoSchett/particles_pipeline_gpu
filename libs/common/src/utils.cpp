#include "utils.hpp"
#include "particle_domain.hpp"
#include <cstdio>
#include <ctime>
#include <sys/stat.h>
#include <cmath>

void print_particles(t_particle *particle_array, int size, int rank)
{
    for (int i = 0; i < size; i++)
    {
        printf("P_rank: %d, %d, %f, %f, %f, key: %lld, key_bin: ",
               rank, particle_array[i].mpi_rank,
               particle_array[i].coord[0],
               particle_array[i].coord[1],
               particle_array[i].coord[2],
               particle_array[i].key);

        for (int b = 63; b >= 0; b--)
        {
            printf("%lld", (particle_array[i].key >> b) & 1LL);
        }

        printf("\n");
    }
}

void print_pepc_particles(t_pepc_particle *particle_array, int size, int rank)
{
    for (int i = 0; i < size; i++)
    {
        printf("P_rank: %d, %f, %f, %f, m: %f ",
               rank,
               particle_array[i].x[0],
               particle_array[i].x[1],
               particle_array[i].x[2],
               particle_array[i].data.m);

        printf("\n");
    }
}

int fit_particle_domain(const t_particle *particles, int64_t count,
                        ExecConfig &cfg, MPI_Comm comm)
{
    const double infinity = std::numeric_limits<double>::infinity();
    double lower[3] = {infinity, infinity, infinity};
    double upper[3] = {-infinity, -infinity, -infinity};
    int invalid = 0;
    for (int64_t i = 0; i < count; ++i)
        for (int axis = 0; axis < 3; ++axis)
        {
            const double value = particles[i].coord[axis];
            invalid |= !std::isfinite(value);
            lower[axis] = std::min(lower[axis], value);
            upper[axis] = std::max(upper[axis], value);
        }
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, comm);
    if (invalid)
        return 1;
    MPI_Allreduce(MPI_IN_PLACE, lower, 3, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, upper, 3, MPI_DOUBLE, MPI_MAX, comm);
    double length = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        cfg.box_origin[axis] = lower[axis] == infinity ? 0.0 : lower[axis];
        if (lower[axis] != infinity)
            length = std::max(length, upper[axis] - lower[axis]);
    }
    // Empty and coincident particle sets still need a positive domain size.
    cfg.box_length = length == 0.0 ? 1.0 : std::nextafter(length, infinity);
    return std::isfinite(cfg.box_length) ? 0 : 1;
}

void setup_particles_box_length(ExecConfig &cfg)
{
    const long long base = static_cast<long long>(std::pow(10.0, cfg.power));
    const long long n = static_cast<long long>(cfg.nprocs);
    const long long T = n * (n + 1) / 2;

    cfg.box_length = std::pow(10.0, cfg.power);
    cfg.major_r = static_cast<int>(4 * std::pow(10.0, cfg.power - 1));
    cfg.minor_r = static_cast<int>(2 * std::pow(10.0, cfg.power - 1));

    long long target_total = base;
    if (cfg.exp_type == WEAK_SCALING)
        target_total = base * n;

    const long long slice = target_total / T;
    const long long rem = target_total - slice * T;

    cfg.total_particles = target_total;
    cfg.length_per_rank = static_cast<int>(
        (static_cast<long long>(cfg.rank + 1) * slice) +
        ((cfg.rank == cfg.nprocs - 1) ? rem : 0));

    cfg.ram_gb = (cfg.length_per_rank * 40.0) / 1e9;

    if (cfg.rank == 0)
    {
        const char *mode_str = (cfg.exp_type == WEAK_SCALING) ? "weak" : "strong";
        std::printf("%lld particles (%s), triangular across %d processes "
                    "(len_k ≈ (k+1)*%lld, remainder=%lld) in a %.1f box; per-rank RAM ~ %.4f GB.\n",
                    cfg.total_particles, mode_str, cfg.nprocs, slice, rem,
                    cfg.box_length, cfg.ram_gb);
    }
}

void log_results(const ExecConfig &cfg, const exec_times &times, const char *results_path)
{
    time_t rawtime;
    std::tm *ti;
    char ts[64];
    std::time(&rawtime);
    ti = std::localtime(&rawtime);
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", ti);

    struct stat buffer;
    const int exists = (stat(results_path, &buffer) == 0);
    FILE *f = std::fopen(results_path, "a");
    if (!exists)
        std::fprintf(f, "datetime,power,total_particles,length_per_rank,num_procs,box_length,RAM_GB,alloc_time,gen_time,splitters_time,dist_time,tree_time,total_time,device,seed,mode\n");

    const char *mode_str = (cfg.exp_type == WEAK_SCALING) ? "weak" : "strong";

    std::fprintf(f, "%s,%d,%lld,%d,%d,%.1f,%.2f,%f,%f,%f,%f,%f,%f,%s,%d,%s\n",
                 ts, cfg.power, cfg.total_particles, cfg.length_per_rank, cfg.nprocs,
                 cfg.box_length, cfg.ram_gb,
                 times.alloc_time, times.gen_time, times.splitters_time, times.dist_time,
                 times.tree_time, times.total_time,
                 cfg.device, cfg.seed, mode_str);

    std::printf("%s,%d,%lld,%d,%d,%.1f,%.2f,%f,%f,%f,%f,%f,%f,%s,%d,%s\n",
                ts, cfg.power, cfg.total_particles, cfg.length_per_rank, cfg.nprocs,
                cfg.box_length, cfg.ram_gb,
                times.alloc_time, times.gen_time, times.splitters_time, times.dist_time,
                times.tree_time, times.total_time,
                cfg.device, cfg.seed, mode_str);
}
