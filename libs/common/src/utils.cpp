#include "utils.hpp"
#include "particle_domain.hpp"
#include <cstdio>
#include <ctime>
#include <sys/stat.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sys/file.h>
#include <unistd.h>

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

void setup_output_directory(ExecConfig &cfg)
{
    if (const char *directory = std::getenv("P_SFC_OUTPUT_DIR"))
        if (*directory)
            cfg.output_dir = directory;
    int failed = 0;
    if (cfg.rank == 0)
    {
        std::error_code error;
        std::filesystem::create_directories(cfg.output_dir, error);
        failed = static_cast<bool>(error);
        if (failed)
            std::fprintf(stderr, "Cannot create output directory %s: %s\n",
                         cfg.output_dir, error.message().c_str());
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (failed)
        MPI_Abort(MPI_COMM_WORLD, 1);
}

void log_results(const ExecConfig &cfg, const exec_times &times, const char *results_path)
{
    const time_t now = std::time(nullptr);
    std::tm local_time{};
    char timestamp[64];
    if (!localtime_r(&now, &local_time) ||
        !std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &local_time))
    {
        std::fprintf(stderr, "Cannot format results timestamp.\n");
        return;
    }

    const char *mode = cfg.exp_type == WEAK_SCALING ? "weak" : "strong";
    char row[1024];
    const int length = std::snprintf(row, sizeof(row),
        "%s,%d,%lld,%d,%d,%.17g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%s,%d,%s\n",
        timestamp, cfg.power, cfg.total_particles, cfg.length_per_rank, cfg.nprocs,
        cfg.box_length, cfg.ram_gb, times.alloc_time, times.gen_time,
        times.splitters_time, times.dist_time, times.tree_time, times.total_time,
        cfg.device, cfg.seed, mode);
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(row))
    {
        std::fprintf(stderr, "Results CSV row exceeds buffer size.\n");
        return;
    }
    std::fputs(row, stdout);

    FILE *file = std::fopen(results_path, "a");
    if (!file)
    {
        std::perror(results_path);
        return;
    }
    // Serialize cooperating jobs writing the same CSV, including its header.
    if (flock(fileno(file), LOCK_EX) != 0)
    {
        std::perror("Lock results CSV");
        std::fclose(file);
        return;
    }
    struct stat info{};
    bool failed = fstat(fileno(file), &info) != 0;
    if (!failed && info.st_size == 0)
        failed = std::fputs("datetime,power,total_particles,length_per_rank,num_procs,box_length,RAM_GB,alloc_time,gen_time,splitters_time,dist_time,tree_time,total_time,device,seed,mode\n", file) == EOF;
    if (!failed)
        failed = std::fputs(row, file) == EOF;
    if (std::fflush(file) != 0)
        failed = true;
    // fclose releases the lock after flushing and closes the descriptor.
    if (std::fclose(file) != 0)
        failed = true;
    if (failed)
        std::fprintf(stderr, "Failed to write results CSV: %s\n", results_path);
}
