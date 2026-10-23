#include "particle_domain.hpp"
#include "file_handling.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ExecConfig cfg{};
    int failures = 0;
    auto check = [&](bool valid) { failures += !valid; };

    // Rank zero owns both extrema; every other rank is empty.
    t_particle sample[2]{};
    sample[0].coord[0] = -10; sample[0].coord[1] = 5; sample[0].coord[2] = -2;
    sample[1].coord[0] = 20; sample[1].coord[1] = 7; sample[1].coord[2] = 3;
    check(fit_particle_domain(sample, rank == 0 ? 2 : 0, cfg, MPI_COMM_WORLD) == 0);
    check(cfg.box_origin[0] == -10 && cfg.box_origin[1] == 5 && cfg.box_origin[2] == -2);
    check(cfg.box_length > 30 && cfg.box_length < 31);
    check(sample[0].coord[0] == -10 && sample[1].coord[0] == 20);

    check(fit_particle_domain(sample, 0, cfg, MPI_COMM_WORLD) == 0);
    check(cfg.box_length == 1 && cfg.box_origin[0] == 0);
    check(fit_particle_domain(sample, 1, cfg, MPI_COMM_WORLD) == 0);
    check(cfg.box_length == 1 && cfg.box_origin[0] == -10);

    sample[0].coord[0] = std::numeric_limits<double>::quiet_NaN();
    check(fit_particle_domain(sample, rank == 0 ? 1 : 0, cfg, MPI_COMM_WORLD) != 0);

    if (argc > 1)
    {
        t_pepc_particle *input = nullptr;
        t_particle *particles = nullptr;
        int64_t total = 0, count = 0;
        if (parallel_read_pepc_particles(MPI_COMM_WORLD, &input, argv[1], &total, &count) != 0)
            MPI_Abort(MPI_COMM_WORLD, 1);
        pepc_to_simple_particle_array(rank, &input, &particles, count);
        check(total == 10000);
        check(fit_particle_domain(particles, count, cfg, MPI_COMM_WORLD) == 0);
        check(std::abs(cfg.box_origin[0] - (-22490.512274739398)) < 1e-8);
        check(std::abs(cfg.box_length - 45081.157732007385) < 1e-8);
        for (int64_t i = 0; i < count; ++i)
            for (int axis = 0; axis < 3; ++axis)
            {
                const double offset = particles[i].coord[axis] - cfg.box_origin[axis];
                check(offset >= 0 && offset < cfg.box_length);
                check(particles[i].coord[axis] == input[i].x[axis]);
            }
        std::free(input);
        std::free(particles);
        MPI_Type_free(&MPI_particle_pepc);
        MPI_Type_free(&MPI_particle_data);
        MPI_Type_free(&MPI_particle_results);
    }
    int total_failures = 0;
    MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("Particle domain checks: %s\n", total_failures ? "FAIL" : "PASS");
    MPI_Finalize();
    return total_failures ? 1 : 0;
}
