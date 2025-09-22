// utils.cpp
#include "utils.hpp"
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

void setup_particles_box_length(int power, int nprocs, int rank,
                                int *length_per_rank, double *box_length,
                                long long *total_particles, double *RAM_GB,
                                int *major_r, int *minor_r,
                                exp_type_t exp_type)
{
    const long long base = static_cast<long long>(std::pow(10.0, power));

    if (exp_type == WEAK_SCALING)
        *total_particles = base * nprocs;
    else
        *total_particles = base;

    const long long T = static_cast<long long>(nprocs) * (nprocs + 1) / 2;
    const long long slice = base / T;
    const long long rem = base - slice * T;

    *box_length = std::pow(10.0, power);
    *length_per_rank = static_cast<int>((static_cast<long long>(rank + 1) * slice) +
                                        ((rank == nprocs - 1) ? rem : 0));
    *major_r = static_cast<int>(4 * std::pow(10.0, power - 1));
    *minor_r = static_cast<int>(2 * std::pow(10.0, power - 1));
    *RAM_GB = (*total_particles * 40.0) / 1e9;

    if (rank == 0)
    {
        std::printf("%s scaling: %lld particles distributed like (rank_number*%lld) "
                    "between %d processes in a %.1f sized box using %.4f GBs.\n",
                    (exp_type == WEAK_SCALING ? "WEAK" : "STRONG"),
                    *total_particles, slice, nprocs, *box_length, *RAM_GB);
    }
}



void log_results(int rank, int power, long long total_particles, int length_per_rank, int nprocs, double box_length, double RAM_GB, double gen_time, double dist_time, const char *device_type, int seed, exp_type_t exp_type)
{
	time_t rawtime;
	std::tm *timeinfo;
	char time_str[64];

	std::time(&rawtime);
	timeinfo = std::localtime(&rawtime);
	std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", timeinfo);

	struct stat buffer;
	const char *results_path = "../results.csv";
	const int file_exists = (stat(results_path, &buffer) == 0);

	FILE *f = std::fopen(results_path, "a");
	if (!file_exists)
	{
		std::fprintf(f, "datetime,power,total_particles,length_per_rank,num_procs,box_length,RAM_GB,gen_time,dist_time,device,seed,mode\n");
	}

	const char *mode_str = (exp_type == WEAK_SCALING) ? "weak" : "strong";

	std::fprintf(f, "%s,%d,%lld,%d,%d,%.1f,%.2f,%f,%f,%s,%d,%s\n",
				 time_str, power, total_particles, length_per_rank, nprocs,
				 box_length, RAM_GB, gen_time, dist_time, device_type, seed, mode_str);

	std::printf("%s,%d,%lld,%d,%d,%.1f,%.2f,%f,%f,%s,%d,%s\n",
				time_str, power, total_particles, length_per_rank, nprocs,
				box_length, RAM_GB, gen_time, dist_time, device_type, seed, mode_str);

	std::fclose(f);
}
