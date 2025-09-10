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

void setup_particles_box_length(int power, int nprocs, int rank, int *length_per_rank, double *box_length, long long *total_particles, double *RAM_GB, int *major_r, int *minor_r)
{
    const long long slice = static_cast<long long>(std::pow(10, power) / nprocs);
    *total_particles = ((1 + nprocs) * nprocs / 2) * slice;
    *RAM_GB = (*total_particles * 40.0) / 1e9; // sizeof(t_particle) is 36, but it can be considered as 40.
    *box_length = std::pow(10, power);
    *length_per_rank = (rank + 1) * slice;
    *major_r = 4 * std::pow(10, power - 1);
    *minor_r = 2 * std::pow(10, power - 1);

    if (rank == nprocs - 1)
    {
        *length_per_rank += *total_particles % nprocs;
    }

    if (rank == 0)
    {
        std::printf("%lld particles distributed like (rank_number*%lld) between %d processes in a %.1f sized box using %.4f GBs.\n",
                    *total_particles, slice, nprocs, *box_length, *RAM_GB);
    }
}



void log_results(int rank, int power, long long total_particles, int length_per_rank, int nprocs, double box_length, double RAM_GB, double execution_time, const char *device_type)
{
	time_t rawtime;
	std::tm *timeinfo;
	char time_str[64];

	std::time(&rawtime);
	timeinfo = std::localtime(&rawtime);
	std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", timeinfo);

	struct stat buffer;
	const char *results_path = "../../results.csv";
	const int file_exists = (stat(results_path, &buffer) == 0);

	FILE *f = std::fopen(results_path, "a");
	if (!file_exists)
	{
		std::fprintf(f, "datetime,power,total_particles,length_per_rank,num_procs,box_length,RAM_GB,execution_time,device\n");
	}

	std::fprintf(f, "%s,%d,%lld,%d,%d,%.1f,%.2f,%f,%s\n",
				 time_str, power, total_particles, length_per_rank, nprocs,
				 box_length, RAM_GB, execution_time, device_type);
	std::printf("%s,%d,%lld,%d,%d,%.1f,%.2f,%f,%s\n",
				time_str, power, total_particles, length_per_rank, nprocs,
				box_length, RAM_GB, execution_time, device_type);

	std::fclose(f);
}

