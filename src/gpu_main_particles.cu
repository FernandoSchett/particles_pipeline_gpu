#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#include "helper.hpp"

#define DEFAULT_POWER 3

void parse_args(int argc, char **argv, int *power, dist_type_t *dist_type)
{
	*power = DEFAULT_POWER;
	*dist_type = DIST_UNKNOWN;

	if (argc > 1)
	{
		if (std::strcmp(argv[1], "box") == 0)
			*dist_type = DIST_BOX;
		else if (std::strcmp(argv[1], "torus") == 0)
			*dist_type = DIST_TORUS;
	}

	if (*dist_type == DIST_UNKNOWN)
	{
		*dist_type = DIST_BOX;
	}

	if (argc > 2)
	{
		*power = std::atoi(argv[2]);
	}
}

int main(int argc, char **argv)
{
	int rank = 0;
	int nprocs = 1;
	cudaGetDeviceCount(&nprocs);
	std::cout << "Using " << nprocs << " GPUs\n";
	char filename[128];

	int length_per_rank = 0;
	long long total_particles = 0;

	dist_type_t dist_type;
	int power = DEFAULT_POWER;
	double box_length = 0.0;
	int major_r = 0;
	int minor_r = 0;
	double RAM_GB = 0.0;

	const int block = 256;
	int sms = 0;

	parse_args(argc, argv, &power, &dist_type);

	std::vector<t_particle *> d_rank_array(nprocs, nullptr);
	std::vector<t_particle *> h_host_array(nprocs, nullptr);
	std::vector<cudaStream_t> gpu_streams(nprocs);
	std::vector<int> lens(nprocs);
	enable_p2p_all(nprocs);

	for (int dev = 0; dev < nprocs; dev++)
	{
		cudaSetDevice(dev);
		cudaStreamCreate(&gpu_streams[dev]);

		setup_particles_box_length(power, nprocs, dev, &length_per_rank, &box_length, &total_particles, &RAM_GB, &major_r, &minor_r);
		lens[dev] = length_per_rank;
		printf("Before distribution %d:  %d\n", dev, lens[dev]);

		// aloca particulas na gpu.
		cudaMallocAsync(&d_rank_array[dev], length_per_rank * sizeof(t_particle), gpu_streams[dev]);
		cudaMallocHost(&h_host_array[dev], length_per_rank * sizeof(t_particle));

		// set gpu kernel configs
		cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
		int maxBlocks = sms * 20;
		int grid = (length_per_rank + block - 1) / block;
		int seed = dev;
		if (grid > maxBlocks)
			grid = maxBlocks;

		switch (dist_type)
		{
		case DIST_BOX:
			box_distribution_kernel<<<grid, block, 0, gpu_streams[dev]>>>(d_rank_array[dev], length_per_rank, box_length, seed);
			break;
		case DIST_TORUS:
			torus_distribution_kernel<<<grid, block, 0, gpu_streams[dev]>>>(d_rank_array[dev], length_per_rank, major_r, minor_r, box_length, seed);
			break;
		}
	}

	gpu_barrier(nprocs, gpu_streams);

	// cria as keys na gpu.
	auto t0 = std::chrono::steady_clock::now();
	for (int dev = 0; dev < nprocs; dev++)
	{
		cudaSetDevice(dev);
		cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
		int maxBlocks = sms * 20;
		int grid = (length_per_rank + block - 1) / block;
		if (grid > maxBlocks)
			grid = maxBlocks;
		generate_keys_kernel<<<grid, block, 0, gpu_streams[dev]>>>(d_rank_array[dev], length_per_rank, box_length);
	}

	gpu_barrier(nprocs, gpu_streams);
	distribute_gpu_particles(d_rank_array, lens, gpu_streams);
	auto t1 = std::chrono::steady_clock::now();
	double dist_sec = std::chrono::duration<double>(t1 - t0).count();

	for (int dev = 0; dev < nprocs; ++dev)
	{
		cudaSetDevice(dev);

		if (h_host_array[dev])
		{
			cudaFreeHost(h_host_array[dev]);
			h_host_array[dev] = nullptr;
		}

		const size_t bytes = static_cast<size_t>(lens[dev]) * sizeof(t_particle);
		if (lens[dev] > 0)
		{
			cudaMallocHost(&h_host_array[dev], bytes);
			cudaMemcpyAsync(h_host_array[dev], d_rank_array[dev], bytes,
							cudaMemcpyDeviceToHost, gpu_streams[dev]);
		}
	}

	gpu_barrier(nprocs, gpu_streams);

	// write results and log it.
	if (power < 4)
	{
		sprintf(filename, "particle_file_gpu_n%d_total%lld.par", nprocs, total_particles);
		int rc = concat_and_serial_write(h_host_array.data(), lens.data(), nprocs, filename);

		// for (int dev = 0; dev < nprocs; dev++)
		//	print_particles(h_host_array[dev], lens[dev], dev);

		if (rc != 0)
		{
			std::cerr << "Error at writing file, rc=" << rc << "\n";
		}
	}

	log_results(rank, power, total_particles, length_per_rank, nprocs, box_length, RAM_GB, dist_sec, "gpu");

	// Cleans everything
	for (int dev = 0; dev < nprocs; dev++)
	{
		cudaSetDevice(dev);
		if (d_rank_array[dev])
			cudaFreeAsync(d_rank_array[dev], gpu_streams[dev]);
		if (h_host_array[dev])
			cudaFreeHost(h_host_array[dev]);
		cudaStreamDestroy(gpu_streams[dev]);
	}

	return 0;
}
