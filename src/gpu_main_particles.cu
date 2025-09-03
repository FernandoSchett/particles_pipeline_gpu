#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <vector>
#include <cuda_runtime.h>

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

	float gen_ms = 0.0f;
	double kernel_time_sec = 0.0;

	int seed = 0;

	const int block = 256;
	int sms = 0;

	parse_args(argc, argv, &power, &dist_type);

	std::vector<t_particle *> d_rank_array(nprocs, nullptr);
	std::vector<t_particle *> h_host_array(nprocs, nullptr);
	std::vector<cudaEvent_t> kStart_v(nprocs), kStop_v(nprocs);
	std::vector<cudaStream_t> gpu_streams(nprocs);
	std::vector<int> lens(nprocs);

	for (int dev = 0; dev < nprocs; dev++)
	{
		cudaSetDevice(dev);
		cudaStreamCreate(&gpu_streams[dev]);

		setup_particles_box_length(power, nprocs, dev, &length_per_rank, &box_length, &total_particles, &RAM_GB, &major_r, &minor_r);
		lens[dev] = length_per_rank;

		// aloca particulas na gpu.
		cudaMallocAsync(&d_rank_array[dev], length_per_rank * sizeof(t_particle), gpu_streams[dev]);
		cudaMallocHost(&h_host_array[dev], length_per_rank * sizeof(t_particle));

		// set gpu kernel configs
		cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
		int maxBlocks = sms * 20;
		int grid = (length_per_rank + block - 1) / block;
		seed = dev;
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
		default:
			break;
		}

		// cudaEventCreate(&kStart_v[dev]);
		// cudaEventCreate(&kStop_v[dev]);
		// cudaEventRecord(kStart_v[dev], gpu_streams[dev]);

		// cria as keys na gpu.
		generate_keys_kernel<<<grid, block, 0, gpu_streams[dev]>>>(d_rank_array[dev], length_per_rank, box_length);

		// cudaEventRecord(kStop_v[dev], gpu_streams[dev]);
		// cudaMemcpyAsync(h_host_array[dev], d_rank_array[dev], static_cast<size_t>(length_per_rank) * sizeof(t_particle), cudaMemcpyDeviceToHost, gpu_streams[dev]);
	}

	gpu_barrier(nprocs, gpu_streams);
	distribute_gpu_particles(d_rank_array, lens, gpu_streams);

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

	// calculate time.
	for (int dev = 0; dev < nprocs; dev++)
	{
		cudaSetDevice(dev);
		cudaEventSynchronize(kStop_v[dev]);
		cudaEventElapsedTime(&gen_ms, kStart_v[dev], kStop_v[dev]);
		kernel_time_sec = std::max(kernel_time_sec, static_cast<double>(gen_ms) / 1000.0);
		cudaStreamSynchronize(gpu_streams[dev]);
	}

	// write results and log it.
	if (power < 4)
	{
		int rc = concat_and_serial_write(h_host_array.data(), lens.data(), nprocs, "particle_file");
		for (int dev = 0; dev < nprocs; dev++)
			print_particles(h_host_array[dev], lens[dev], dev);

		if (rc != 0)
		{
			std::cerr << "Error at writing file, rc=" << rc << "\n";
		}
	}

    sprintf(filename, "particle_file_gpu_n%d_total%lld", nprocs, total_particles);
	log_results(rank, power, total_particles, length_per_rank, nprocs, box_length, RAM_GB, kernel_time_sec, "gpu");

	// Cleans everything
	for (int dev = 0; dev < nprocs; dev++)
	{
		cudaSetDevice(dev);
		cudaEventDestroy(kStart_v[dev]);
		cudaEventDestroy(kStop_v[dev]);
		if (d_rank_array[dev])
			cudaFreeAsync(d_rank_array[dev], gpu_streams[dev]);
		if (h_host_array[dev])
			cudaFreeHost(h_host_array[dev]);
		cudaStreamDestroy(gpu_streams[dev]);
	}

	return 0;
}
