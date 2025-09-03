#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <vector>
#include <cuda_runtime.h>

#include "helper.hpp"

static long long count_leq_device(int dev, t_particle *d_ptr, int n, unsigned long long mid, cudaStream_t stream)
{
	if (n <= 0)
		return 0;
	t_particle probe;
	probe.key = (long long)mid;
	auto pol = thrust::cuda::par.on(stream);
	thrust::device_ptr<t_particle> first(d_ptr), last(d_ptr + n);
	auto it = thrust::upper_bound(pol, first, last, probe, key_less{});

	return static_cast<long long>(it - first);
}

static void compute_cuts_for_dev(int dev, t_particle *d_ptr, int n, const std::vector<unsigned long long> &splitters, std::vector<int> &cuts_out, cudaStream_t stream)
{
	cuts_out.clear();
	cuts_out.reserve((int)splitters.size() + 2);
	cuts_out.push_back(0);
	for (auto s : splitters)
	{
		t_particle probe;
		probe.key = (long long)s;
		auto pol = thrust::cuda::par.on(stream);
		thrust::device_ptr<t_particle> first(d_ptr), last(d_ptr + n);
		auto it = thrust::upper_bound(pol, first, last, probe, key_less{});
		cuts_out.push_back((int)(it - first));
	}
	cuts_out.push_back(n);
}

int distribute_gpu_particles(std::vector<t_particle *> &d_rank_array, std::vector<int> &lens, std::vector<cudaStream_t> &gpu_streams)
{
	const int nprocs = (int)d_rank_array.size();
	assert((int)lens.size() == nprocs && (int)gpu_streams.size() == nprocs);

	enable_p2p_all(nprocs);

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

	std::vector<unsigned long long> local_min(nprocs, std::numeric_limits<unsigned long long>::max());
	std::vector<unsigned long long> local_max(nprocs, 0ull);

	for (int dev = 0; dev < nprocs; ++dev)
	{
		cudaSetDevice(dev);
		int n = lens[dev];
		if (n <= 0)
			continue;
		t_particle first_h{}, last_h{};
		cudaMemcpyAsync(&first_h, d_rank_array[dev], sizeof(t_particle),
						cudaMemcpyDeviceToHost, gpu_streams[dev]);
		cudaMemcpyAsync(&last_h, d_rank_array[dev] + (n - 1), sizeof(t_particle),
						cudaMemcpyDeviceToHost, gpu_streams[dev]);
	}
	gpu_barrier(nprocs, gpu_streams);
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
	splitters.reserve(nprocs > 0 ? nprocs - 1 : 0);
	unsigned long long lo_base = gmin;
	for (int i = 1; i < nprocs; ++i)
	{
		const long long target = (N_global * i + nprocs - 1) / nprocs; // ceil div
		unsigned long long lo = lo_base, hi = gmax;
		while (lo < hi)
		{
			unsigned long long mid = lo + ((hi - lo) >> 1);
			long long c_global = 0;
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
	{
		for (int b = 0; b < nprocs; ++b)
		{
			int begin = cuts[src][b];
			int end = cuts[src][b + 1];
			sendcounts[src][b] = std::max(0, end - begin);
		}
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
	{
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
		if (n == 0)
			continue;
		int grid = (n + block - 1) / block;
		set_rank_kernel<<<grid, block, 0, gpu_streams[dev]>>>(d_rank_array[dev], n, dev);
	}
	gpu_barrier(nprocs, gpu_streams);

	return 0;
}

#define DEFAULT_POWER 3

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

void log_results(int rank,
				 int power,
				 long long total_particles,
				 int length_per_rank,
				 int nprocs,
				 double box_length,
				 double RAM_GB,
				 double execution_time)
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

	std::fprintf(f, "%s,%d,%lld,%d,%d,%.1f,%.2f,%f,gpu\n",
				 time_str, power, total_particles, length_per_rank, nprocs,
				 box_length, RAM_GB, execution_time);
	std::printf("%s,%d,%lld,%d,%d,%.1f,%.2f,%f,gpu\n",
				time_str, power, total_particles, length_per_rank, nprocs,
				box_length, RAM_GB, execution_time);

	std::fclose(f);
}

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

// Grava no formato portável (36 bytes/part), igual ao serial_write_to_file,
// mas aceitando N buffers (um por GPU) e escrevendo um ÚNICO arquivo:
// [ int64 total_particles ] + N blocos de partículas, cada registro:
//   int32 mpi_rank; int64 key; float64 x; float64 y; float64 z
int singleproc_write_to_file_portable_concat(t_particle **arrays,
											 const int *counts,
											 int nprocs,
											 const char *filename)
{
	FILE *fp = std::fopen(filename, "wb");
	if (!fp)
	{
		std::perror("open particles_file");
		return 1;
	}

	// Header: total de partículas (int64)
	long long int tnp = 0;
	for (int d = 0; d < nprocs; ++d)
	{
		if (counts[d] < 0)
		{
			std::fclose(fp);
			return 2;
		}
		tnp += (long long)counts[d];
	}
	if (std::fwrite(&tnp, sizeof(long long), 1, fp) != 1)
	{
		std::perror("write tnp");
		std::fclose(fp);
		return 3;
	}

	// Registros: exatamente como serial_write_to_file (36 bytes por partícula)
	for (int d = 0; d < nprocs; ++d)
	{
		const int n = counts[d];
		const t_particle *p = arrays[d];
		if (n == 0)
			continue;
		if (!p)
		{
			std::fclose(fp);
			return 4;
		}

		for (int i = 0; i < n; ++i)
		{
			// tamanhos fixos e ordem fixa
			int32_t mpi_rank = (int32_t)p[i].mpi_rank; // 4
			int64_t key = (int64_t)p[i].key;		   // 8
			double x = p[i].coord[0];				   // 8
			double y = p[i].coord[1];				   // 8
			double z = p[i].coord[2];				   // 8

			if (std::fwrite(&mpi_rank, sizeof(int32_t), 1, fp) != 1 ||
				std::fwrite(&key, sizeof(int64_t), 1, fp) != 1 ||
				std::fwrite(&x, sizeof(double), 1, fp) != 1 ||
				std::fwrite(&y, sizeof(double), 1, fp) != 1 ||
				std::fwrite(&z, sizeof(double), 1, fp) != 1)
			{
				std::perror("write particle");
				std::fclose(fp);
				return 5;
			}
		}
	}

	std::fclose(fp);
	return 0;
}

int concat_and_serial_write(t_particle **arrays,
							const int *counts,
							int nprocs,
							const char *filename)
{
	// 1) Soma total
	long long total_ll = 0;
	for (int d = 0; d < nprocs; ++d)
	{
		if (counts[d] < 0)
			return 1;
		total_ll += (long long)counts[d];
	}

	// serial_write_to_file espera 'int count'; cheque overflow (opcional)
	if (total_ll > std::numeric_limits<int>::max())
	{
		std::fprintf(stderr, "[E] total particles > INT_MAX (%lld)\n", total_ll);
		return 2;
	}
	const int total = (int)total_ll;

	// 2) Concatena
	std::vector<t_particle> tmp;
	tmp.reserve((size_t)total);

	for (int d = 0; d < nprocs; ++d)
	{
		const int n = counts[d];
		if (n <= 0)
			continue;
		if (!arrays[d])
			return 3;

		// copia exatamente sizeof(t_particle) p/ memória temporária
		// (isso é só memória; a escrita usará campo-a-campo pela serial_write_to_file)
		tmp.insert(tmp.end(), arrays[d], arrays[d] + n);
	}

	// 3) Chama sua função já validada (mesmo layout de 36 bytes)
	// ATENÇÃO: a assinatura pede char* (não const char*), então fazemos um cast.
	return serial_write_to_file(tmp.data(), total, const_cast<char *>(filename));
}

int main(int argc, char **argv)
{
	int rank = 0;
	int nprocs = 1;
	cudaGetDeviceCount(&nprocs);
	std::cout << "Using " << nprocs << " GPUs\n";

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
	// Barrier
	gpu_barrier(nprocs, gpu_streams);
	distribute_gpu_particles(d_rank_array, lens, gpu_streams); // atualiza d_rank_array[] e lens[]

	// *** (Re)alocar host e copiar de volta para host para poder imprimir ***
	for (int dev = 0; dev < nprocs; ++dev)
	{
		cudaSetDevice(dev);

		// se já existia, descarta o buffer antigo para evitar overflow
		if (h_host_array[dev])
		{
			cudaFreeHost(h_host_array[dev]);
			h_host_array[dev] = nullptr;
		}

		const size_t bytes = static_cast<size_t>(lens[dev]) * sizeof(t_particle);
		if (lens[dev] > 0)
		{
			// aloca host pinned com o tamanho *novo* pós-redistribuição
			cudaMallocHost(&h_host_array[dev], bytes);
			// copia D -> H no stream da GPU destino
			cudaMemcpyAsync(h_host_array[dev], d_rank_array[dev], bytes,
							cudaMemcpyDeviceToHost, gpu_streams[dev]);
		}
	}

	// garante que todas as cópias D->H terminaram
	gpu_barrier(nprocs, gpu_streams);

	// agora sim: imprime do host
	for (int dev = 0; dev < nprocs; ++dev)
	{
		if (lens[dev] == 0)
			continue;
		print_particles(h_host_array[dev], lens[dev], dev);
	}

	for (int dev = 0; dev < nprocs; dev++)
	{
		cudaSetDevice(dev);
		print_particles(h_host_array[dev], lens[dev], dev);

		cudaEventSynchronize(kStop_v[dev]);
		cudaEventElapsedTime(&gen_ms, kStart_v[dev], kStop_v[dev]);
		kernel_time_sec = std::max(kernel_time_sec, static_cast<double>(gen_ms) / 1000.0);
		cudaStreamSynchronize(gpu_streams[dev]);
	}

	gpu_barrier(nprocs, gpu_streams);

	int rc = concat_and_serial_write(h_host_array.data(), lens.data(), nprocs, "particle_file");
	if (rc != 0)
	{
		std::cerr << "Falha ao escrever particles_file via serial_write_to_file, rc=" << rc << "\n";
	}

	// distribui as keys na gpu.
	// distribute_particles(&rank_array, &length_per_rank, nprocs);

	// cada gpu manda pro host

	// host escreve em paralelo.
	// sprintf(filename, "particle_file_n%d_total%lld", nprocs, total_particles);
	// parallel_write_to_file(rank_array, length_vector, filename);

	log_results(rank, power, total_particles, length_per_rank, nprocs, box_length, RAM_GB, kernel_time_sec);

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
