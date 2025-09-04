// main_mpi_cuda_aware.cu
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

// Thrust
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/binary_search.h>

#include "helper.hpp"

#define DEFAULT_POWER 3

// -------------------------------------------------------------
// Utilidades
// -------------------------------------------------------------
static int get_local_rank()
{
    const char *s = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!s)
        s = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (!s)
        s = std::getenv("SLURM_LOCALID");
    return s ? std::atoi(s) : 0;
}

static void parse_args(int argc, char **argv, int *power, dist_type_t *dist_type)
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
        *dist_type = DIST_BOX;
    if (argc > 2)
        *power = std::atoi(argv[2]);
}

// Conta elementos <= limite (em vetor já ordenado por key) — roda no stream indicado
static long long count_leq_device(t_particle *d_ptr, int n, long long limit_key, cudaStream_t stream)
{
    if (n <= 0)
        return 0;
    t_particle probe;
    probe.key = limit_key;
    auto pol = thrust::cuda::par.on(stream);
    thrust::device_ptr<t_particle> first(d_ptr), last(d_ptr + n);
    auto it = thrust::upper_bound(pol, first, last, probe, key_less{});
    return static_cast<long long>(it - first);
}

// Escolhe (p-1) splitters aproximadamente uniformes a partir de todas as amostras globais
static std::vector<long long> choose_splitters_even(const std::vector<long long> &all_samples, int world_size)
{
    std::vector<long long> sorted = all_samples;
    std::sort(sorted.begin(), sorted.end());
    std::vector<long long> splitters;
    splitters.reserve(std::max(0, world_size - 1));
    if (world_size <= 1 || sorted.empty())
        return splitters;

    const size_t K = world_size - 1;
    for (size_t i = 1; i <= K; ++i)
    {
        double pos = (static_cast<double>(i) * (sorted.size())) / static_cast<double>(world_size);
        size_t idx = static_cast<size_t>(std::min<double>(std::max(0.0, pos), sorted.size() - 1));
        splitters.push_back(sorted[idx]);
    }
    return splitters;
}

// -------------------------------------------------------------
// Programa principal (1 rank = 1 GPU)
// -------------------------------------------------------------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Bind rank -> GPU local
    int ngpus_node = 0;
    cudaGetDeviceCount(&ngpus_node);
    int local_rank = get_local_rank();
    int dev = (ngpus_node > 0) ? (local_rank % ngpus_node) : 0;
    cudaSetDevice(dev);

    // (Opcional) fixar afinidade de CPU aqui, se desejado, para progress do MPI.

    // Argumentos de execução
    dist_type_t dist_type;
    int power = DEFAULT_POWER;
    parse_args(argc, argv, &power, &dist_type);

    // Parâmetros e buffers locais
    char filename[128];
    int length_local = 0;
    long long total_particles_hint = 0; // preenchido por setup_*
    double box_length = 0.0;
    int major_r = 0, minor_r = 0;
    double RAM_GB = 0.0;

    float gen_ms = 0.0f;
    double kernel_time_sec = 0.0;
    const int block = 256;
    int sms = 0;

    // stream e timing
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t kStart, kStop;
    cudaEventCreate(&kStart);
    cudaEventCreate(&kStop);

    // Cada rank configura seu tamanho local (a função já conhece 'world_size' e 'world_rank')
    setup_particles_box_length(power, world_size, world_rank,
                               &length_local, &box_length, &total_particles_hint,
                               &RAM_GB, &major_r, &minor_r);

    // Para logging global, podemos fazer um Allreduce se precisar validar total_particles
    long long localN = static_cast<long long>(length_local), totalN = 0;
    MPI_Allreduce(&localN, &totalN, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        std::cout << "MPI ranks (GPUs): " << world_size << "\n";
    }
    std::cout << "[Rank " << world_rank << " @ GPU " << dev << "] local length = " << length_local << "\n";

    // Alocação device/host (host só para escrita/depuração)
    t_particle *d_local = nullptr;
    cudaMallocAsync(&d_local, static_cast<size_t>(length_local) * sizeof(t_particle), stream);

    // (Opcional) host page-locked apenas quando necessário
    t_particle *h_local = nullptr;

    // Config do kernel
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int maxBlocks = sms * 20;
    int grid = (length_local + block - 1) / block;
    if (grid > maxBlocks)
        grid = maxBlocks;

    // Semente por rank
    int seed = world_rank;

    // Geração das partículas no device
    switch (dist_type)
    {
    case DIST_BOX:
        box_distribution_kernel<<<grid, block, 0, stream>>>(d_local, length_local, box_length, seed);
        break;
    case DIST_TORUS:
        torus_distribution_kernel<<<grid, block, 0, stream>>>(d_local, length_local, major_r, minor_r, box_length, seed);
        break;
    }

    // Gera keys
    cudaEventRecord(kStart, stream);
    generate_keys_kernel<<<grid, block, 0, stream>>>(d_local, length_local, box_length);

    // Sort local por key
    {
        auto pol = thrust::cuda::par.on(stream);
        thrust::device_ptr<t_particle> first(d_local), last(d_local + length_local);
        thrust::sort(pol, first, last, key_less{});
    }
    cudaEventRecord(kStop, stream);
    cudaEventSynchronize(kStop);
    cudaEventElapsedTime(&gen_ms, kStart, kStop);
    kernel_time_sec = std::max(kernel_time_sec, static_cast<double>(gen_ms) / 1000.0);

    // ---------------------------------------------------------
    // Distributed Sample Sort (CUDA-aware)
    // 1) Amostrar localmente
    // ---------------------------------------------------------
    const int samples_per_rank = std::max(1, 32); // ajustável
    std::vector<long long> local_samples;
    local_samples.reserve(samples_per_rank);

    if (length_local > 0)
    {
        // pegar pontos uniformes do array ordenado
        std::vector<int> idx(samples_per_rank);
        for (int i = 0; i < samples_per_rank; ++i)
        {
            long long pos = static_cast<long long>((i + 1) * (static_cast<double>(length_local) / (samples_per_rank + 1)));
            if (pos < 0)
                pos = 0;
            if (pos >= length_local)
                pos = length_local - 1;
            idx[i] = static_cast<int>(pos);
        }
        // copiar somente as keys (fazendo uma cópia pequena para host)
        // Para simplicidade: copiar os t_particle e extrair key no host
        local_samples.resize(samples_per_rank);
        // Buffer temporário
        std::vector<t_particle> tmp(samples_per_rank);
        for (int i = 0; i < samples_per_rank; ++i)
        {
            cudaMemcpyAsync(&tmp[i], d_local + idx[i], sizeof(t_particle), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
        for (int i = 0; i < samples_per_rank; ++i)
            local_samples[i] = static_cast<long long>(tmp[i].key);
    }

    // 2) Allgather de amostras
    int my_ns = static_cast<int>(local_samples.size());
    std::vector<int> all_ns(world_size, 0);
    MPI_Allgather(&my_ns, 1, MPI_INT, all_ns.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_samples = 0;
    for (int v : all_ns)
        total_samples += v;

    std::vector<int> recv_disp(world_size, 0);
    for (int i = 1; i < world_size; ++i)
        recv_disp[i] = recv_disp[i - 1] + all_ns[i - 1];

    std::vector<long long> all_samples(total_samples);
    MPI_Allgatherv(local_samples.data(), my_ns, MPI_LONG_LONG,
                   all_samples.data(), all_ns.data(), recv_disp.data(), MPI_LONG_LONG,
                   MPI_COMM_WORLD);

    // 3) Escolher splitters e broadcast
    std::vector<long long> splitters;
    if (world_rank == 0)
    {
        splitters = choose_splitters_even(all_samples, world_size);
    }
    else
    {
        splitters.resize(std::max(0, world_size - 1));
    }
    if (!splitters.empty())
    {
        MPI_Bcast(splitters.data(), static_cast<int>(splitters.size()), MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    // 4) Contar buckets locais (usando upper_bound no device)
    const int P = world_size;
    std::vector<int> sendcounts(P, 0);
    std::vector<long long> bounds(P + 1, 0); // índices de corte: [0, b1), [b1, b2), ... [bP-1, n)
    bounds[0] = 0;
    for (int i = 0; i < P - 1; ++i)
    {
        long long b = count_leq_device(d_local, length_local, splitters[i], stream);
        bounds[i + 1] = b;
    }
    bounds[P] = length_local;

    for (int i = 0; i < P; ++i)
    {
        long long cnt = bounds[i + 1] - bounds[i];
        sendcounts[i] = static_cast<int>(std::max<long long>(0, cnt));
    }

    // 5) Alltoall para descobrir recvcounts e alocar buffer de recepção
    std::vector<int> recvcounts(P, 0);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // displacements (em elementos)
    std::vector<int> sdisp(P, 0), rdisp(P, 0);
    for (int i = 1; i < P; ++i)
    {
        sdisp[i] = sdisp[i - 1] + sendcounts[i - 1];
        rdisp[i] = rdisp[i - 1] + recvcounts[i - 1];
    }
    const int new_local_n = rdisp[P - 1] + recvcounts[P - 1];

    t_particle *d_new = nullptr;
    if (new_local_n > 0)
    {
        cudaMallocAsync(&d_new, static_cast<size_t>(new_local_n) * sizeof(t_particle), stream);
    }

    // 6) Alltoallv CUDA-aware diretamente do d_local (buckets já contíguos)
    //    Envia fatias [bounds[i], bounds[i+1]) para o rank i.
    //    Usamos MPI_BYTE para evitar criar MPI_Datatype de t_particle.
    //    Displacements em BYTES:
    std::vector<int> sendcountsB(P), recvcountsB(P), sdispB(P), rdispB(P);
    for (int i = 0; i < P; ++i)
    {
        sendcountsB[i] = sendcounts[i] * static_cast<int>(sizeof(t_particle));
        recvcountsB[i] = recvcounts[i] * static_cast<int>(sizeof(t_particle));
        sdispB[i] = sdisp[i] * static_cast<int>(sizeof(t_particle));
        rdispB[i] = rdisp[i] * static_cast<int>(sizeof(t_particle));
    }

    // Observação: sdispB deve apontar para o início real das fatias.
    // Nosso sdisp[] presume concatenar buckets em ordem original.
    // Como os buckets já são contíguos em d_local mas NÃO estão concatenados,
    // precisamos que sdispB reflita 'bounds[i]' (offset real no d_local).
    // Ajuste:
    for (int i = 0; i < P; ++i)
    {
        sdispB[i] = static_cast<int>(bounds[i] * sizeof(t_particle)); // início real do bucket i em d_local
    }

    // Chamada coletiva (CUDA-aware): envia d_local e recebe em d_new
    MPI_Alltoallv(d_local, sendcountsB.data(), sdispB.data(), MPI_BYTE,
                  d_new, recvcountsB.data(), rdispB.data(), MPI_BYTE,
                  MPI_COMM_WORLD);

    // 7) Ordenação final local (para garantir ordenação após concatenação de segmentos)
    {
        auto pol = thrust::cuda::par.on(stream);
        thrust::device_ptr<t_particle> f(d_new), l(d_new + new_local_n);
        thrust::sort(pol, f, l, key_less{});
    }

    // Troca buffers
    cudaFreeAsync(d_local, stream);
    d_local = d_new;
    int local_len_after = new_local_n;

    cudaStreamSynchronize(stream);
    MPI_Barrier(MPI_COMM_WORLD);

    // ---------------------------------------------------------
    // Escrita (opcional: somente para tamanhos pequenos como no seu código)
    // Rank 0 coleta e escreve sequencialmente em um único arquivo
    // ---------------------------------------------------------
    if (power < 4)
    {
        std::snprintf(filename, sizeof(filename), "particle_file_gpu_n%d_total%lld.par", world_size, static_cast<long long>(totalN));

        // 1) Coletar tamanhos
        std::vector<int> all_sizes(world_size, 0);
        int my_size = local_len_after;
        MPI_Gather(&my_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            // Abre arquivo e recebe dados de cada rank (incluindo 0)
            FILE *fp = std::fopen(filename, "wb");
            if (!fp)
            {
                std::perror("fopen");
            }
            else
            {
                // Rank 0: copia do device para host e escreve
                if (local_len_after > 0)
                {
                    cudaMallocHost(&h_local, static_cast<size_t>(local_len_after) * sizeof(t_particle));
                    cudaMemcpyAsync(h_local, d_local,
                                    static_cast<size_t>(local_len_after) * sizeof(t_particle),
                                    cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    std::fwrite(h_local, sizeof(t_particle), local_len_after, fp);
                    cudaFreeHost(h_local);
                    h_local = nullptr;
                }
                // Recebe dos outros ranks
                for (int r = 1; r < world_size; ++r)
                {
                    int rcount = all_sizes[r];
                    if (rcount <= 0)
                        continue;
                    t_particle *h_tmp = nullptr;
                    cudaMallocHost(&h_tmp, static_cast<size_t>(rcount) * sizeof(t_particle));
                    // Recebe para host (o emissor enviará direto do device)
                    MPI_Recv(h_tmp, rcount * static_cast<int>(sizeof(t_particle)),
                             MPI_BYTE, r, 1234, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    std::fwrite(h_tmp, sizeof(t_particle), rcount, fp);
                    cudaFreeHost(h_tmp);
                }
                std::fclose(fp);
            }
        }
        else
        {
            // Ranks != 0: enviam diretamente do device
            if (local_len_after > 0)
            {
                MPI_Send(d_local, local_len_after * static_cast<int>(sizeof(t_particle)),
                         MPI_BYTE, 0, 1234, MPI_COMM_WORLD);
            }
            else
            {
                // Ainda envia zero bytes para manter matching simples (opcional)
                MPI_Send(d_local, 0, MPI_BYTE, 0, 1234, MPI_COMM_WORLD);
            }
        }
    }

    // ---------------------------------------------------------
    // Log (mantive a chamada; ajuste se quiser outro formato)
    // length_per_rank: podemos logar o tamanho final (balanceado)
    // ---------------------------------------------------------
    log_results(world_rank, power, totalN, local_len_after, world_size,
                box_length, RAM_GB, kernel_time_sec, "gpu_mpi_cuda_aware");

    // Limpeza
    if (d_local)
        cudaFreeAsync(d_local, stream);
    cudaEventDestroy(kStart);
    cudaEventDestroy(kStop);
    cudaStreamDestroy(stream);

    MPI_Finalize();
    return 0;
}
