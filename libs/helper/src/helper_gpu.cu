#include "./helper.hpp"


__global__ void box_distribution_kernel(t_particle* particles, int N, double L, unsigned long long seed)
{
    using RNG = r123::Philox4x32;
    RNG::key_type key = {{ (uint32_t)seed, (uint32_t)(seed >> 32) }};

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        RNG::ctr_type ctr = {{ (uint32_t)i, 0u, 0u, 0u }};
        RNG::ctr_type r   = RNG()(ctr, key);

        particles[i].coord[0] = r123::u01<double>(r.v[0]) * L;
        particles[i].coord[1] = r123::u01<double>(r.v[1]) * L;
        particles[i].coord[2] = r123::u01<double>(r.v[2]) * L;
    }
}

__global__ void torus_distribution_kernel(t_particle* particles, int N, double major_r, double minor_r, double box_length, unsigned long long seed)
{
    using RNG = r123::Philox4x32;
    RNG::key_type key = {{ (uint32_t)seed, (uint32_t)(seed >> 32) }};
    const double TWO_PI = 6.283185307179586476925286766559;
    const double center = box_length * 0.5;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        RNG::ctr_type ctr = {{ (uint32_t)i, 0u, 0u, 0u }};
        RNG::ctr_type rnum = RNG()(ctr, key);

        double u0 = r123::u01<double>(rnum.v[0]);
        double u1 = r123::u01<double>(rnum.v[1]);
        double u2 = r123::u01<double>(rnum.v[2]);

        double theta = TWO_PI * u0;
        double phi   = TWO_PI * u1;
        double r     = minor_r * sqrt(u2);

        double cphi = cos(phi);
        double sphi = sin(phi);
        double cth  = cos(theta);
        double sth  = sin(theta);

        double Rplus = major_r + r * cphi;

        particles[i].coord[0] = center + Rplus * cth;
        particles[i].coord[1] = center + Rplus * sth;
        particles[i].coord[2] = center + r * sphi;
    }
}

__global__ void generate_keys_kernel(t_particle* particles, int N, double box_length)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        double x = particles[i].coord[0];
        double y = particles[i].coord[1];
        double z = particles[i].coord[2];

        double ox = 0.0, oy = 0.0, oz = 0.0;
        double len = box_length;

        unsigned long long key = 0ull;

        #pragma unroll 1
        for (int d = 0; d < MAX_DEPTH; ++d) {
            len *= 0.5;                  
            int oct = 0;

            double cx = ox + len;        
            double cy = oy + len;
            double cz = oz + len;

            if (x >= cx) { oct |= 1; ox += len; }
            if (y >= cy) { oct |= 2; oy += len; }
            if (z >= cz) { oct |= 4; oz += len; }

            key = (key << 3) | (unsigned long long)oct;
        }

        particles[i].key = (long long)key;
    }
}

__global__ void set_rank_kernel(t_particle* p, int n, int rank_id){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i].mpi_rank = rank_id;
}

void gpu_barrier(int nprocs, const std::vector<cudaStream_t>& streams){
    for(int d = 0; d < nprocs; ++d){  
        cudaSetDevice(d); 
        cudaStreamSynchronize(streams[d]); 
    }
}

void enable_p2p_all(int ndev){
    for (int i = 0;i < ndev; ++i){
        cudaSetDevice(i);   
        for (int j = 0; j < ndev; ++j){
            if (i == j) continue;
            
            int can = 0; 
            cudaDeviceCanAccessPeer(&can, i, j);
            
            if (can){
                auto err = cudaDeviceEnablePeerAccess(j,0);
                if (err!=cudaSuccess && err!=cudaErrorPeerAccessAlreadyEnabled){
                    cudaGetLastError(); 
                }
            }
        }
    }
}
