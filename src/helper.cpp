#include "./helper.hpp"


MPI_Datatype MPI_particle;
int register_MPI_Particle(MPI_Datatype *MPI_Particle){
    int blocklengths[NPROPS_PARTICLE] = {1, 1, 3}; 
    MPI_Datatype array_types[NPROPS_PARTICLE] = {MPI_INT, MPI_LONG_LONG_INT, MPI_DOUBLE};
    t_particle dummy_particle[2];
    MPI_Aint address[NPROPS_PARTICLE + 1];
    MPI_Aint displacements[NPROPS_PARTICLE];
    MPI_Aint extent_add;
    int i, type_size;

    MPI_Get_address(&dummy_particle[0], &address[0]);
    MPI_Get_address(&dummy_particle[0].mpi_rank, &address[1]);
    MPI_Get_address(&dummy_particle[0].key, &address[2]);
    MPI_Get_address(&dummy_particle[0].coord, &address[3]);

    for (int i = 0; i < NPROPS_PARTICLE; i++){
        displacements[i] = address[i+1] - address[0]; 
    }

    MPI_Type_create_struct(NPROPS_PARTICLE, blocklengths, displacements, array_types, MPI_Particle);

    MPI_Get_address(&dummy_particle[1], &extent_add);
    extent_add = extent_add - address[0];
    MPI_Type_create_resized(*MPI_Particle, 0, extent_add, MPI_Particle);
    MPI_Type_size(*MPI_Particle, &type_size);

    MPI_Type_commit(MPI_Particle);
    return 0;
}

int allocate_particle(t_particle **particle_array, int count){
    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    (*particle_array) = (t_particle *)malloc(count*sizeof(t_particle));

    for (int i = 0; i < count; i++){
        (*particle_array)[i].mpi_rank = p_rank;
        (*particle_array)[i].key = 0;
        (*particle_array)[i].coord[0] = 0.0;
        (*particle_array)[i].coord[1] = 0.0;
        (*particle_array)[i].coord[2] = 0.0;
    }

    return 0;
}


int box_distribution(t_particle **particle_array, int count, double box_length){
    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);

    typedef r123::Philox4x32 RNG;
    RNG rng;
    RNG::ctr_type c={{}};
    RNG::ukey_type uk={{}};
    uk[0] = p_rank; // some user_supplied_seed
    RNG::key_type k=uk;
    RNG::ctr_type r;

    c[0] = 07072025;
    c[1] = 31106712;

    for (int i = 0; i < count; i++){
        c[0] += 1;
        c[1] += 1;
        r = rng(c, k);
        (*particle_array)[i].coord[0] = r123::u01<double>(r.v[0])*box_length;

        c[0] += 1;
        c[1] += 1;
        r = rng(c, k);
        (*particle_array)[i].coord[1] = r123::u01<double>(r.v[0])*box_length;

        c[0] += 1;
        c[1] += 1;
        r = rng(c, k);
        (*particle_array)[i].coord[2] = r123::u01<double>(r.v[0])*box_length;
    }
    return 0;
}

int torus_distribution(t_particle **particle_array, int count, double major_r, double minor_r, double box_length){
    int p_rank;
    double theta, phi, r;
    double center = box_length / 2.0;

    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);

    typedef r123::Philox4x32 RNG;
    RNG rng;
    RNG::ctr_type c={{}};
    RNG::ukey_type uk={{}};
    uk[0] = p_rank;
    RNG::key_type k=uk;
    RNG::ctr_type rnum;

    c[0] = 25082025;
    c[1] = 85712394;

    for (int i = 0; i < count; i++){
        c[0] += 1;
        c[1] += 1;
        rnum = rng(c, k);
        theta = 2.0 * M_PI * r123::u01<double>(rnum.v[0]);

        c[0] += 1;
        c[1] += 1;
        rnum = rng(c, k);
        phi = 2.0 * M_PI * r123::u01<double>(rnum.v[0]);

        c[0] += 1;
        c[1] += 1;
        rnum = rng(c, k);
        r = minor_r * sqrt(r123::u01<double>(rnum.v[0]));

        (*particle_array)[i].coord[0] = center + (major_r + r * cos(phi)) * cos(theta);
        (*particle_array)[i].coord[1] = center + (major_r + r * cos(phi)) * sin(theta);
        (*particle_array)[i].coord[2] = center + r * sin(phi);
    }

    return 0;
}

int parallel_write_to_file(t_particle *particle_array, int *count, char *filename){
    int access_mode;
    MPI_File fh;
    MPI_Status status;
    MPI_Offset disp, rank_offset, init_ind_ptr, fin_ind_ptr, init_shr_ptr, fin_shr_ptr;
    int p_rank, nprocs, particle_type_size;
    long long int tnp;
    MPI_Type_size(MPI_particle, &particle_type_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    access_mode = MPI_MODE_CREATE|MPI_MODE_RDWR;
    MPI_File_open(MPI_COMM_WORLD, filename, access_mode, MPI_INFO_NULL, &fh);

    if (p_rank == 0){
	tnp = 0;
        for (int i = 0; i < nprocs; i++){
            tnp += count[i];
        }
	MPI_File_write(fh, &tnp, 1, MPI_LONG_LONG_INT, &status);
    }

    rank_offset = 8;
    for (int i = 0; i < p_rank; i++){
        rank_offset += count[i]*particle_type_size;
    }
    MPI_File_seek(fh, rank_offset, MPI_SEEK_SET);

    MPI_File_write(fh, particle_array, count[p_rank], MPI_particle, &status);

    MPI_File_close(&fh);
    return 0;
}

int serial_write_to_file(t_particle *particle_array, int count, char *filename){
    std::fstream file;
    long long int ll_count;

    file.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    ll_count = count;
    file.write(reinterpret_cast<char *>(&ll_count), 8);
    
    for (int i = 0; i < count; i++){
        file.write(reinterpret_cast<char *>(&particle_array[i].mpi_rank), 4);
	file.write(reinterpret_cast<char *>(&particle_array[i].key), 8);
	file.write(reinterpret_cast<char *>(&particle_array[i].coord[0]), 8);
	file.write(reinterpret_cast<char *>(&particle_array[i].coord[1]), 8);
	file.write(reinterpret_cast<char *>(&particle_array[i].coord[2]), 8);
    }	

    file.close();
    return 0;
}

int serial_read_from_file(t_particle **particle_array, int *count, char *filename){
    std::fstream file;
    long long int ll_count;
    int temp_rank;
    long long int temp_key;
    double temp_coords;
    char bytes[256];

    file.open(filename, std::ios::in | std::ios::binary);

    file.read(bytes, 8);
    std::memcpy(&ll_count, bytes, sizeof(long long int));
    *count = ll_count;

    (*particle_array) = (t_particle *)malloc((*count)*sizeof(t_particle));

    for (int i = 0; i < *count; i++){
        file.read(bytes, 4);
        std::memcpy(&temp_rank, bytes, 4);
        (*particle_array)[i].mpi_rank = temp_rank;

        file.read(bytes, 8);
        std::memcpy(&temp_key, bytes, 8);
        (*particle_array)[i].key = temp_key;

        file.read(bytes, 8);
        std::memcpy(&temp_coords, bytes, 8);
        (*particle_array)[i].coord[0] = temp_coords;

            file.read(bytes, 8);
            std::memcpy(&temp_coords, bytes, 8);
        (*particle_array)[i].coord[1] = temp_coords;
            
            file.read(bytes, 8);
            std::memcpy(&temp_coords, bytes, 8);
        (*particle_array)[i].coord[2] = temp_coords;
    }	

    file.close();
    return 0;
}

void run_oct_tree_recursive(std::vector<t_particle*>& particles, int depth, long long key_prefix, double box_length, const std::array<double, 3>& origin) 
{
    
    //std::cout << "Call: count=" << particles.size() << " depth=" << depth << " prefix=" << key_prefix << "\n";

    if (particles.empty()) return;

    if (depth >= MAX_DEPTH) {
        for (auto* p : particles) {
            p->key = key_prefix;
        }
        //std::cout << "MAX_DEPTH\n";
        return;
    }

    double half = box_length / 2.0;
    std::array<double, 3> center = {
        origin[0] + half,
        origin[1] + half,
        origin[2] + half
    };

    std::vector<t_particle*> octants[8];

    for (auto* p : particles) {
        int oct = 0;
        if (p->coord[0] >= center[0]) oct |= 1;
        if (p->coord[1] >= center[1]) oct |= 2;
        if (p->coord[2] >= center[2]) oct |= 4;
        octants[oct].push_back(p);
    }

    for (int i = 0; i < 8; i++) {
        if (!octants[i].empty()) {
            long long new_key = (key_prefix << 3) | i;

            std::array<double, 3> new_origin = {
                origin[0] + (i & 1 ? half : 0),
                origin[1] + (i & 2 ? half : 0),
                origin[2] + (i & 4 ? half : 0)
            };

            run_oct_tree_recursive(octants[i], depth + 1, new_key, half, new_origin);
        }
    }
}



int generate_particles_keys(t_particle *particle_array, int count, double box_length) {
    std::vector<t_particle*> particles;
    particles.reserve(count);

    for (int i = 0; i < count; i++) {
        particles.push_back(&particle_array[i]);
    }

    std::array<double, 3> origin = {0.0, 0.0, 0.0};
    run_oct_tree_recursive(particles, 0, 0, box_length, origin);

    return 0;
}


bool compare_particles(const t_particle &a, const t_particle &b) {
    return a.key < b.key;
}

void radix_sort_particles(t_particle *particles, int n) {
    t_particle *tmp = (t_particle*)malloc(n * sizeof(t_particle));
    const int BITS = 64;       
    const int BASE = 256;      
    int count[BASE];
    
    for (int shift = 0; shift < BITS; shift += 8) {
        memset(count, 0, sizeof(count));
        
        for (int i = 0; i < n; i++)
            count[(particles[i].key >> shift) & 0xFF]++;
        
        for (int i = 1; i < BASE; i++)
            count[i] += count[i-1];
        
        for (int i = n-1; i >= 0; i--)
            tmp[--count[(particles[i].key >> shift) & 0xFF]] = particles[i];
        
        memcpy(particles, tmp, n * sizeof(t_particle));
    }
    free(tmp);
}


int distribute_particles(t_particle **particles, int *particle_vector_size, int nprocs){
    radix_sort_particles(*particles, *particle_vector_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int local_n = *particle_vector_size;

    int Rs = std::max(1, std::min(nprocs - 1, std::max(1, local_n / std::max(1, (int)std::floor((double)local_n / (nprocs))))));
    std::vector<long long> local_samples;
    local_samples.reserve(Rs);
    if (local_n > 0) {
        for (int i = 1; i <= Rs; ++i) {
            int idx = (int)((1.0 * i * local_n) / (Rs + 1));
            if (idx >= local_n) idx = local_n - 1;
            local_samples.push_back((*particles)[idx].key);
        }
    }

    std::vector<int> recvcounts, displs;
    int root = 0;
    int sendcount = (int)local_samples.size();
    if (rank == root) recvcounts.resize(nprocs);
    MPI_Gather(&sendcount, 1, MPI_INT,
               rank == root ? recvcounts.data() : nullptr, 1, MPI_INT,
               root, MPI_COMM_WORLD);

    int total_samples = 0;
    if (rank == root) {
        displs.resize(nprocs, 0);
        for (int i = 0; i < nprocs; ++i) {
            displs[i] = total_samples;
            total_samples += recvcounts[i];
        }
    }

    std::vector<long long> all_samples;
    if (rank == root) all_samples.resize(total_samples);

    MPI_Gatherv(local_samples.data(), sendcount, MPI_LONG_LONG_INT,
                rank == root ? all_samples.data() : nullptr,
                rank == root ? recvcounts.data() : nullptr,
                rank == root ? displs.data() : nullptr,
                MPI_LONG_LONG_INT,
                root, MPI_COMM_WORLD);

    std::vector<long long> splitters; 
    if (rank == root) {
        std::sort(all_samples.begin(), all_samples.end());
        splitters.resize(std::max(0, nprocs - 1));
        for (int i = 1; i < nprocs; ++i) {
            long long pos = (long long)std::llround((double)i * (all_samples.size()) / nprocs);
            if (pos <= 0) pos = 1;
            if (pos >= (long long)all_samples.size()) pos = (long long)all_samples.size() - 1;
            splitters[i - 1] = all_samples[(size_t)pos - 1];
        }
    } else {
        splitters.resize(std::max(0, nprocs - 1));
    }

    if (nprocs > 1) {
        MPI_Bcast(splitters.data(), (int)splitters.size(), MPI_LONG_LONG_INT, root, MPI_COMM_WORLD);
    }

    std::vector<int> sendcounts(nprocs, 0), sdispls(nprocs, 0);
    if (local_n > 0 && nprocs > 1) {
        std::vector<int> cuts; cuts.reserve(nprocs+1);
        cuts.push_back(0);
        for (int i = 0; i < (int)splitters.size(); ++i) {
            long long s = splitters[i];
            t_particle keyprobe; keyprobe.key = s;
            int pos = (int)(std::upper_bound(
                *particles, *particles + local_n,
                keyprobe,
                [](const t_particle& lhs, const t_particle& rhs){
                    return lhs.key < rhs.key;
                }
            ) - *particles);
            cuts.push_back(pos);
        }
        cuts.push_back(local_n);

        for (int b = 0; b < nprocs; ++b) {
            int begin = cuts[b];
            int end   = cuts[b+1];
            sendcounts[b] = std::max(0, end - begin);
        }
    } else {
        sendcounts[0] = local_n;
    }
    for (int i = 1; i < nprocs; ++i) sdispls[i] = sdispls[i-1] + sendcounts[i-1];

    std::vector<int> recvcounts2(nprocs, 0), rdispls(nprocs, 0);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts2.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 1; i < nprocs; ++i) rdispls[i] = rdispls[i-1] + recvcounts2[i-1];
    int recv_total = 0; for (int x: recvcounts2) recv_total += x;

    std::vector<t_particle> sendbuf(sdispls.back() + sendcounts.back());
    if (local_n > 0) {
        if (nprocs == 1) {
            std::memcpy(sendbuf.data(), *particles, local_n * sizeof(t_particle));
        } else {
            std::vector<int> cuts; cuts.reserve(nprocs+1);
            cuts.push_back(0);
            for (int i = 0; i < (int)splitters.size(); ++i) {
                long long s = splitters[i];
                t_particle keyprobe; keyprobe.key = s;
                int pos = (int)(std::upper_bound(
                    *particles, *particles + local_n,
                    keyprobe,
                    [](const t_particle& lhs, const t_particle& rhs){
                        return lhs.key < rhs.key;
                    }
                ) - *particles);
                cuts.push_back(pos);
            }
            cuts.push_back(local_n);

            for (int b = 0; b < nprocs; ++b) {
                int begin = cuts[b], end = cuts[b+1];
                int amt = end - begin;
                if (amt > 0) {
                    std::memcpy(sendbuf.data() + sdispls[b],
                                *particles + begin,
                                amt * sizeof(t_particle));
                }
            }
        }
    }

    std::vector<t_particle> recvbuf(recv_total);
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_particle,
                  recvbuf.data(), recvcounts2.data(), rdispls.data(), MPI_particle,
                  MPI_COMM_WORLD);

    for (auto &p : recvbuf) p.mpi_rank = rank;

    free(*particles);
    t_particle *newbuf = (t_particle*)malloc(recvbuf.size() * sizeof(t_particle));
    std::memcpy(newbuf, recvbuf.data(), recvbuf.size() * sizeof(t_particle));
    *particles = newbuf;
    *particle_vector_size = (int)recvbuf.size();

    printf("Rank %d, Number Particles: %d\n", rank, *particle_vector_size);
    return 0;
}



int redistribute_equal_counts(t_particle **particles, int *particle_vector_size, int nprocs) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_n = *particle_vector_size;

    radix_sort_particles(*particles, *particle_vector_size);    
    std::vector<int> counts(nprocs, 0);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    long long total = 0;
    for (int c : counts) total += c;
    int q = (int)(total / nprocs);
    int r = (int)(total % nprocs);

    std::vector<int> target(nprocs, q);
    for (int i = 0; i < r; ++i) target[i]++;

    struct Slot { int rank; int amt; };
    std::vector<Slot> surplus, deficit;
    surplus.reserve(nprocs);
    deficit.reserve(nprocs);
    for (int i = 0; i < nprocs; ++i) {
        int diff = counts[i] - target[i];
        if (diff > 0) surplus.push_back({i, diff});
        else if (diff < 0) deficit.push_back({i, -diff});
    }
    if (surplus.empty() && deficit.empty()) {
        for (int i = 0; i < local_n; ++i) (*particles)[i].mpi_rank = rank;
        return 0;
    }

    std::vector<int> S(nprocs * nprocs, 0);
    size_t si = 0, di = 0;
    while (si < surplus.size() && di < deficit.size()) {
        int s_rank = surplus[si].rank;
        int d_rank = deficit[di].rank;
        int amt = std::min(surplus[si].amt, deficit[di].amt);
        S[s_rank * nprocs + d_rank] += amt;
        surplus[si].amt -= amt;
        deficit[di].amt -= amt;
        if (surplus[si].amt == 0) ++si;
        if (deficit[di].amt == 0) ++di;
    }

    std::vector<int> sendcounts(nprocs, 0), recvcounts(nprocs, 0);
    for (int j = 0; j < nprocs; ++j) sendcounts[j] = S[rank * nprocs + j];
    for (int i = 0; i < nprocs; ++i) recvcounts[i] = S[i * nprocs + rank];

    std::vector<int> sdispls(nprocs, 0), rdispls(nprocs, 0);
    for (int i = 1; i < nprocs; ++i) {
        sdispls[i] = sdispls[i-1] + sendcounts[i-1];
        rdispls[i] = rdispls[i-1] + recvcounts[i-1];
    }
    int send_total = 0, recv_total = 0;
    for (int i = 0; i < nprocs; ++i) { send_total += sendcounts[i]; recv_total += recvcounts[i]; }

    int L = 0;                        
    for (int dst = 0; dst < rank; ++dst) L += sendcounts[dst];
    int R = 0;                         
    for (int dst = rank+1; dst < nprocs; ++dst) R += sendcounts[dst];

    int left_start  = 0;
    int keep_start  = left_start + L;
    int right_start = local_n - R;     
    int keep_end    = right_start - 1;
    int keep        = keep_end - keep_start + 1;

    if (keep < 0 || keep != (local_n - send_total)) {
        fprintf(stderr, "Rank %d: corte invalido (keep=%d, local=%d, send_total=%d)\n",
                rank, keep, local_n, send_total);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    std::vector<t_particle> sendbuf(send_total);

    int left_cursor = 0;
    for (int dst = 0; dst < rank; ++dst) {
        int amt = sendcounts[dst];
        if (amt > 0) {
            std::copy_n((*particles) + (left_start + left_cursor),
                        amt,
                        sendbuf.data() + sdispls[dst]);
            left_cursor += amt;
        }
    }

    int right_cursor = 0;
    for (int dst = rank+1; dst < nprocs; ++dst) {
        int amt = sendcounts[dst];
        if (amt > 0) {
            std::copy_n((*particles) + (right_start + right_cursor),
                        amt,
                        sendbuf.data() + sdispls[dst]);
            right_cursor += amt;
        }
    }

    std::vector<t_particle> recvbuf(recv_total);
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_particle,
                  recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_particle,
                  MPI_COMM_WORLD);

    for (auto &p : recvbuf) p.mpi_rank = rank;

    std::vector<t_particle> balanced;
    balanced.reserve(keep + recv_total);

    if (keep > 0) {
        balanced.insert(balanced.end(),
                        (*particles) + keep_start,
                        (*particles) + keep_start + keep);
    }
    balanced.insert(balanced.end(), recvbuf.begin(), recvbuf.end());

    free(*particles);
    t_particle *newbuf = (t_particle*)malloc(balanced.size() * sizeof(t_particle));
    std::memcpy(newbuf, balanced.data(), balanced.size() * sizeof(t_particle));
    *particles = newbuf;
    *particle_vector_size = (int)balanced.size();


    printf("Rank %d, Number Particles: %d\n", rank, *particle_vector_size);
    return 0;
}

void print_particles(t_particle *particle_array, int size, int rank) {        
    for (int i = 0; i < size; i++){ 
        printf("P_rank: %d, %d, %f, %f, %f, key: %lld, key_bin: ", 
               rank, particle_array[i].mpi_rank, 
               particle_array[i].coord[0], 
               particle_array[i].coord[1], 
               particle_array[i].coord[2], 
               particle_array[i].key);

        for(int b = 63; b >= 0; b--){ 
            printf("%lld", (particle_array[i].key >> b) & 1LL);
        }

        printf("\n");
    }
}