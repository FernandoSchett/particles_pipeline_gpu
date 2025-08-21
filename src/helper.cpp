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
    //print_particles(*particles, *particle_vector_size, 0);

    int *send_counts = (int*)calloc(nprocs, sizeof(int));
    int dest;   
    
    int bits_needed = 0;
    int tmp = nprocs - 1;
    while (tmp > 0) {
        bits_needed++;
        tmp >>= 1;
    }

    for (int i = 0; i < *particle_vector_size; i++){
        dest = ((*particles)[i].key >> (3 * MAX_DEPTH - bits_needed)) % nprocs;
        //printf("Dest: %d\n", dest);   
        send_counts[dest]++;
    }

    int *send_disp = (int*)malloc(nprocs * sizeof(int));
    send_disp[0] = 0;
    for (int i = 1; i < nprocs; i++)
        send_disp[i] = send_disp[i-1] + send_counts[i-1];

    int *recv_counts = (int*)malloc(nprocs * sizeof(int));
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = 0;
    int *recv_disp = (int*)malloc(nprocs * sizeof(int));
    recv_disp[0] = 0;
    for (int i = 0; i < nprocs; i++){
        if (i > 0) recv_disp[i] = recv_disp[i-1] + recv_counts[i-1];
        total_recv += recv_counts[i];
    }

    t_particle *recv_buffer = (t_particle*)malloc(total_recv * sizeof(t_particle));

    MPI_Alltoallv(*particles, send_counts, send_disp, MPI_particle,
                  recv_buffer, recv_counts, recv_disp, MPI_particle,
                  MPI_COMM_WORLD);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Rank %d, Number Particles: %d\n", rank, total_recv);
    for(int i = 0; i < total_recv; i++){
        recv_buffer[i].mpi_rank = rank;  
    }
                  
    free(*particles);
    *particles = recv_buffer;
    *particle_vector_size = total_recv;
    

    free(send_counts);
    free(send_disp);
    free(recv_counts);
    free(recv_disp);
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