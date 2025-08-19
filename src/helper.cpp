#include "./helper.hpp"

MPI_Datatype MPI_particle;
int register_MPI_Particle(MPI_Datatype *MPI_Particle){
    int blocklengths[NPROPS_PARTICLE] = {1, 1, 3}; 
    MPI_Datatype array_types[NPROPS_PARTICLE] = {MPI_INT, MPI_LONG_LONG, MPI_DOUBLE};
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

void run_oct_tree_recursive(t_particle **particles, int count, int depth, long long key_prefix) {
    if (depth >= MAX_DEPTH) {
        printf("MAX_DEPTH SUBDIVISIONS WERE NOT ENOUGH FOR THIS PARTICLES SET.\n");
        exit(0);
    }

    if (count == 0) return;

    if (count == 1) {
        long long final_key = key_prefix;
        int remaining_depth = MAX_DEPTH - depth;
        final_key <<= (3 * remaining_depth); 
        particles[0]->key = final_key;

        // gets msb
        particles[0]->quad = ( particles[0]->key >> (3 * MAX_DEPTH - 3)) & 0b111;
        return;
    }

    t_particle **octants[8];
    int oct_count[8] = {0};

    for (int i = 0; i < 8; i++) {
        octants[i] = (t_particle **)malloc(count * sizeof(t_particle *));
    }

    for (int i = 0; i < count; i++) {
        int oct = 0;
        if (particles[i]->coord[0] >= 0.5) oct |= 1;
        if (particles[i]->coord[1] >= 0.5) oct |= 2;
        if (particles[i]->coord[2] >= 0.5) oct |= 4;
        octants[oct][oct_count[oct]++] = particles[i];
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < oct_count[i]; j++) {
            if (i & 1) octants[i][j]->coord[0] = (octants[i][j]->coord[0] - 0.5) * 2;
            else       octants[i][j]->coord[0] *= 2;
            if (i & 2) octants[i][j]->coord[1] = (octants[i][j]->coord[1] - 0.5) * 2;
            else       octants[i][j]->coord[1] *= 2;
            if (i & 4) octants[i][j]->coord[2] = (octants[i][j]->coord[2] - 0.5) * 2;
            else       octants[i][j]->coord[2] *= 2;
        }
    }

    for (int i = 0; i < 8; i++) {
        if (oct_count[i] > 0) {
            long long new_key = (key_prefix << 3) | i;
            run_oct_tree_recursive(octants[i], oct_count[i], depth + 1, new_key);
        }
    }


    for (int i = 0; i < 8; i++) free(octants[i]);
}

int generate_particles_keys(t_particle **particle_array, int count, double box_length){
    long long key_prefix = 0;
    run_oct_tree_recursive(particle_array, count, 0, key_prefix);
    return 0;
}


bool compare_particles(const t_particle &a, const t_particle &b) {
    return a.quad > b.quad;
}

int distribute_particles(t_particle **particles, int *particle_vector_size, int nprocs){
    std::sort(*particles, *particles + *particle_vector_size, compare_particles);

    int *send_counts = (int*)calloc(nprocs, sizeof(int));
    for (int i = 0; i < *particle_vector_size; i++){
        int dest = (*particles)[i].quad;  
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

    free(*particles);
    *particles = recv_buffer;
    *particle_vector_size = total_recv;

    free(send_counts);
    free(send_disp);
    free(recv_counts);
    free(recv_disp);
}