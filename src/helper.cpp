#include "./helper.hpp"

MPI_Datatype MPI_particle;
int register_MPI_Particle(MPI_Datatype *MPI_Particle){
    int blocklengths[NPROPS_PARTICLE] = {1, 1, 3}; 
    MPI_Datatype array_types[NPROPS_PARTICLE] = {MPI_INT, MPI_INTEGER8, MPI_DOUBLE};
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

int torus_distribution(t_particle **particle_array, int count, double major_r, double minor_r){
    int p_rank, rep;
    double outer_dist, inner_dist, temp_x, temp_y, temp_z, ran_dist, inplane_dist;

    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);

    typedef r123::Philox4x32 RNG;
    RNG rng;
    RNG::ctr_type c={{}};
    RNG::ukey_type uk={{}};
    uk[0] = p_rank;
    RNG::key_type k=uk;
    RNG::ctr_type r;

    c[0] = 25082025;
    c[1] = 85712394;

    for (int i = 0; i < count; i++){
        rep = 1;
        while (rep == 1){
            outer_dist = major_r + minor_r;
            inner_dist = major_r - minor_r;
    
            c[0] += 1;
            c[1] += 1;
            r = rng(c, k);
            temp_x = (2.*r123::u01<double>(r.v[0]) - 1.) * outer_dist;
            temp_y = (2.*r123::u01<double>(r.v[1]) - 1.) * outer_dist;
            ran_dist = sqrt(temp_x*temp_x + temp_y*temp_y);
    
            if ((ran_dist <= outer_dist) && (ran_dist >= inner_dist)){
                temp_z = (2.*r123::u01<double>(r.v[2]) - 1.) * minor_r;
                inplane_dist = sqrt((ran_dist - major_r)*(ran_dist - major_r) + temp_z*temp_z);

                if (inplane_dist <= minor_r){
                    (*particle_array)[i].coord[0] = temp_x;
                    (*particle_array)[i].coord[1] = temp_y;
                    (*particle_array)[i].coord[2] = temp_z;
                    rep = 0;
                }
            } 
        }
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
