#include "file_handling.hpp"

#include <cstdlib>

MPI_Datatype MPI_particle, MPI_particle_data, MPI_particle_results, MPI_particle_pepc;

int register_MPI_Particle(MPI_Datatype *MPI_Particle)
{
    int blocklengths[NPROPS_PARTICLE] = {1, 1, 3};
    MPI_Datatype array_types[NPROPS_PARTICLE] = {MPI_INT, MPI_LONG_LONG_INT, MPI_DOUBLE};
    t_particle dummy_particle[2];
    MPI_Aint address[NPROPS_PARTICLE + 1], displacements[NPROPS_PARTICLE], extent_add;

    MPI_Get_address(&dummy_particle[0], &address[0]);
    MPI_Get_address(&dummy_particle[0].mpi_rank, &address[1]);
    MPI_Get_address(&dummy_particle[0].key, &address[2]);
    MPI_Get_address(&dummy_particle[0].coord, &address[3]);

    for (int i = 0; i < NPROPS_PARTICLE; i++)
        displacements[i] = address[i + 1] - address[0];

    MPI_Datatype tmp;
    MPI_Type_create_struct(NPROPS_PARTICLE, blocklengths, displacements, array_types, &tmp);

    MPI_Get_address(&dummy_particle[1], &extent_add);
    extent_add -= address[0];

    MPI_Type_create_resized(tmp, 0, extent_add, MPI_Particle);
    MPI_Type_free(&tmp);
    MPI_Type_commit(MPI_Particle);
    return 0;
}

int concat_and_serial_write(t_particle **arrays, const int *counts, int nprocs, const char *filename)
{
    long long total_ll = 0;
    for (int d = 0; d < nprocs; ++d)
    {
        if (counts[d] < 0)
            return 1;
        total_ll += (long long)counts[d];
    }

    if (total_ll > std::numeric_limits<int>::max())
    {
        std::fprintf(stderr, "[E] total particles > INT_MAX (%lld)\n", total_ll);
        return 2;
    }
    const int total = (int)total_ll;

    std::vector<t_particle> tmp;
    tmp.reserve((size_t)total);

    for (int d = 0; d < nprocs; ++d)
    {
        const int n = counts[d];
        if (n <= 0)
            continue;
        if (!arrays[d])
            return 3;

        tmp.insert(tmp.end(), arrays[d], arrays[d] + n);
    }
    return serial_write_to_file(tmp.data(), total, const_cast<char *>(filename));
}

int parallel_write_to_file(t_particle *particle_array, int *count, char *filename)
{
    int access_mode;
    MPI_File fh;
    MPI_Status status;
    MPI_Offset disp, rank_offset, init_ind_ptr, fin_ind_ptr, init_shr_ptr, fin_shr_ptr;
    int p_rank, nprocs, particle_type_size;
    long long int tnp;
    MPI_Type_size(MPI_particle, &particle_type_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    access_mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    MPI_File_open(MPI_COMM_WORLD, filename, access_mode, MPI_INFO_NULL, &fh);

    if (p_rank == 0)
    {
        tnp = 0;
        for (int i = 0; i < nprocs; i++)
        {
            tnp += count[i];
        }
        MPI_File_write(fh, &tnp, 1, MPI_LONG_LONG_INT, &status);
    }

    rank_offset = 8;
    for (int i = 0; i < p_rank; i++)
    {
        rank_offset += count[i] * particle_type_size;
    }
    MPI_File_seek(fh, rank_offset, MPI_SEEK_SET);

    MPI_File_write(fh, particle_array, count[p_rank], MPI_particle, &status);

    MPI_File_close(&fh);
    return 0;
}

int register_MPI_particle_data(MPI_Datatype *MPI_Particle_data)
{
    int blocklengths[NPROPS_PEPC_PARTICLE_DATA] = {1, 3, 1, 3, 3, 3, 1, 1, 1};
    MPI_Datatype array_types[NPROPS_PEPC_PARTICLE_DATA] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, \
                                                           MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, \
                                                           MPI_INT, MPI_INT, MPI_DOUBLE};
    t_pepc_particle_data dummy_particle_data[2];
    MPI_Aint address[NPROPS_PEPC_PARTICLE_DATA + 1], displacements[NPROPS_PEPC_PARTICLE_DATA], extent_add;

    MPI_Get_address(&dummy_particle_data[0], &address[0]);
    MPI_Get_address(&dummy_particle_data[0].q, &address[1]);
    MPI_Get_address(&dummy_particle_data[0].v, &address[2]);
    MPI_Get_address(&dummy_particle_data[0].m, &address[3]);
    MPI_Get_address(&dummy_particle_data[0].b, &address[4]);
    MPI_Get_address(&dummy_particle_data[0].f_e, &address[5]);
    MPI_Get_address(&dummy_particle_data[0].f_b, &address[6]);
    MPI_Get_address(&dummy_particle_data[0].species, &address[7]);
    MPI_Get_address(&dummy_particle_data[0].mp_int1, &address[8]);
    MPI_Get_address(&dummy_particle_data[0].age, &address[9]);

    for (int i = 0; i < NPROPS_PEPC_PARTICLE_DATA; i++)
        displacements[i] = address[i + 1] - address[0];

    MPI_Datatype tmp;
    MPI_Type_create_struct(NPROPS_PEPC_PARTICLE_DATA, blocklengths, displacements, array_types, &tmp);

    MPI_Get_address(&dummy_particle_data[1], &extent_add);
    extent_add -= address[0];

    MPI_Type_create_resized(tmp, 0, extent_add, MPI_Particle_data);
    MPI_Type_free(&tmp);
    MPI_Type_commit(MPI_Particle_data);
    return 0;
}

int register_MPI_particle_results(MPI_Datatype *MPI_Particle_results)
{
    int blocklengths[NPROPS_PEPC_PARTICLE_RESULTS] = {3, 1};
    MPI_Datatype array_types[NPROPS_PEPC_PARTICLE_RESULTS] = {MPI_DOUBLE, MPI_DOUBLE};

    t_pepc_particle_results dummy_particle_results[2];
    MPI_Aint address[NPROPS_PEPC_PARTICLE_RESULTS + 1], displacements[NPROPS_PEPC_PARTICLE_RESULTS], extent_add;

    MPI_Get_address(&dummy_particle_results[0], &address[0]);
    MPI_Get_address(&dummy_particle_results[0].e, &address[1]);
    MPI_Get_address(&dummy_particle_results[0].pot, &address[2]);

    for (int i = 0; i < NPROPS_PEPC_PARTICLE_RESULTS; i++)
        displacements[i] = address[i + 1] - address[0];

    MPI_Datatype tmp;
    MPI_Type_create_struct(NPROPS_PEPC_PARTICLE_RESULTS, blocklengths, displacements, array_types, &tmp);

    MPI_Get_address(&dummy_particle_results[1], &extent_add);
    extent_add -= address[0];

    MPI_Type_create_resized(tmp, 0, extent_add, MPI_Particle_results);
    MPI_Type_free(&tmp);
    MPI_Type_commit(MPI_Particle_results);
    return 0;
}

int register_MPI_Particle_pepc(MPI_Datatype *MPI_Particle_pepc)
{
    register_MPI_particle_results(&MPI_particle_results);
    register_MPI_particle_data(&MPI_particle_data);
    int blocklengths[NPROPS_PEPC_PARTICLE] = {3, 1, 1, 1, 1, 1, 1};
    MPI_Datatype array_types[NPROPS_PEPC_PARTICLE] = {MPI_DOUBLE, MPI_DOUBLE, MPI_LONG_LONG_INT, \
                                                      MPI_LONG_LONG_INT, MPI_LONG_LONG_INT, \
                                                      MPI_particle_data, MPI_particle_results};
    t_pepc_particle dummy_particle_pepc[2];
    MPI_Aint address[NPROPS_PEPC_PARTICLE + 1], displacements[NPROPS_PEPC_PARTICLE], extent_add;

    MPI_Get_address(&dummy_particle_pepc[0], &address[0]);
    MPI_Get_address(&dummy_particle_pepc[0].x, &address[1]);
    MPI_Get_address(&dummy_particle_pepc[0].work, &address[2]);
    MPI_Get_address(&dummy_particle_pepc[0].key, &address[3]);
    MPI_Get_address(&dummy_particle_pepc[0].node_leaf, &address[4]);
    MPI_Get_address(&dummy_particle_pepc[0].label, &address[5]);
    MPI_Get_address(&dummy_particle_pepc[0].data, &address[6]);
    MPI_Get_address(&dummy_particle_pepc[0].results, &address[7]);

    for (int i = 0; i < NPROPS_PEPC_PARTICLE; i++)
        displacements[i] = address[i + 1] - address[0];

    MPI_Datatype tmp;
    MPI_Type_create_struct(NPROPS_PEPC_PARTICLE, blocklengths, displacements, array_types, &tmp);

    MPI_Get_address(&dummy_particle_pepc[1], &extent_add);
    extent_add -= address[0];

    MPI_Type_create_resized(tmp, 0, extent_add, MPI_Particle_pepc);
    MPI_Type_free(&tmp);
    MPI_Type_commit(MPI_Particle_pepc);
    return 0;
}

int parallel_read_pepc_particles(MPI_Comm comm, t_pepc_particle **pepc_array, \
                                 char *filename, int64_t *n_total, int64_t *n_local)
{
    MPI_File fh;
    MPI_Status status;
    int comm_size, comm_rank;
    int dummy;
    int64_t particles_remainder;
    int64_t setView_offset = 0;
    int64_t total_p, local_p;
    register_MPI_Particle_pepc(&MPI_particle_pepc);

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Can't open file, check if it exists.\n" << std::endl;
        return 1;
    }

    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
    MPI_File_read(fh, &total_p, 1, MPI_INT64_T, &status);
    MPI_File_read(fh, &dummy, 1, MPI_INT, &status);

    local_p = total_p/comm_size;
    particles_remainder = total_p%comm_size;
    if (comm_rank < particles_remainder) local_p += 1;

    *pepc_array = (t_pepc_particle *)malloc(local_p*sizeof(t_pepc_particle));

    setView_offset = 128;
    MPI_File_set_view(fh, setView_offset, MPI_particle_pepc, MPI_particle_pepc, "native", MPI_INFO_NULL);
    MPI_File_read_ordered(fh, (*pepc_array), local_p, MPI_particle_pepc, &status);
    MPI_File_close(&fh);

    *n_total = total_p;
    *n_local = local_p;
    return 0;
}

void pepc_to_simple_particle_array(int rank, t_pepc_particle **pepc_array, t_particle **particle_array, int64_t n_local)
{
// NOTE: intended to be called by all MPI rank.
    *particle_array = (t_particle *)malloc(n_local*sizeof(t_particle));

    for (int64_t i = 0; i < n_local; i++)
    {
       (*particle_array)[i].mpi_rank = rank;
       (*particle_array)[i].key = (*pepc_array)[i].key;
       (*particle_array)[i].coord[0] = (*pepc_array)[i].x[0];
       (*particle_array)[i].coord[1] = (*pepc_array)[i].x[1];
       (*particle_array)[i].coord[2] = (*pepc_array)[i].x[2];
    }
}

int serial_write_to_file(t_particle *particle_array, int count, char *filename)
{
    std::fstream file;
    long long int ll_count;

    file.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    ll_count = count;
    file.write(reinterpret_cast<char *>(&ll_count), 8);

    for (int i = 0; i < count; i++)
    {
        file.write(reinterpret_cast<char *>(&particle_array[i].mpi_rank), 4);
        file.write(reinterpret_cast<char *>(&particle_array[i].key), 8);
        file.write(reinterpret_cast<char *>(&particle_array[i].coord[0]), 8);
        file.write(reinterpret_cast<char *>(&particle_array[i].coord[1]), 8);
        file.write(reinterpret_cast<char *>(&particle_array[i].coord[2]), 8);
    }

    file.close();
    return 0;
}

int serial_read_from_file(t_particle **particle_array, int *count, char *filename)
{
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

    (*particle_array) = (t_particle *)malloc((*count) * sizeof(t_particle));

    for (int i = 0; i < *count; i++)
    {
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
