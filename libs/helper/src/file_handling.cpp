#include "file_handling.hpp"

MPI_Datatype MPI_particle;
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