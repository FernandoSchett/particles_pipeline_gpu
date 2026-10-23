#ifndef P_SFC_PARTICLE_DOMAIN_HPP
#define P_SFC_PARTICLE_DOMAIN_HPP

#include <mpi.h>
#include "particle_types.hpp"

/**
 * @brief Fit a shared cubic Morton domain to distributed host coordinates.
 * @param particles Local host particles; coordinates remain unchanged.
 * @param count Local particle count, which may be zero.
 * @param cfg Configuration receiving the global origin and side length.
 * @param comm Communicator whose ranks all participate.
 * @return Zero on success, one for nonfinite coordinates or an unrepresentable span.
 */
int fit_particle_domain(const t_particle *particles, int64_t count,
                        ExecConfig &cfg, MPI_Comm comm);

#endif
