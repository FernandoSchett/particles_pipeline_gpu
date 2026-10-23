add_library(p_sfc_common STATIC
  src/utils.cpp
  src/file_handling.cpp
)

target_compile_features(p_sfc_common PUBLIC cxx_std_20)
target_include_directories(p_sfc_common
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
)
target_link_libraries(p_sfc_common
  PUBLIC
    MPI::MPI_CXX
)
set_target_properties(p_sfc_common PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if(BUILD_TESTING)
  add_executable(p_sfc_particle_domain_tests tests/particle_domain_tests.cpp)
  target_link_libraries(p_sfc_particle_domain_tests PRIVATE p_sfc_common)
  foreach(process_count IN ITEMS 1 2 4)
    add_test(NAME p_sfc_particle_domain_mpi_${process_count}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${process_count}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:p_sfc_particle_domain_tests>
        ${MPIEXEC_POSTFLAGS}
        ${PROJECT_SOURCE_DIR}/results/particles/mpi/particle_000000000001.mpi)
  endforeach()
endif()
