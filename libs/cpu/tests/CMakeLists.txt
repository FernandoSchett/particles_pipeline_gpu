add_executable(p_sfc_cpu_tree_tests distributed_hashed_octree_tests.cpp)
target_link_libraries(p_sfc_cpu_tree_tests PRIVATE P_SFC::cpu)

foreach(process_count IN ITEMS 1 2 4)
  add_test(
    NAME p_sfc_cpu_tree_mpi_${process_count}
    COMMAND
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} ${process_count}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:p_sfc_cpu_tree_tests>
      ${MPIEXEC_POSTFLAGS}
  )
endforeach()
