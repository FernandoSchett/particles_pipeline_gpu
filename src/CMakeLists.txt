# src/CMakeLists.txt

# Multi CPU
add_executable(p_sfc_exe cpu_main_particles.cpp)
target_link_libraries(p_sfc_exe PRIVATE P_SFC::cpu)

# Multi GPU, Multi Node
if(P_SFC_BUILD_GPU)
  add_executable(gpu_mpi_p_sfc_exe gpu_mpi_main_particles.cu)
  target_link_libraries(gpu_mpi_p_sfc_exe PRIVATE P_SFC::gpu)
  set_target_properties(gpu_mpi_p_sfc_exe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(gpu_mpi_p_sfc_exe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
  )
endif()

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME p_sfc_python_tree_oracles
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
      -s ${PROJECT_SOURCE_DIR}/py_apps/tests -v)
  foreach(process_count IN ITEMS 1 2 4)
    add_test(NAME p_sfc_cpu_tree_reference_mpi_${process_count}
      COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/py_apps/run_tree_correctness.py
        --device cpu --executable $<TARGET_FILE:p_sfc_exe>
        --ranks ${process_count} --launcher ${MPIEXEC_EXECUTABLE}
        --rank-flag=${MPIEXEC_NUMPROC_FLAG}
        --input-pepc ${PROJECT_SOURCE_DIR}/results/particles/mpi/particle_000000000001.mpi)
    set_tests_properties(p_sfc_cpu_tree_reference_mpi_${process_count}
      PROPERTIES TIMEOUT 240 PROCESSORS ${process_count} LABELS "correctness;cpu;mpi")
  endforeach()
  option(P_SFC_TEST_GPU "Run correctness tests requiring CUDA GPUs and CUDA-aware MPI" OFF)
  if(P_SFC_TEST_GPU)
    if(NOT P_SFC_BUILD_GPU)
      message(FATAL_ERROR "P_SFC_TEST_GPU requires P_SFC_BUILD_GPU")
    endif()
    foreach(process_count IN ITEMS 1 2 4)
      add_test(NAME p_sfc_gpu_tree_reference_mpi_${process_count}
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/py_apps/run_tree_correctness.py
          --device gpu --executable $<TARGET_FILE:gpu_mpi_p_sfc_exe>
          --ranks ${process_count} --launcher ${MPIEXEC_EXECUTABLE}
          --rank-flag=${MPIEXEC_NUMPROC_FLAG}
          --input-pepc ${PROJECT_SOURCE_DIR}/results/particles/mpi/particle_000000000001.mpi)
      set_tests_properties(p_sfc_gpu_tree_reference_mpi_${process_count}
        PROPERTIES TIMEOUT 240 PROCESSORS ${process_count} LABELS "correctness;gpu;mpi" RUN_SERIAL TRUE)
    endforeach()
  endif()
endif()
