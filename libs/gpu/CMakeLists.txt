add_library(p_sfc_gpu STATIC
  src/distributed_octree.cu
  src/particles_gpu.cu
)

target_compile_features(p_sfc_gpu PUBLIC cxx_std_20)
target_include_directories(p_sfc_gpu
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
    ${MPI_CXX_COMPILER_INCLUDE_DIRS}
    ${MPI_CXX_ADDITIONAL_INCLUDE_DIRS}
)
target_link_libraries(p_sfc_gpu
  PUBLIC
    p_sfc_common
    MPI::MPI_CXX
    CUDA::cudart
    third_party::random123
)
set_target_properties(p_sfc_gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
)
