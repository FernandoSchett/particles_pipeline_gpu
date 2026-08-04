add_library(p_sfc_cpu STATIC
  src/hashed_octree.cpp
  src/particles_cpu.cpp
)

target_compile_features(p_sfc_cpu PUBLIC cxx_std_20)
target_include_directories(p_sfc_cpu
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
)
target_link_libraries(p_sfc_cpu
  PUBLIC
    p_sfc_common
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    third_party::random123
    Boost::headers
)
set_target_properties(p_sfc_cpu PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
