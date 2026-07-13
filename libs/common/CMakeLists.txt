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
