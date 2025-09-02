# libs/helper/CMakeLists.txt


# Library Dependecies
find_package(Boost REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

# Create TARGET (Helper Library)
add_library(helper
  src/helper.cpp
  src/helper_gpu.cu 
)

# Define librarie properties
set_target_properties(helper PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
)

target_compile_features(helper PUBLIC cxx_std_20)

target_include_directories(helper
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(helper
  PUBLIC
    Boost::headers
    MPI::MPI_CXX
    third_party::random123
    CUDA::cudart  
)


