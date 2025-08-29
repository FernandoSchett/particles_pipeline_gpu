# libs/helper/CMakeLists.txt


# Library Dependecies
find_package(Boost REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)


# Create TARGET (Helper Library)
add_library(helper
  src/helper.cpp
)

# Define librarie properties
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
)
