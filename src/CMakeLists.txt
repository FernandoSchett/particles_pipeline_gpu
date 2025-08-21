find_package(Boost 1.86.0 REQUIRED COMPONENTS sort)
add_library(helper STATIC helper.cpp helper.hpp)

target_include_directories(helper PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}" 
    "./random123/include/"
    ${Boost_INCLUDE_DIRS}
)

add_executable(p_sfc_exe main.cpp)
target_link_libraries(p_sfc_exe PRIVATE ${MPI_LIBRARIES} helper)
