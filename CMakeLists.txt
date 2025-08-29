cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20) # Set CPP version
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # No compiler extensions

project(P_SFC VERSION 0.0.1 LANGUAGES CXX)

# Dependencies
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Boost REQUIRED)

add_subdirectory(third_party)
add_subdirectory(libs)
add_subdirectory(src)