cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -g")
project(P_SFC VERSION 0.0.1 LANGUAGES CXX)

find_package(MPI REQUIRED CXX)

include_directories(${MPI_CXX_INCLUDE_DIRS})

if (MPI_FOUND)
	message(STATUS "MPI_INC: [${MPI_CXX_INCLUDE_DIRS}]")
endif (MPI_FOUND)

add_subdirectory(src)
