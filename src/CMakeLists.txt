# src/CMakeLists.txt

add_executable(p_sfc_exe cpu_main_particles.cpp)
target_link_libraries(p_sfc_exe PRIVATE P_SFC::helper)

add_executable(gpu_p_sfc_exe gpu_main_particles.cpp)
target_link_libraries(gpu_p_sfc_exe PRIVATE P_SFC::helper)
