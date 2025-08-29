# src/CMakeLists.txt

add_executable(p_sfc_exe cpu_main_particles.cpp)
target_link_libraries(p_sfc_exe PRIVATE P_SFC::helper)

add_executable(gpu_p_sfc_exe gpu_main_particles.cu)

target_link_libraries(gpu_p_sfc_exe PRIVATE P_SFC::helper CUDA::cudart)
set_target_properties(gpu_p_sfc_exe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(gpu_p_sfc_exe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-w>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-w>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
)
