add_subdirectory(common)
add_subdirectory(cpu)

if(P_SFC_BUILD_GPU)
  add_subdirectory(gpu)
endif()

add_library(P_SFC::common ALIAS p_sfc_common)
add_library(P_SFC::cpu ALIAS p_sfc_cpu)

if(TARGET p_sfc_gpu)
  add_library(P_SFC::gpu ALIAS p_sfc_gpu)
endif()
