# src/CMakeLists.txt

add_executable(p_sfc_exe main.cpp)
target_link_libraries(p_sfc_exe PRIVATE P_SFC::helper)
# (Opcional: o helper já publica cxx_std_20, então isto é redundante)
# target_compile_features(p_sfc_exe PRIVATE cxx_std_20)
