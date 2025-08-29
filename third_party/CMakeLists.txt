# random123
add_library(random123 INTERFACE)
target_include_directories(random123 INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/random123/include)

add_library(third_party::random123 ALIAS random123)