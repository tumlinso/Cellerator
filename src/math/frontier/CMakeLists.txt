get_filename_component(_ce_frontier_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(cellerator_frontier STATIC operators.cc)
add_library(Cellerator::frontier ALIAS cellerator_frontier)
target_include_directories(cellerator_frontier PUBLIC
  $<BUILD_INTERFACE:${_ce_frontier_root}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(cellerator_frontier PUBLIC Cellerator::effects Cellerator::native_numeric)
target_compile_features(cellerator_frontier PUBLIC cxx_std_20)
target_compile_options(cellerator_frontier PRIVATE -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)
unset(_ce_frontier_root)
