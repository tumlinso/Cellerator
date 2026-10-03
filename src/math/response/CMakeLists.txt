get_filename_component(_ce_response_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(cellerator_response STATIC requested.cc)
add_library(Cellerator::response ALIAS cellerator_response)
target_include_directories(cellerator_response PUBLIC
  $<BUILD_INTERFACE:${_ce_response_root}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(cellerator_response PUBLIC Cellerator::prepared_host)
target_compile_features(cellerator_response PUBLIC cxx_std_20)
target_compile_options(cellerator_response PRIVATE -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)
unset(_ce_response_root)
