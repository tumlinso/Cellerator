cmake_minimum_required(VERSION 3.18)
project(CelleratorEffects LANGUAGES CXX)
include(CTest)
get_filename_component(CE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_subdirectory("${CE_ROOT}/src/math/effects" effects-owner)
add_executable(ce_is1_effects effects_test.cc)
target_link_libraries(ce_is1_effects PRIVATE Cellerator::effects)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Historical header formatting generates misleading-indentation diagnostics;
  # all other strict warnings remain enabled without altering preserved sources.
  target_compile_options(ce_is1_effects PRIVATE -Wall -Wextra -Werror -pedantic -Wno-misleading-indentation)
endif()
add_test(NAME ce_is1_effects COMMAND ce_is1_effects)
