cmake_minimum_required(VERSION 3.18)
project(cellerator_compiler_pass_tests LANGUAGES CXX)
enable_testing()
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

add_executable(ce_ccp1_g02_001
  freeze_the_pass_pipeline_stage_taxonomy_test.cc
  ${CELLERATOR_ROOT}/src/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy.cc)
target_include_directories(ce_ccp1_g02_001 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_g02_001 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_g02_001 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_g02_001 COMMAND ce_ccp1_g02_001)
