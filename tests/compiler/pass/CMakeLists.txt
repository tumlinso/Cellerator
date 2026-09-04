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

add_executable(ce_ccp1_g02_002
  implement_pass_manager_and_analysis_invalidation_test.cc
  ${CELLERATOR_ROOT}/src/compiler/pass/implement_pass_manager_and_analysis_invalidation.cc
  ${CELLERATOR_ROOT}/src/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy.cc)
target_include_directories(ce_ccp1_g02_002 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_g02_002 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_g02_002 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_g02_002 COMMAND ce_ccp1_g02_002)

add_executable(ce_ccp1_g02_003
  implement_pipeline_configuration_syntax_and_apis_test.cc
  ${CELLERATOR_ROOT}/src/compiler/pass/implement_pipeline_configuration_syntax_and_apis.cc
  ${CELLERATOR_ROOT}/src/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy.cc)
target_include_directories(ce_ccp1_g02_003 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_g02_003 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_g02_003 PRIVATE -Wall -Wextra -Werror)
add_test(NAME ce_ccp1_g02_003 COMMAND ce_ccp1_g02_003)

add_executable(ce_ccp1_g02_004
  implement_custom_semantic_pass_api_test.cc
  ${CELLERATOR_ROOT}/src/compiler/pass/implement_custom_semantic_pass_api.cc)
target_include_directories(ce_ccp1_g02_004 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_g02_004 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_g02_004 PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(ce_ccp1_g02_004 PRIVATE
  CELLERATOR_TEST_INCLUDE_ROOT="${CELLERATOR_ROOT}/include")
target_link_libraries(ce_ccp1_g02_004 PRIVATE dl)
add_test(NAME ce_ccp1_g02_004 COMMAND ce_ccp1_g02_004)
