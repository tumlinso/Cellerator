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

add_executable(ce_ccp1_g02_005
  implement_custom_planning_pass_api_test.cc
  ${CELLERATOR_ROOT}/src/compiler/pass/implement_custom_planning_pass_api.cc)
target_include_directories(ce_ccp1_g02_005 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_g02_005 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_g02_005 PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(ce_ccp1_g02_005 PRIVATE
  CELLERATOR_TEST_INCLUDE_ROOT="${CELLERATOR_ROOT}/include")
target_link_libraries(ce_ccp1_g02_005 PRIVATE dl)
add_test(NAME ce_ccp1_g02_005 COMMAND ce_ccp1_g02_005)

add_executable(ce_ccp1_g02_006
  implement_custom_realization_pass_api_test.cc
  ${CELLERATOR_ROOT}/src/compiler/pass/implement_custom_realization_pass_api.cc)
target_include_directories(ce_ccp1_g02_006 PRIVATE ${CELLERATOR_ROOT}/include)
target_compile_features(ce_ccp1_g02_006 PRIVATE cxx_std_17)
target_compile_options(ce_ccp1_g02_006 PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(ce_ccp1_g02_006 PRIVATE
  CELLERATOR_TEST_INCLUDE_ROOT="${CELLERATOR_ROOT}/include")
target_link_libraries(ce_ccp1_g02_006 PRIVATE dl)
add_test(NAME ce_ccp1_g02_006 COMMAND ce_ccp1_g02_006)

add_executable(ce_ccp1_g02_007
    implement_complete_built_in_stage_replacement_test.cc
    ${CELLERATOR_ROOT}/src/compiler/pass/implement_complete_built_in_stage_replacement.cc
    ${CELLERATOR_ROOT}/src/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy.cc)
target_include_directories(ce_ccp1_g02_007 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_g02_007 COMMAND ce_ccp1_g02_007)

add_executable(ce_ccp1_g02_008
    define_extensible_operation_type_and_attribute_registrat_test.cc
    ${CELLERATOR_ROOT}/src/compiler/pass/define_extensible_operation_type_and_attribute_registrat.cc)
target_include_directories(ce_ccp1_g02_008 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_g02_008 COMMAND ce_ccp1_g02_008)

add_executable(ce_ccp1_g02_009
    implement_unknown_extension_preservation_and_forwarding_test.cc
    ${CELLERATOR_ROOT}/src/compiler/pass/implement_unknown_extension_preservation_and_forwarding.cc)
target_include_directories(ce_ccp1_g02_009 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_g02_009 COMMAND ce_ccp1_g02_009)

add_executable(ce_ccp1_g02_010
    implement_extension_capability_negotiation_test.cc
    ${CELLERATOR_ROOT}/src/compiler/pass/implement_extension_capability_negotiation.cc
    ${CELLERATOR_ROOT}/src/compiler/pass/implement_unknown_extension_preservation_and_forwarding.cc)
target_include_directories(ce_ccp1_g02_010 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_g02_010 COMMAND ce_ccp1_g02_010)
