cmake_minimum_required(VERSION 3.20)
project(cellerator_sdk_h02_tests LANGUAGES C CXX)
find_package(Threads REQUIRED)
enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_executable(ce_ccp1_h02_001
    ../compiler/h02/freeze_libcellerator_component_architecture_test.cc
    ${CELLERATOR_ROOT}/src/compiler/api/freeze_libcellerator_component_architecture.cc)
target_include_directories(ce_ccp1_h02_001 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_h02_001 COMMAND ce_ccp1_h02_001)

add_executable(ce_ccp1_h02_002
    define_c_compiler_session_api_test.c
    ${CELLERATOR_ROOT}/src/compiler/api/define_c_compiler_session_api.cc)
target_include_directories(ce_ccp1_h02_002 PRIVATE ${CELLERATOR_ROOT}/include)
set_target_properties(ce_ccp1_h02_002 PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME ce_ccp1_h02_002 COMMAND ce_ccp1_h02_002)

add_executable(ce_ccp1_h02_003
    ../compiler/h02/define_c_compiler_session_api_test.cc
    ${CELLERATOR_ROOT}/src/compiler/api/define_c_compiler_session_api.cc)
target_include_directories(ce_ccp1_h02_003 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_h02_003 COMMAND ce_ccp1_h02_003)

add_executable(ce_ccp1_h02_004
    ../compiler/h02/expose_parse_and_semantic_analysis_apis_test.cc
    ${CELLERATOR_ROOT}/src/compiler/api/expose_parse_and_semantic_analysis_apis.cc)
target_include_directories(ce_ccp1_h02_004 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_h02_004 COMMAND ce_ccp1_h02_004)
add_executable(ce_ccp1_h02_005 ../compiler/h02/expose_ceir_ownership_and_mutation_apis_test.cc ${CELLERATOR_ROOT}/src/compiler/api/expose_ceir_ownership_and_mutation_apis.cc)
target_include_directories(ce_ccp1_h02_005 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_h02_005 COMMAND ce_ccp1_h02_005)
add_executable(ce_ccp1_h02_006 ../compiler/h02/expose_profile_build_load_query_apis_test.cc ${CELLERATOR_ROOT}/src/compiler/api/expose_profile_build_load_query_apis.cc)
target_include_directories(ce_ccp1_h02_006 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_h02_006 COMMAND ce_ccp1_h02_006)
add_executable(ce_ccp1_h02_007 ../compiler/h02/expose_planning_and_candidate_apis_test.cc ${CELLERATOR_ROOT}/src/compiler/api/expose_planning_and_candidate_apis.cc)
target_include_directories(ce_ccp1_h02_007 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_h02_007 COMMAND ce_ccp1_h02_007)
add_executable(ce_ccp1_h02_008 ../compiler/h02/expose_realization_and_backend_apis_test.cc ${CELLERATOR_ROOT}/src/compiler/api/expose_realization_and_backend_apis.cc)
target_include_directories(ce_ccp1_h02_008 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_h02_008 COMMAND ce_ccp1_h02_008)
add_executable(ce_ccp1_h02_009 ../compiler/h02/expose_pass_reflection_and_extension_apis_test.cc ${CELLERATOR_ROOT}/src/compiler/api/expose_pass_reflection_and_extension_apis.cc ${CELLERATOR_ROOT}/src/compiler/pass/define_extensible_operation_type_and_attribute_registrat.cc)
target_include_directories(ce_ccp1_h02_009 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_h02_009 COMMAND ce_ccp1_h02_009)
add_executable(ce_ccp1_h02_010 ../compiler/h02/expose_structured_diagnostics_and_explainability_apis_test.cc ${CELLERATOR_ROOT}/src/compiler/api/expose_structured_diagnostics_and_explainability_apis.cc)
target_include_directories(ce_ccp1_h02_010 PRIVATE ${CELLERATOR_ROOT}/include)
target_link_libraries(ce_ccp1_h02_010 PRIVATE Threads::Threads)
add_test(NAME ce_ccp1_h02_010 COMMAND ce_ccp1_h02_010)
add_executable(ce_ccp1_h02_011 ../compiler/h02/define_thread_safety_and_context_isolation_test.cc ${CELLERATOR_ROOT}/src/compiler/api/define_thread_safety_and_context_isolation.cc)
target_include_directories(ce_ccp1_h02_011 PRIVATE ${CELLERATOR_ROOT}/include)
target_link_libraries(ce_ccp1_h02_011 PRIVATE Threads::Threads)
add_test(NAME ce_ccp1_h02_011 COMMAND ce_ccp1_h02_011)
add_executable(ce_ccp1_h02_012 ../compiler/h02/preserve_direct_runtime_execution_access_test.cc ${CELLERATOR_ROOT}/src/compiler/api/preserve_direct_runtime_execution_access.cc)
target_include_directories(ce_ccp1_h02_012 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_h02_012 COMMAND ce_ccp1_h02_012)
