cmake_minimum_required(VERSION 3.20)
project(cellerator_sdk_h02_tests LANGUAGES C CXX)
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
