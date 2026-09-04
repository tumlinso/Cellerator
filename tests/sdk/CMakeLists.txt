cmake_minimum_required(VERSION 3.20)
project(cellerator_sdk_h02_tests LANGUAGES CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_executable(ce_ccp1_h02_001
    ../compiler/h02/freeze_libcellerator_component_architecture_test.cc
    ${CELLERATOR_ROOT}/src/compiler/api/freeze_libcellerator_component_architecture.cc)
target_include_directories(ce_ccp1_h02_001 PRIVATE ${CELLERATOR_ROOT}/include)
add_test(NAME ce_ccp1_h02_001 COMMAND ce_ccp1_h02_001)
