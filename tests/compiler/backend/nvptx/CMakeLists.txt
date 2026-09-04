cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_nvptx_tests LANGUAGES CXX)

enable_testing()
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(ce_ccp1_f04_001
    freeze_clang_cuda_and_nvptx_backend_contracts_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/freeze_clang_cuda_and_nvptx_backend_contracts.cc")
target_include_directories(ce_ccp1_f04_001 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_001 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_001 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_001 COMMAND ce_ccp1_f04_001)

add_executable(ce_ccp1_f04_002
    implement_clang_cuda_action_mapping_test.cc
    "${CELLERATOR_ROOT}/src/compiler/backend/nvptx/implement_clang_cuda_action_mapping.cc")
target_include_directories(ce_ccp1_f04_002 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_f04_002 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_f04_002 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_f04_002 COMMAND ce_ccp1_f04_002)
