cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_d01_tests LANGUAGES CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(ce_ccp1_d01_001
    freeze_ceir_common_lexical_conventions_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/freeze_ceir_common_lexical_conventions.cc")
target_include_directories(ce_ccp1_d01_001 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d01_001 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d01_001 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d01_001 COMMAND ce_ccp1_d01_001)

add_executable(ce_ccp1_d01_002
    implement_arena_and_ownership_model_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/implement_arena_and_ownership_model.cc")
target_include_directories(ce_ccp1_d01_002 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d01_002 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d01_002 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d01_002 COMMAND ce_ccp1_d01_002)

add_executable(ce_ccp1_d01_003
    implement_common_type_and_attribute_interning_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/implement_common_type_and_attribute_interning.cc")
target_include_directories(ce_ccp1_d01_003 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d01_003 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d01_003 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d01_003 COMMAND ce_ccp1_d01_003)

add_executable(ce_ccp1_d01_004
    implement_regions_blocks_values_and_use_def_chains_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/implement_regions_blocks_values_and_use_def_chains.cc")
target_include_directories(ce_ccp1_d01_004 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d01_004 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d01_004 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d01_004 COMMAND ce_ccp1_d01_004)

add_executable(ce_ccp1_d01_005
    implement_common_operation_and_extension_records_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/implement_common_operation_and_extension_records.cc")
target_include_directories(ce_ccp1_d01_005 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d01_005 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d01_005 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d01_005 COMMAND ce_ccp1_d01_005)

add_executable(ce_ccp1_d01_006
    implement_persistent_and_transient_identity_layers_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/implement_persistent_and_transient_identity_layers.cc")
target_include_directories(ce_ccp1_d01_006 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d01_006 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d01_006 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d01_006 COMMAND ce_ccp1_d01_006)

add_executable(ce_ccp1_d01_007
    implement_removable_provenance_sidecars_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/implement_removable_provenance_sidecars.cc")
target_include_directories(ce_ccp1_d01_007 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d01_007 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d01_007 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d01_007 COMMAND ce_ccp1_d01_007)

add_executable(ce_ccp1_d01_008
    implement_the_ceir_text_lexer_and_parser_framework_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/implement_the_ceir_text_lexer_and_parser_framework.cc")
target_include_directories(ce_ccp1_d01_008 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d01_008 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d01_008 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d01_008 COMMAND ce_ccp1_d01_008)

add_executable(ce_ccp1_d01_009
    implement_deterministic_canonical_printing_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/implement_deterministic_canonical_printing.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/implement_the_ceir_text_lexer_and_parser_framework.cc")
target_include_directories(ce_ccp1_d01_009 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d01_009 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d01_009 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d01_009 COMMAND ce_ccp1_d01_009)
