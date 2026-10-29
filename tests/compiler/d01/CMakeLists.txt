cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_d01_tests LANGUAGES CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(cellerator_ceir_common_lexical_conventions_test
    ceir_common_lexical_conventions_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/ceir_common_lexical_conventions.cc")
target_include_directories(cellerator_ceir_common_lexical_conventions_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_ceir_common_lexical_conventions_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_ceir_common_lexical_conventions_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_001 ALIAS cellerator_ceir_common_lexical_conventions_test)
add_test(NAME ce_ccp1_d01_001 COMMAND cellerator_ceir_common_lexical_conventions_test)

add_executable(cellerator_arena_and_ownership_model_test
    arena_and_ownership_model_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/arena_and_ownership_model.cc")
target_include_directories(cellerator_arena_and_ownership_model_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_arena_and_ownership_model_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_arena_and_ownership_model_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_002 ALIAS cellerator_arena_and_ownership_model_test)
add_test(NAME ce_ccp1_d01_002 COMMAND cellerator_arena_and_ownership_model_test)

add_executable(cellerator_common_type_and_attribute_interning_test
    common_type_and_attribute_interning_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/common_type_and_attribute_interning.cc")
target_include_directories(cellerator_common_type_and_attribute_interning_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_common_type_and_attribute_interning_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_common_type_and_attribute_interning_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_003 ALIAS cellerator_common_type_and_attribute_interning_test)
add_test(NAME ce_ccp1_d01_003 COMMAND cellerator_common_type_and_attribute_interning_test)

add_executable(cellerator_regions_blocks_values_and_use_def_chains_test
    regions_blocks_values_and_use_def_chains_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/regions_blocks_values_and_use_def_chains.cc")
target_include_directories(cellerator_regions_blocks_values_and_use_def_chains_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_regions_blocks_values_and_use_def_chains_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_regions_blocks_values_and_use_def_chains_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_004 ALIAS cellerator_regions_blocks_values_and_use_def_chains_test)
add_test(NAME ce_ccp1_d01_004 COMMAND cellerator_regions_blocks_values_and_use_def_chains_test)

add_executable(cellerator_common_operation_and_extension_records_test
    common_operation_and_extension_records_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/common_operation_and_extension_records.cc")
target_include_directories(cellerator_common_operation_and_extension_records_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_common_operation_and_extension_records_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_common_operation_and_extension_records_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_005 ALIAS cellerator_common_operation_and_extension_records_test)
add_test(NAME ce_ccp1_d01_005 COMMAND cellerator_common_operation_and_extension_records_test)

add_executable(cellerator_persistent_and_transient_identity_layers_test
    persistent_and_transient_identity_layers_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/persistent_and_transient_identity_layers.cc")
target_include_directories(cellerator_persistent_and_transient_identity_layers_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_persistent_and_transient_identity_layers_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_persistent_and_transient_identity_layers_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_006 ALIAS cellerator_persistent_and_transient_identity_layers_test)
add_test(NAME ce_ccp1_d01_006 COMMAND cellerator_persistent_and_transient_identity_layers_test)

add_executable(cellerator_removable_provenance_sidecars_test
    removable_provenance_sidecars_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/removable_provenance_sidecars.cc")
target_include_directories(cellerator_removable_provenance_sidecars_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_removable_provenance_sidecars_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_removable_provenance_sidecars_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_007 ALIAS cellerator_removable_provenance_sidecars_test)
add_test(NAME ce_ccp1_d01_007 COMMAND cellerator_removable_provenance_sidecars_test)

add_executable(cellerator_ceir_text_lexer_and_parser_framework_test
    ceir_text_lexer_and_parser_framework_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/ceir_text_lexer_and_parser_framework.cc")
target_include_directories(cellerator_ceir_text_lexer_and_parser_framework_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_ceir_text_lexer_and_parser_framework_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_ceir_text_lexer_and_parser_framework_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_008 ALIAS cellerator_ceir_text_lexer_and_parser_framework_test)
add_test(NAME ce_ccp1_d01_008 COMMAND cellerator_ceir_text_lexer_and_parser_framework_test)

add_executable(cellerator_deterministic_canonical_printing_test
    deterministic_canonical_printing_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/deterministic_canonical_printing.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/ceir_text_lexer_and_parser_framework.cc")
target_include_directories(cellerator_deterministic_canonical_printing_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_deterministic_canonical_printing_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_deterministic_canonical_printing_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_009 ALIAS cellerator_deterministic_canonical_printing_test)
add_test(NAME ce_ccp1_d01_009 COMMAND cellerator_deterministic_canonical_printing_test)

add_executable(cellerator_sectioned_binary_ceir_serialization_test
    sectioned_binary_ceir_serialization_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/sectioned_binary_ceir_serialization.cc")
target_include_directories(cellerator_sectioned_binary_ceir_serialization_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_sectioned_binary_ceir_serialization_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_sectioned_binary_ceir_serialization_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_010 ALIAS cellerator_sectioned_binary_ceir_serialization_test)
add_test(NAME ce_ccp1_d01_010 COMMAND cellerator_sectioned_binary_ceir_serialization_test)

add_executable(cellerator_standalone_ceir_compiler_input_detection_test
    standalone_ceir_compiler_input_detection_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/standalone_ceir_compiler_input_detection.cc")
target_include_directories(cellerator_standalone_ceir_compiler_input_detection_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_standalone_ceir_compiler_input_detection_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_standalone_ceir_compiler_input_detection_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_011 ALIAS cellerator_standalone_ceir_compiler_input_detection_test)
add_test(NAME ce_ccp1_d01_011 COMMAND cellerator_standalone_ceir_compiler_input_detection_test)

add_executable(cellerator_validation_mode_plumbing_test
    validation_mode_plumbing_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/validation_mode_plumbing.cc")
target_include_directories(cellerator_validation_mode_plumbing_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_validation_mode_plumbing_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_validation_mode_plumbing_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_012 ALIAS cellerator_validation_mode_plumbing_test)
add_test(NAME ce_ccp1_d01_012 COMMAND cellerator_validation_mode_plumbing_test)

add_executable(ce_ccp1_d01_013
    expose_the_public_ceir_c_api_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/expose_the_public_ceir_c_api.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/common_operation_and_extension_records.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/deterministic_canonical_printing.cc")
target_include_directories(ce_ccp1_d01_013 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d01_013 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d01_013 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d01_013 COMMAND ce_ccp1_d01_013)

add_executable(cellerator_ceir_common_round_trip_and_artifact_compatibility_test
    ceir_common_round_trip_and_artifact_compatibility_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/ceir_common_round_trip_and_artifact_compatibility.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/deterministic_canonical_printing.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/ceir_text_lexer_and_parser_framework.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/sectioned_binary_ceir_serialization.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/standalone_ceir_compiler_input_detection.cc")
target_include_directories(cellerator_ceir_common_round_trip_and_artifact_compatibility_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_ceir_common_round_trip_and_artifact_compatibility_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_ceir_common_round_trip_and_artifact_compatibility_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_d01_014 ALIAS cellerator_ceir_common_round_trip_and_artifact_compatibility_test)
add_test(NAME ce_ccp1_d01_014 COMMAND cellerator_ceir_common_round_trip_and_artifact_compatibility_test)
