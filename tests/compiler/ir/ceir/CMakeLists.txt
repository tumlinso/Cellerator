cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_d01_tests LANGUAGES CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(cellerator_ceir_common_lexical_conventions_test
    ceir_common_lexical_conventions_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/ceir_common_lexical_conventions.cc")
target_include_directories(cellerator_ceir_common_lexical_conventions_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_ceir_common_lexical_conventions_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_ceir_common_lexical_conventions_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_ceir_common_lexical_conventions_test COMMAND cellerator_ceir_common_lexical_conventions_test)

add_executable(cellerator_arena_and_ownership_model_test
    arena_and_ownership_model_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/arena_and_ownership_model.cc")
target_include_directories(cellerator_arena_and_ownership_model_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_arena_and_ownership_model_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_arena_and_ownership_model_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_arena_and_ownership_model_test COMMAND cellerator_arena_and_ownership_model_test)

add_executable(cellerator_common_type_and_attribute_interning_test
    common_type_and_attribute_interning_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/common_type_and_attribute_interning.cc")
target_include_directories(cellerator_common_type_and_attribute_interning_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_common_type_and_attribute_interning_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_common_type_and_attribute_interning_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_common_type_and_attribute_interning_test COMMAND cellerator_common_type_and_attribute_interning_test)

add_executable(cellerator_regions_blocks_values_and_use_def_chains_test
    regions_blocks_values_and_use_def_chains_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/regions_blocks_values_and_use_def_chains.cc")
target_include_directories(cellerator_regions_blocks_values_and_use_def_chains_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_regions_blocks_values_and_use_def_chains_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_regions_blocks_values_and_use_def_chains_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_regions_blocks_values_and_use_def_chains_test COMMAND cellerator_regions_blocks_values_and_use_def_chains_test)

add_executable(cellerator_common_operation_and_extension_records_test
    common_operation_and_extension_records_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/common_operation_and_extension_records.cc")
target_include_directories(cellerator_common_operation_and_extension_records_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_common_operation_and_extension_records_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_common_operation_and_extension_records_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_common_operation_and_extension_records_test COMMAND cellerator_common_operation_and_extension_records_test)

add_executable(cellerator_persistent_and_transient_identity_layers_test
    persistent_and_transient_identity_layers_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/persistent_and_transient_identity_layers.cc")
target_include_directories(cellerator_persistent_and_transient_identity_layers_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_persistent_and_transient_identity_layers_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_persistent_and_transient_identity_layers_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_persistent_and_transient_identity_layers_test COMMAND cellerator_persistent_and_transient_identity_layers_test)

add_executable(cellerator_removable_provenance_sidecars_test
    removable_provenance_sidecars_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/removable_provenance_sidecars.cc")
target_include_directories(cellerator_removable_provenance_sidecars_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_removable_provenance_sidecars_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_removable_provenance_sidecars_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_removable_provenance_sidecars_test COMMAND cellerator_removable_provenance_sidecars_test)

add_executable(cellerator_ceir_text_lexer_and_parser_framework_test
    ceir_text_lexer_and_parser_framework_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/ceir_text_lexer_and_parser_framework.cc")
target_include_directories(cellerator_ceir_text_lexer_and_parser_framework_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_ceir_text_lexer_and_parser_framework_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_ceir_text_lexer_and_parser_framework_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_ceir_text_lexer_and_parser_framework_test COMMAND cellerator_ceir_text_lexer_and_parser_framework_test)

add_executable(cellerator_deterministic_canonical_printing_test
    deterministic_canonical_printing_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/deterministic_canonical_printing.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/ceir_text_lexer_and_parser_framework.cc")
target_include_directories(cellerator_deterministic_canonical_printing_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_deterministic_canonical_printing_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_deterministic_canonical_printing_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_deterministic_canonical_printing_test COMMAND cellerator_deterministic_canonical_printing_test)

add_executable(cellerator_sectioned_binary_ceir_serialization_test
    sectioned_binary_ceir_serialization_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/sectioned_binary_ceir_serialization.cc")
target_include_directories(cellerator_sectioned_binary_ceir_serialization_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_sectioned_binary_ceir_serialization_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_sectioned_binary_ceir_serialization_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_sectioned_binary_ceir_serialization_test COMMAND cellerator_sectioned_binary_ceir_serialization_test)

add_executable(cellerator_standalone_ceir_compiler_input_detection_test
    standalone_ceir_compiler_input_detection_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/standalone_ceir_compiler_input_detection.cc")
target_include_directories(cellerator_standalone_ceir_compiler_input_detection_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_standalone_ceir_compiler_input_detection_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_standalone_ceir_compiler_input_detection_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_standalone_ceir_compiler_input_detection_test COMMAND cellerator_standalone_ceir_compiler_input_detection_test)

add_executable(cellerator_validation_mode_plumbing_test
    validation_mode_plumbing_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/validation_mode_plumbing.cc")
target_include_directories(cellerator_validation_mode_plumbing_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_validation_mode_plumbing_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_validation_mode_plumbing_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_validation_mode_plumbing_test COMMAND cellerator_validation_mode_plumbing_test)

add_executable(cellerator_ceir_builder_api_test
    ceir_builder_api_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/common/ceir_builder_api.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/common_operation_and_extension_records.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/common/deterministic_canonical_printing.cc")
target_include_directories(cellerator_ceir_builder_api_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_ceir_builder_api_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_ceir_builder_api_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_ceir_builder_api_test COMMAND cellerator_ceir_builder_api_test)

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
add_test(NAME cellerator_ceir_common_round_trip_and_artifact_compatibility_test COMMAND cellerator_ceir_common_round_trip_and_artifact_compatibility_test)
