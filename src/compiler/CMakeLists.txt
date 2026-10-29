include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/host_only_root_project_contract.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/cuda_backend_configuration.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/compiler_component_targets.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/compiler_language_standards.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/legacy_cuda_target_configuration.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/llvm_clang_discovery.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/backend_resource_manifest.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/generated_compiler_header.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/host_only_compiler_smoke_targets.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/accelerator_compiler_smoke_target.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/compiler_ci_presets.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/compiler_target_graph.cmake)

get_filename_component(CELLERATOR_COMPILER_REPOSITORY_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# This directory is intentionally safe to include from a project declared with
# `project(... LANGUAGES CXX)`. CUDA is considered only by the accelerator
# configuration layer.
cellerator_validate_host_only_root_project_contract()
cellerator_configure_optional_cuda_backend()
cellerator_create_compiler_component_targets()
cellerator_discover_llvm_clang()
cellerator_generate_backend_resource_manifest(
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cellerator/backend-resources.txt")
cellerator_generate_compiler_build_header(
    "${CMAKE_CURRENT_BINARY_DIR}/generated/Cellerator/compiler/build/features_v1.hh")
cellerator_create_host_only_compiler_smoke_targets()
cellerator_create_accelerator_compiler_smoke_target()
cellerator_validate_compiler_ci_matrix()
cellerator_validate_frozen_compiler_target_graph()
