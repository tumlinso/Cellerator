include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/define_the_host_only_root_project_contract.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/define_tri_state_cuda_backend_configuration.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/create_compiler_component_target_boundaries.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/set_modern_compiler_tooling_language_standards.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/isolate_legacy_cuda_target_requirements.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/define_optional_llvm_and_clang_library_discovery.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/define_backend_resource_discovery_manifests.cmake)

# This directory is intentionally safe to include from a project declared with
# `project(... LANGUAGES CXX)`. CUDA is considered only by the accelerator
# configuration layer.
cellerator_validate_host_only_root_project_contract()
cellerator_configure_optional_cuda_backend()
cellerator_create_compiler_component_targets()
cellerator_discover_llvm_clang()
cellerator_generate_backend_resource_manifest(
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cellerator/backend-resources.txt")
