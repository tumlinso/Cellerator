include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/define_the_host_only_root_project_contract.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/compiler/define_tri_state_cuda_backend_configuration.cmake)

# This directory is intentionally safe to include from a project declared with
# `project(... LANGUAGES CXX)`. CUDA is considered only by the accelerator
# configuration layer.
cellerator_validate_host_only_root_project_contract()
cellerator_configure_optional_cuda_backend()
