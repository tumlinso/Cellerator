#pragma once

#include <Cellerator/compiler/backend/nvptx/compare_nvcc_clang_cuda_and_direct_ptx_routes_v1.hh>
#include <Cellerator/compiler/backend/nvptx/direct_ptx_typed_operation_model_v1.hh>
#include <Cellerator/compiler/backend/nvptx/direct_ptx_hot_path_demonstration_v1.hh>
#include <Cellerator/compiler/backend/nvptx/clang_cuda_and_nvptx_backend_contracts_v1.hh>
#include <Cellerator/compiler/backend/nvptx/optional_nvidia_backend_routes_v1.hh>
#include <Cellerator/compiler/backend/nvptx/clang_cuda_action_mapping_v1.hh>
#include <Cellerator/compiler/backend/nvptx/fatbinary_object_embedding_for_direct_ptx_v1.hh>
#include <Cellerator/compiler/backend/nvptx/inline_ptx_native_block_binding_v1.hh>
#include <Cellerator/compiler/backend/nvptx/llvm_nvptx_module_boundary_v1.hh>
#include <Cellerator/compiler/backend/nvptx/ptx_emission_and_ptxas_assembly_v1.hh>
#include <Cellerator/compiler/backend/nvptx/source_to_ptx_provenance_v1.hh>
#include <Cellerator/compiler/backend/nvptx/map_target_capabilities_and_instruction_families_v1.hh>
#include <Cellerator/compiler/backend/nvptx/validate_backend_fallback_and_mixed_routes_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::backend::nvptx {

inline constexpr std::uint32_t nvptx_backend_interface_version_v1 = 1u;

}  // namespace Cellerator::compiler::backend::nvptx
