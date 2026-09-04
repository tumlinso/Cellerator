#pragma once

#include <Cellerator/compiler/backend/benchmark_cpu_backend_complete_cost_v1.hh>
#include <Cellerator/compiler/backend/deliver_the_first_cpu_object_milestone_v1.hh>
#include <Cellerator/compiler/backend/implement_cpu_projection_packing_and_order_transforms_v1.hh>
#include <Cellerator/compiler/backend/implement_cpu_segment_gate_update_bundle_and_chain_paths_v1.hh>
#include <Cellerator/compiler/backend/implement_cpu_transpose_and_contraction_v1.hh>
#include <Cellerator/compiler/backend/implement_generic_cpu_relation_apply_v1.hh>
#include <Cellerator/compiler/backend/implement_host_runtime_binding_abi_v1.hh>

namespace cellerator::compiler::backend::cpu::v1 {
inline constexpr std::uint32_t cpu_backend_contract_version_v1 = 1;
}
