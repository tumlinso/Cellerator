#pragma once
#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <Cellerator/compute/operation/candidate_catalog_v3/catalog.h>
#include <cstdint>
#include <type_traits>
namespace cellerator::compiler::ir::planning::v1 {
namespace catalog_v3=cellerator::compute::operation::catalog_v3;
enum stage_resource_flags_v1:std::uint32_t{stage_graph_capture_v1=1u<<0u,stage_synchronizes_v1=1u<<1u,stage_uses_library_v1=1u<<2u,stage_transfers_v1=1u<<3u};
struct stage_inventory_v1{planning_identity_v1 stage{};std::uint64_t persistent_bytes=0,transient_bytes=0,transfer_bytes=0,library_identity=0,target_capabilities=0;std::uint32_t workspace_alignment=1,launch_count=0,stream_count=0,flags=0;};
struct resource_inventory_alternative_v1{planning_identity_v1 alternative{};const stage_inventory_v1*stages=nullptr;std::uint32_t stage_count=0,reserved=0;std::uint64_t persistent_bytes=0,peak_transient_bytes=0,total_transfer_bytes=0;std::uint32_t total_launch_count=0,required_stream_count=0;};
enum class resource_inventory_status_v1:std::uint8_t{ok=0,invalid_argument,invalid_identity,invalid_alignment,invalid_flags,aggregate_mismatch,catalog_mismatch};
resource_inventory_status_v1 validate_resource_inventory_alternative_v1(const resource_inventory_alternative_v1&) noexcept;
resource_inventory_status_v1 compare_candidate_resources_v1(const resource_inventory_alternative_v1&,const catalog_v3::candidate_descriptor_v3&) noexcept;
static_assert(std::is_trivially_copyable_v<stage_inventory_v1>);
} // namespace cellerator::compiler::ir::planning::v1
