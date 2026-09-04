#pragma once
#include <Cellerator/compiler/ir/realization/implement_memory_workspace_and_residency_requirements_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_order_transforms_and_persistent_physical_order_v1.hh>
#include <Cellerator/execution/program/program_v2.h>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::ir::realization::v1 {
enum class prepared_stage_kind_v1:std::uint8_t{kernel=1,host_stub,pack,order_transform,publish};
struct stage_range_v1{std::uint64_t global_begin=0,global_count=0,local_begin=0,local_count=0;};
struct prepared_stage_ir_v1{stable_identity_v1 identity{},candidate{},binding{};prepared_stage_kind_v1 kind=prepared_stage_kind_v1::kernel;std::vector<std::uint32_t>dependencies;std::vector<stable_identity_v1>resources;order_identity_v1 input_order{},output_order{};std::uint64_t structure_epoch=0,input_generation=0,output_generation=0;stage_range_v1 range{};std::uint32_t profiler_index=0;std::uint64_t workspace_bytes=0;};
struct prepared_stage_graph_v1{stable_identity_v1 identity{};std::vector<prepared_stage_ir_v1>stages;};
enum class stage_graph_status_v1:std::uint8_t{valid=0,invalid_identity,duplicate_stage,forward_dependency,duplicate_dependency,invalid_order,invalid_epoch,invalid_range,duplicate_profiler_index,program_v2_mismatch};
[[nodiscard]] stage_graph_status_v1 validate_prepared_stage_graph_v1(const prepared_stage_graph_v1&,std::string*error=nullptr) noexcept;
[[nodiscard]] stage_graph_status_v1 compare_program_v2_graph_v1(const prepared_stage_graph_v1&,const cellerator::execution::program::prepared_program_v2&,std::string*error=nullptr) noexcept;
} // namespace cellerator::compiler::ir::realization::v1
