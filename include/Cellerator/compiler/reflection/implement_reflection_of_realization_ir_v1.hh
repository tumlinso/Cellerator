#pragma once
#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
struct reflected_realization_v1{ir_handle_v1 handle{};std::string backend;std::vector<std::uint64_t>selected_cover,extents,projections,packing,stage_graph,resources,native_fragments;std::uint64_t structure_epoch=0,value_generation=0;};
[[nodiscard]] bool validate_reflected_realization_v1(const reflected_realization_v1&)noexcept;
[[nodiscard]] bool realization_is_accelerated_v1(const reflected_realization_v1&)noexcept;
}
