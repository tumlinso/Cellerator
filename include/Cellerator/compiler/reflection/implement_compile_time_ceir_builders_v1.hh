#pragma once
#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
enum class ceir_builder_node_kind_v1:std::uint8_t{semantic_operation=1,attribute,region,planning_alternative,candidate,native_fragment};
struct ceir_builder_node_v1{ir_handle_v1 handle{};ceir_builder_node_kind_v1 kind=ceir_builder_node_kind_v1::semantic_operation;std::string name,payload;std::vector<std::uint32_t>children;};
struct ceir_builder_v1{std::uint64_t arena_epoch=1,next_identity=1;std::vector<ceir_builder_node_v1>nodes;std::vector<std::string>diagnostics;};
[[nodiscard]] std::uint32_t append_ceir_node_v1(ceir_builder_v1*,ceir_builder_node_kind_v1,const std::string&,const std::string&,const std::vector<std::uint32_t>& = {});
[[nodiscard]] bool validate_ceir_builder_v1(const ceir_builder_v1&)noexcept;
}
