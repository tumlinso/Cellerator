#pragma once
#include <Cellerator/compiler/composition/import_portable_schedule_ruleset_representation_v1.hh>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
struct cellshard_materialization_request_v1{std::uint64_t schedule_identity=0,structure_epoch=0,value_generation=0,byte_budget=0;std::vector<std::string> atom_requirements,target_classes;std::string delivery_contract;};
[[nodiscard]] std::optional<cellshard_materialization_request_v1> make_cellshard_materialization_request_v1(const portable_schedule_v1&,std::uint64_t epoch,std::uint64_t generation,std::uint64_t budget,std::vector<std::string> targets);
} // namespace Cellerator::compiler::composition
