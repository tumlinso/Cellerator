#pragma once
// PROPOSED POST-EPIC DECLARATIONS ONLY. This bridge must really parse and lower.
#include <Cellerator/compute/operation/relation_update.hh>
#include <string>
#include <string_view>
#include <vector>
namespace Cellerator::compiler::sema {
// Loaded metadata is not an already constructed operation or a provider choice.
struct relation_update_source_environment {
    std::string relation_name, input_name, cotangent_name, destination_name;
    cellerator::compute::relation::topology_descriptor topology;
    cellerator::compute::relation::arithmetic_policy arithmetic;
    std::uint32_t dense_width = 16;
    cellerator::compute::relation::gradient_arithmetic gradient_arithmetic =
        cellerator::compute::relation::gradient_arithmetic::full_f32;
};
struct relation_update_source_result {
    cellerator::compute::relation::relation_calculus_descriptor semantic;
    std::vector<std::string> diagnostics;
    bool lowered = false; // set only after real parse/Sema/IR validation
    // Production implementation should retain actual source ranges/provenance too.
    bool accepted() const noexcept { return lowered && diagnostics.empty(); }
};
relation_update_source_result lower_relation_update_source_slice_v1(
    std::string_view source,const relation_update_source_environment&);
} // namespace Cellerator::compiler::sema
