#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::composition {

enum class production_value_role_v1 : std::uint8_t { input = 1, output, parameter };
enum class production_effect_v1 : std::uint8_t { pure = 1, reads_state, writes_values, changes_structure };

struct production_value_v1 {
    std::string name;
    std::string type;
    production_value_role_v1 role = production_value_role_v1::input;
    std::string domain_identity;
    std::string order_identity;
    std::uint64_t generation = 0;
};

struct exact_coverage_term_v1 { std::string value; std::int64_t coefficient = 1; };
struct exact_coverage_equation_v1 {
    std::string universe;
    std::vector<exact_coverage_term_v1> terms;
    bool disjoint = true;
};

struct production_cost_v1 {
    std::uint64_t preparation_bytes = 0;
    std::uint64_t execution_bytes = 0;
    std::uint64_t launch_count = 0;
    double estimated_nanoseconds = 0.0;
};

struct typed_production_contract_v1 {
    std::string stable_name;
    std::vector<production_value_v1> values;
    std::vector<exact_coverage_equation_v1> coverage;
    std::vector<production_effect_v1> effects;
    production_cost_v1 cost;
    std::string identity_rule;
    std::string order_rule;
    std::string generation_rule;
    std::string verifier;
};

[[nodiscard]] bool validate_typed_production_contract_v1(
    const typed_production_contract_v1 &contract, std::string *error = nullptr);

} // namespace Cellerator::compiler::composition
