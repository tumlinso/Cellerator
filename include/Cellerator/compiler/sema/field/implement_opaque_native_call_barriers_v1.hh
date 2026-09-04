#pragma once

#include <Cellerator/compiler/frontend/cxx/model_opaque_native_calls_v1.hh>
#include <Cellerator/compiler/sema/field/implement_statement_ordering_and_observable_effects_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::sema::field {

struct opaque_native_call_barrier_v1 {
    std::uint64_t statement_id = 0;
    std::string qualified_name;
    std::vector<semantic_value_id_v1> invalidated_generation_values;
    bool invalidates_profile_state = true;
    bool stops_cross_call_planning = true;
    field_statement_semantics_v1 statement;
    std::string diagnostic;
    std::string planning_report;
};

enum class opaque_barrier_status_v1 : std::uint8_t {
    success = 0,
    invalid_call,
    contracted_call,
    invalid_statement,
    invalid_affected_value,
};

[[nodiscard]] opaque_barrier_status_v1 implement_opaque_native_call_barriers_v1(
    const frontend::cxx::opaque_native_call_v1& call,
    std::uint64_t statement_id,
    const std::vector<semantic_value_id_v1>& affected_values,
    opaque_native_call_barrier_v1* barrier) noexcept;

}  // namespace Cellerator::compiler::sema::field
