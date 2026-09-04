#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::sema::field {

enum class planning_fact_or_preference_kind_v1 : std::uint8_t {
    reuse = 1,
    memory,
    latency,
    throughput,
    compilation_budget,
    target_preference,
    graph_capture,
    canonical_output,
};

struct planning_fact_or_preference_v1 {
    std::uint64_t source_identity = 0;
    planning_fact_or_preference_kind_v1 kind =
        planning_fact_or_preference_kind_v1::latency;
    double magnitude = 1.0;
    std::uint64_t named_value_identity = 0;
    bool fact = false;
    bool supported = true;
};

enum class planning_hint_disposition_v1 : std::uint8_t {
    applied = 1,
    ignored,
    dominated,
};

struct resolved_planning_hint_v1 {
    planning_fact_or_preference_v1 hint;
    planning_hint_disposition_v1 disposition = planning_hint_disposition_v1::ignored;
    std::string diagnostic;
};

struct planning_facts_and_preferences_v1 {
    std::vector<resolved_planning_hint_v1> hints;
    std::size_t applied_count = 0;
    std::size_t ignored_count = 0;
    std::size_t dominated_count = 0;
};

enum class planning_facts_and_preferences_status_v1 : std::uint8_t {
    success = 0,
    invalid_output,
    invalid_hint,
};

[[nodiscard]] planning_facts_and_preferences_status_v1
implement_planning_facts_and_preferences_v1(
    const std::vector<planning_fact_or_preference_v1>& hints,
    planning_facts_and_preferences_v1* resolved) noexcept;

}  // namespace Cellerator::compiler::sema::field
