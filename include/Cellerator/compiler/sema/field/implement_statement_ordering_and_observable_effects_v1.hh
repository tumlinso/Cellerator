#pragma once

#include <Cellerator/compiler/sema/field/define_execution_field_semantic_ownership_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::sema::field {

using semantic_value_id_v1 = std::uint64_t;

struct generation_access_v1 {
    semantic_value_id_v1 value = 0;
    std::uint64_t generation = 0;
};

struct field_statement_semantics_v1 {
    std::uint64_t statement_id = 0;
    std::vector<semantic_value_id_v1> reads;
    std::vector<semantic_value_id_v1> writes;
    std::vector<generation_access_v1> generation_reads;
    std::vector<generation_access_v1> generation_writes;
    std::uint32_t observable_effects = field_effect_none_v1;
    std::uint64_t numerical_contract_id = 0;
    std::uint64_t field_constraint_set_id = 0;
};

enum class ordering_blocker_v1 : std::uint8_t {
    none = 0,
    invalid_statement,
    data_dependency,
    observable_effect,
    generation_dependency,
    numerical_contract,
    field_constraint,
};

struct statement_pair_analysis_v1 {
    ordering_blocker_v1 reorder_blocker = ordering_blocker_v1::none;
    ordering_blocker_v1 fusion_blocker = ordering_blocker_v1::none;

    [[nodiscard]] bool reorder_permitted() const noexcept {
        return reorder_blocker == ordering_blocker_v1::none;
    }
    [[nodiscard]] bool fusion_permitted() const noexcept {
        return fusion_blocker == ordering_blocker_v1::none;
    }
};

[[nodiscard]] statement_pair_analysis_v1
implement_statement_ordering_and_observable_effects_v1(
    const field_statement_semantics_v1& before,
    const field_statement_semantics_v1& after) noexcept;

}  // namespace Cellerator::compiler::sema::field
