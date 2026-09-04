#pragma once

#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_execution_field_operations_and_regions_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_state_and_value_plane_ir_types_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

struct semantic_canonical_record_v1 {
    semantic_identity_v1 operation_identity{};
    std::string operation_spelling;
    std::vector<semantic_identity_v1> input_types;
    std::vector<semantic_identity_v1> output_types;
    std::vector<semantic_identity_v1> biological_identities;
    numeric_tuple_ir_v1 numerical{};
    std::uint32_t effects = 0;
    std::uint64_t field_identity = 0;
    execution_field_boundary_ir_v1 field_boundary =
        execution_field_boundary_ir_v1::transparent;
};

struct semantic_fingerprint_v1 {
    std::uint64_t low = 0;
    std::uint64_t high = 0;

    [[nodiscard]] constexpr bool valid() const noexcept { return low != 0 || high != 0; }
};

enum class semantic_canonicalization_status_v1 : std::uint8_t {
    success = 0,
    invalid_operation_identity,
    invalid_operation_spelling,
    invalid_type_identity,
    invalid_biological_identity,
    invalid_numerical_contract,
    invalid_field_identity,
    invalid_field_boundary,
};

[[nodiscard]] std::optional<semantic_canonical_record_v1>
canonicalize_semantic_record_v1(
    semantic_canonical_record_v1 record,
    semantic_canonicalization_status_v1* status = nullptr) noexcept;

[[nodiscard]] std::optional<semantic_fingerprint_v1>
fingerprint_semantic_record_v1(
    const semantic_canonical_record_v1& record,
    semantic_canonicalization_status_v1* status = nullptr) noexcept;

[[nodiscard]] bool semantic_equivalent_v1(
    const semantic_canonical_record_v1& left,
    const semantic_canonical_record_v1& right) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
