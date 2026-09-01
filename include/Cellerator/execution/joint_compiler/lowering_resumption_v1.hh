#pragma once

#include <Cellerator/execution/joint_compiler/persistent_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::joint_compiler {

inline constexpr std::uint32_t lowering_resumption_schema_version_v1 = 1u;
inline constexpr std::uint64_t lowering_stage_count_v1 = 9u;

enum class lowering_stage_v1 : std::uint8_t {
    canonical = 1u,
    evidence = 2u,
    semantic_atom_basis = 3u,
    target_cover = 4u,
    physical_projection = 5u,
    packed_operand_value = 6u,
    executable_recipe = 7u,
    topology_linked = 8u,
    resident = 9u
};

enum class lowering_compatibility_v1 : std::uint8_t {
    compatible = 1u,
    requires_validation = 2u,
    incompatible = 3u
};

enum class lowering_fallback_v1 : std::uint8_t {
    reject = 1u,
    rebuild_from_canonical = 2u,
    rebuild_from_nearest_compatible = 3u
};

struct lowering_stage_record_v1 {
    lowering_stage_v1 stage = lowering_stage_v1::canonical;
    lowering_compatibility_v1 compatibility =
        lowering_compatibility_v1::requires_validation;
    std::uint8_t reserved[6]{};
    persistent_identity_v1 artifact_identity{};
    persistent_identity_v1 artifact_schema{};
    persistent_identity_v1 validation_receipt{};
};

struct lowering_resumption_v1 {
    std::uint32_t schema_version = lowering_resumption_schema_version_v1;
    std::uint32_t record_bytes = sizeof(lowering_resumption_v1);
    persistent_identity_v1 resumption_identity{};
    lowering_stage_v1 source_stage = lowering_stage_v1::canonical;
    lowering_stage_v1 target_stage = lowering_stage_v1::resident;
    lowering_fallback_v1 fallback = lowering_fallback_v1::reject;
    std::uint8_t reserved[5]{};
    const lowering_stage_record_v1 *stage_records = nullptr;
    std::uint64_t stage_record_count = 0u;
    const lowering_stage_v1 *bypassed_stages = nullptr;
    std::uint64_t bypassed_stage_count = 0u;
};

enum class lowering_resumption_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    nonzero_reserved = 3u,
    invalid_resumption_identity = 4u,
    invalid_source_stage = 5u,
    invalid_target_stage = 6u,
    invalid_stage_order = 7u,
    invalid_fallback = 8u,
    missing_stage_records = 9u,
    too_many_stage_records = 10u,
    invalid_stage_record = 11u,
    duplicate_or_unordered_stage_record = 12u,
    invalid_artifact_identity = 13u,
    invalid_artifact_schema = 14u,
    invalid_validation_receipt = 15u,
    source_record_missing = 16u,
    inconsistent_bypass_pointer = 17u,
    invalid_bypassed_stage = 18u,
    duplicate_or_unordered_bypassed_stage = 19u,
    bypass_record_missing = 20u,
    bypass_not_compatible = 21u
};

struct lowering_resumption_validation_result_v1 {
    lowering_resumption_validation_code_v1 code =
        lowering_resumption_validation_code_v1::ok;
    std::uint64_t index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == lowering_resumption_validation_code_v1::ok;
    }
};

lowering_resumption_validation_result_v1 validate_lowering_resumption_v1(
    const lowering_resumption_v1 &resumption) noexcept;

static_assert(std::is_standard_layout_v<lowering_stage_record_v1>);
static_assert(std::is_trivially_copyable_v<lowering_stage_record_v1>);
static_assert(std::is_standard_layout_v<lowering_resumption_v1>);
static_assert(std::is_trivially_copyable_v<lowering_resumption_v1>);

}  // namespace cellerator::execution::joint_compiler
