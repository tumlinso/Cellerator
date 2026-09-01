#pragma once

#include <Cellerator/execution/geometry_acquisition_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::lowering_resumption {

using stable_identity_v1 = acquisition_v2::stable_identity;

enum class lowering_stage_v1 : std::uint8_t {
    canonical_source = 1u,
    atom_evidence = 2u,
    semantic_atom = 3u,
    target_cover = 4u,
    physical_projection = 5u,
    packed_operand = 6u,
    executable_recipe = 7u,
    local_realization = 8u,
};

enum class compatibility_code_v1 : std::uint8_t {
    compatible = 0u,
    invalid_argument,
    corrupt_artifact,
    identity_mismatch,
    structure_epoch_mismatch,
    order_mismatch,
    target_mismatch,
    value_generation_stale,
    toolchain_mismatch,
    insufficient_capacity,
};

struct compatibility_status_v1 {
    compatibility_code_v1 code = compatibility_code_v1::compatible;
    lowering_stage_v1 inspected_stage = lowering_stage_v1::canonical_source;
    lowering_stage_v1 earliest_compatible_stage =
        lowering_stage_v1::canonical_source;
    std::uint8_t reserved[5]{};
    std::uint64_t detail = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == compatibility_code_v1::compatible;
    }
};

struct lowering_identity_context_v1 {
    stable_identity_v1 structure_identity{};
    std::uint64_t structure_epoch = 0u;
    stable_identity_v1 order_identity{};
    stable_identity_v1 target_identity{};
    stable_identity_v1 toolchain_identity{};
    std::uint64_t value_generation = 0u;
};

struct canonical_source_input_v1 {
    stable_identity_v1 source_identity{};
    const void *bytes = nullptr;
    std::uint64_t byte_count = 0u;
    std::uint64_t content_hash[4]{};
};

struct resumption_cursor_v1 {
    lowering_stage_v1 stage = lowering_stage_v1::canonical_source;
    stable_identity_v1 artifact_identity{};
    lowering_identity_context_v1 context{};
    const void *payload = nullptr;
    std::uint64_t payload_bytes = 0u;
};

struct lowering_artifact_v1 {
    std::uint32_t version = 1u;
    std::uint32_t record_bytes = sizeof(lowering_artifact_v1);
    lowering_stage_v1 stage = lowering_stage_v1::atom_evidence;
    std::uint8_t reserved[7]{};
    stable_identity_v1 artifact_identity{};
    stable_identity_v1 cover_identity{};
    lowering_identity_context_v1 context{};
    const void *payload = nullptr;
    std::uint64_t payload_bytes = 0u;
    std::uint64_t content_hash[4]{};
};

constexpr lowering_stage_v1 earliest_stage_for_v1(
    compatibility_code_v1 code, lowering_stage_v1 inspected) noexcept {
    switch (code) {
        case compatibility_code_v1::compatible:
            return inspected;
        case compatibility_code_v1::value_generation_stale:
            return lowering_stage_v1::packed_operand;
        case compatibility_code_v1::target_mismatch:
        case compatibility_code_v1::toolchain_mismatch:
            return lowering_stage_v1::physical_projection;
        case compatibility_code_v1::order_mismatch:
            return lowering_stage_v1::semantic_atom;
        case compatibility_code_v1::structure_epoch_mismatch:
            return lowering_stage_v1::atom_evidence;
        case compatibility_code_v1::corrupt_artifact:
        case compatibility_code_v1::identity_mismatch:
        case compatibility_code_v1::invalid_argument:
        case compatibility_code_v1::insufficient_capacity:
            return lowering_stage_v1::canonical_source;
    }
    return lowering_stage_v1::canonical_source;
}

constexpr compatibility_status_v1 make_status_v1(
    compatibility_code_v1 code, lowering_stage_v1 inspected,
    std::uint64_t detail = 0u) noexcept {
    return {code, inspected, earliest_stage_for_v1(code, inspected), {}, detail};
}

constexpr bool valid_identity_v1(stable_identity_v1 identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

constexpr bool valid_hash_v1(const std::uint64_t (&hash)[4]) noexcept {
    return hash[0] != 0u || hash[1] != 0u || hash[2] != 0u || hash[3] != 0u;
}

constexpr bool valid_context_v1(
    const lowering_identity_context_v1 &context) noexcept {
    return valid_identity_v1(context.structure_identity) &&
        context.structure_epoch != 0u &&
        valid_identity_v1(context.order_identity) &&
        valid_identity_v1(context.target_identity) &&
        valid_identity_v1(context.toolchain_identity);
}

inline compatibility_status_v1 resume_from_canonical_source_v1(
    const canonical_source_input_v1 &source,
    const lowering_identity_context_v1 &context,
    resumption_cursor_v1 *cursor) noexcept {
    if (cursor == nullptr || !valid_identity_v1(source.source_identity) ||
        source.bytes == nullptr || source.byte_count == 0u ||
        !valid_hash_v1(source.content_hash) || !valid_context_v1(context)) {
        return make_status_v1(compatibility_code_v1::invalid_argument,
            lowering_stage_v1::canonical_source);
    }
    *cursor = {lowering_stage_v1::canonical_source,
        source.source_identity, context, source.bytes, source.byte_count};
    return make_status_v1(compatibility_code_v1::compatible,
        lowering_stage_v1::canonical_source);
}

constexpr bool same_identity_v1(
    stable_identity_v1 left, stable_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

inline compatibility_status_v1 validate_artifact_for_stage_v1(
    const lowering_artifact_v1 &artifact, lowering_stage_v1 expected_stage,
    const lowering_identity_context_v1 &expected) noexcept {
    if (artifact.version != 1u ||
        artifact.record_bytes != sizeof(lowering_artifact_v1) ||
        artifact.stage != expected_stage ||
        !valid_identity_v1(artifact.artifact_identity) ||
        artifact.payload == nullptr || artifact.payload_bytes == 0u ||
        !valid_hash_v1(artifact.content_hash) ||
        !valid_context_v1(artifact.context) || !valid_context_v1(expected)) {
        return make_status_v1(compatibility_code_v1::corrupt_artifact,
            expected_stage);
    }
    if (!same_identity_v1(artifact.context.structure_identity,
            expected.structure_identity)) {
        return make_status_v1(compatibility_code_v1::identity_mismatch,
            expected_stage);
    }
    if (artifact.context.structure_epoch != expected.structure_epoch) {
        return make_status_v1(
            compatibility_code_v1::structure_epoch_mismatch, expected_stage,
            artifact.context.structure_epoch);
    }
    if (static_cast<std::uint8_t>(expected_stage) >=
            static_cast<std::uint8_t>(lowering_stage_v1::semantic_atom) &&
        !same_identity_v1(
            artifact.context.order_identity, expected.order_identity)) {
        return make_status_v1(
            compatibility_code_v1::order_mismatch, expected_stage);
    }
    return make_status_v1(
        compatibility_code_v1::compatible, expected_stage);
}

inline compatibility_status_v1 resume_from_atom_evidence_v1(
    const lowering_artifact_v1 &artifact,
    const lowering_identity_context_v1 &expected,
    resumption_cursor_v1 *cursor) noexcept {
    if (cursor == nullptr) {
        return make_status_v1(compatibility_code_v1::invalid_argument,
            lowering_stage_v1::atom_evidence);
    }
    const auto status = validate_artifact_for_stage_v1(
        artifact, lowering_stage_v1::atom_evidence, expected);
    if (!status) {
        *cursor = {};
        return status;
    }
    *cursor = {artifact.stage, artifact.artifact_identity, artifact.context,
        artifact.payload, artifact.payload_bytes};
    return status;
}

inline compatibility_status_v1 resume_from_semantic_atom_v1(
    const lowering_artifact_v1 &artifact,
    const lowering_identity_context_v1 &expected,
    resumption_cursor_v1 *cursor) noexcept {
    if (cursor == nullptr) {
        return make_status_v1(compatibility_code_v1::invalid_argument,
            lowering_stage_v1::semantic_atom);
    }
    const auto status = validate_artifact_for_stage_v1(
        artifact, lowering_stage_v1::semantic_atom, expected);
    if (!status) {
        *cursor = {};
        return status;
    }
    *cursor = {artifact.stage, artifact.artifact_identity, artifact.context,
        artifact.payload, artifact.payload_bytes};
    return status;
}

inline compatibility_status_v1 resume_from_target_cover_v1(
    const lowering_artifact_v1 &artifact,
    const lowering_identity_context_v1 &expected,
    stable_identity_v1 expected_cover_identity,
    resumption_cursor_v1 *cursor) noexcept {
    if (cursor == nullptr || !valid_identity_v1(expected_cover_identity)) {
        return make_status_v1(compatibility_code_v1::invalid_argument,
            lowering_stage_v1::target_cover);
    }
    const auto status = validate_artifact_for_stage_v1(
        artifact, lowering_stage_v1::target_cover, expected);
    if (!status) {
        *cursor = {};
        return status;
    }
    if (!valid_identity_v1(artifact.cover_identity) ||
        !same_identity_v1(
            artifact.cover_identity, expected_cover_identity)) {
        *cursor = {};
        return make_status_v1(compatibility_code_v1::identity_mismatch,
            lowering_stage_v1::target_cover);
    }
    *cursor = {artifact.stage, artifact.artifact_identity, artifact.context,
        artifact.payload, artifact.payload_bytes};
    return status;
}

static_assert(std::is_trivially_copyable_v<compatibility_status_v1>);
static_assert(std::is_trivially_copyable_v<lowering_identity_context_v1>);
static_assert(std::is_trivially_copyable_v<canonical_source_input_v1>);
static_assert(std::is_trivially_copyable_v<resumption_cursor_v1>);
static_assert(std::is_trivially_copyable_v<lowering_artifact_v1>);

}  // namespace cellerator::execution::lowering_resumption
