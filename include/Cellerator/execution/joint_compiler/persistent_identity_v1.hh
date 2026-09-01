#pragma once

#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::joint_compiler {

inline constexpr std::uint32_t persistent_identity_schema_version_v1 = 1u;

// The namespace identifies the producer or semantic ID authority. The local
// identity is meaningful only within that namespace. Neither field is a
// content digest, runtime address, device ordinal, or transient registry slot.
struct persistent_identity_v1 {
    std::uint64_t producer_namespace = 0u;
    std::uint64_t local_identity = 0u;
};

// Persistence uses an explicit envelope instead of native-struct bytes. The
// fields are encoded individually by artifact owners in their declared byte
// order; record_bytes permits deterministic adjacent-version rejection.
struct persistent_identity_record_v1 {
    std::uint32_t schema_version = persistent_identity_schema_version_v1;
    std::uint32_t record_bytes = sizeof(persistent_identity_record_v1);
    persistent_identity_v1 identity{};
};

enum class persistent_identity_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    missing_producer_namespace = 3u,
    missing_local_identity = 4u
};

struct persistent_identity_validation_result_v1 {
    persistent_identity_validation_code_v1 code =
        persistent_identity_validation_code_v1::ok;

    constexpr explicit operator bool() const noexcept {
        return code == persistent_identity_validation_code_v1::ok;
    }
};

constexpr bool same_persistent_identity_v1(
    persistent_identity_v1 lhs,
    persistent_identity_v1 rhs) noexcept {
    return lhs.producer_namespace == rhs.producer_namespace
        && lhs.local_identity == rhs.local_identity;
}

// operation_core_v2 stable IDs already carry two exact 64-bit fields. The
// bridge gives those fields namespace/local meaning without hashing, folding,
// or changing any bits.
constexpr persistent_identity_v1 from_operation_core_stable_id_v1(
    compute::operation::v2::stable_id identity) noexcept {
    return {identity.high, identity.low};
}

constexpr compute::operation::v2::stable_id to_operation_core_stable_id_v1(
    persistent_identity_v1 identity) noexcept {
    return {identity.local_identity, identity.producer_namespace};
}

// Legacy 64-bit IDs, including CellShard strong IDs, must arrive with an
// explicit producer namespace supplied by the caller. Consumers pass the
// strong ID's value; this standalone Cellerator header has no CellShard type
// dependency.
constexpr persistent_identity_v1 from_namespaced_local_identity_v1(
    std::uint64_t producer_namespace,
    std::uint64_t local_identity) noexcept {
    return {producer_namespace, local_identity};
}

persistent_identity_validation_result_v1 validate_persistent_identity_v1(
    persistent_identity_v1 identity) noexcept;

persistent_identity_validation_result_v1
validate_persistent_identity_record_v1(
    const persistent_identity_record_v1 &record) noexcept;

static_assert(sizeof(persistent_identity_v1) == 16u,
    "persistent identity v1 must remain one namespace and one local u64");
static_assert(sizeof(persistent_identity_record_v1) == 24u,
    "persistent identity record v1 layout is part of the bridge ABI");
static_assert(std::is_standard_layout_v<persistent_identity_v1>);
static_assert(std::is_trivially_copyable_v<persistent_identity_v1>);
static_assert(std::is_standard_layout_v<persistent_identity_record_v1>);
static_assert(std::is_trivially_copyable_v<persistent_identity_record_v1>);

}  // namespace cellerator::execution::joint_compiler
