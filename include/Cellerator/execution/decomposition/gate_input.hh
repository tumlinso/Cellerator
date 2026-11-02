#pragma once

#include <Cellerator/compute/decomposition/vocabulary_v1.hh>
#include <Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::decomposition {

inline constexpr std::uint32_t gate_input_schema_version_v1 = 1u;

enum class GateInputReplicationV1 : std::uint8_t {
    partition_local = 1u,
    replicated_read_only = 2u,
    halo_read_only = 3u,
    producer_routed = 4u
};

struct GateInputV1 {
    std::uint64_t operand_index =
        ::cellerator::compute::operation::v2::invalid_binding_index;
    ::cellerator::compute::operation::v2::gate_indexing dependency =
        ::cellerator::compute::operation::v2::gate_indexing::none;
    ::cellerator::compute::decomposition::split_axis_kind_v1 split_axis =
        ::cellerator::compute::decomposition::split_axis_kind_v1::none;
    GateInputReplicationV1 replication =
        GateInputReplicationV1::partition_local;
    bool read_only = true;
    std::uint32_t replica_or_halo_count = 1u;
};

struct GateInputSetV1 {
    std::uint32_t schema_version = gate_input_schema_version_v1;
    std::uint32_t reserved = 0u;
    ::cellerator::compute::operation::v2::stable_id identity{};
    const ::cellerator::compute::operation::v2::relation_algebra_problem *
        problem = nullptr;
    const GateInputV1 *inputs = nullptr;
    std::uint64_t input_count = 0u;
};

enum class GateInputValidationCodeV1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    missing_inputs,
    invalid_input_count,
    invalid_operand,
    operand_order_mismatch,
    invalid_dependency,
    dependency_mismatch,
    invalid_split_axis,
    invalid_replication,
    mutable_replica,
    invalid_replica_count,
    missing_factorized_dependency
};

struct GateInputValidationResultV1 {
    GateInputValidationCodeV1 code = GateInputValidationCodeV1::ok;
    std::uint64_t input_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == GateInputValidationCodeV1::ok;
    }
};

GateInputValidationResultV1 validate_gate_input_set_v1(
    const GateInputSetV1 &set) noexcept;

static_assert(std::is_trivially_copyable_v<GateInputV1>);
static_assert(std::is_standard_layout_v<GateInputV1>);
static_assert(std::is_trivially_copyable_v<GateInputSetV1>);
static_assert(std::is_standard_layout_v<GateInputSetV1>);
static_assert(std::is_trivially_copyable_v<GateInputValidationResultV1>);

}  // namespace cellerator::execution::decomposition
