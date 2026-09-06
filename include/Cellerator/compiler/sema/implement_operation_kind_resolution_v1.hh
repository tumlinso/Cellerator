#pragma once

#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <optional>
#include <variant>

namespace cellerator::compiler::sema::v1 {

enum class source_operation_kind : std::uint8_t {
    relation_apply = 1,
    relation_transpose,
    support_contraction,
    segment_statistics,
    normalization,
    edge_map_or_gate,
    sparse_update,
    relation_bundle,
    relation_chain,
    moments,
    hierarchy,
    exchange,
    gradient,
    publication
};

enum class composition_kind : std::uint8_t {
    relation_chain, moments, hierarchy, exchange, gradient
};

enum class effect_kind : std::uint8_t { publication };

using operation_meaning = std::variant<compute::operation::v2::operation_kind,
    composition_kind, effect_kind>;

struct operation_kind_resolution {
    source_operation_kind source{};
    const char *syntax = nullptr;
    operation_meaning meaning;

    // Compositions/effects contain no representative executable opcode.
    std::optional<compute::operation::v2::operation_kind> primitive() const noexcept {
        const auto *kind = std::get_if<compute::operation::v2::operation_kind>(
            &meaning);
        return kind ? std::optional{*kind} : std::nullopt;
    }
};

const operation_kind_resolution *operation_kind_coverage_table() noexcept;
std::uint32_t operation_kind_coverage_count() noexcept;
const operation_kind_resolution *resolve_operation_kind(
    source_operation_kind kind) noexcept;

}  // namespace cellerator::compiler::sema::v1
