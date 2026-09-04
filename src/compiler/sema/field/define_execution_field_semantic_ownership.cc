#include <Cellerator/compiler/sema/field/define_execution_field_semantic_ownership_v1.hh>

#include <algorithm>
#include <string_view>
#include <utility>

namespace Cellerator::compiler::sema::field {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;

void hash_bytes(std::uint64_t& hash, std::string_view bytes) noexcept {
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= fnv_prime;
    }
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift != 64; shift += 8) {
        hash ^= static_cast<unsigned char>(value >> shift);
        hash *= fnv_prime;
    }
}

execution_field_identity_v1 source_identity(
    const execution_field_definition_v1& definition) noexcept {
    std::uint64_t low = fnv_offset;
    std::uint64_t high = fnv_offset ^ 0x9e3779b97f4a7c15ull;
    hash_bytes(low, definition.stable_source_name);
    hash_bytes(high, definition.explicit_field_name);
    hash_u64(low, definition.source.begin.space);
    hash_u64(low, definition.source.begin.byte_offset);
    hash_u64(high, definition.source.end.space);
    hash_u64(high, definition.source.end.byte_offset);
    // A valid source identity must never use the all-zero sentinel.
    if (low == 0 && high == 0) high = 1;
    return {low, high};
}

bool contains(frontend::source::source_span_v1 outer,
              frontend::source::source_span_v1 inner) noexcept {
    return outer.valid() && inner.valid() &&
        outer.begin.space == inner.begin.space &&
        outer.begin.byte_offset <= inner.begin.byte_offset &&
        inner.end.byte_offset <= outer.end.byte_offset;
}

}  // namespace

execution_field_definition_status_v1 define_execution_field_semantic_ownership_v1(
    const execution_field_definition_v1& definition,
    execution_field_semantics_v1* semantics) noexcept {
    if (semantics == nullptr || !definition.source.valid()) {
        return execution_field_definition_status_v1::invalid_source;
    }
    if (definition.schema_version != execution_field_semantics_schema_version_v1) {
        return execution_field_definition_status_v1::schema_mismatch;
    }
    if (definition.stable_source_name.empty()) {
        return execution_field_definition_status_v1::missing_source_identity;
    }
    for (std::size_t index = 0; index < definition.captured_values.size(); ++index) {
        const auto& capture = definition.captured_values[index];
        if (capture.canonical_name.empty() || capture.declaration_identity == 0) {
            return execution_field_definition_status_v1::invalid_capture;
        }
        if (std::find_if(definition.captured_values.begin(),
                         definition.captured_values.begin() + index,
                         [&capture](const captured_value_v1& prior) {
                             return prior.declaration_identity == capture.declaration_identity;
                         }) != definition.captured_values.begin() + index) {
            return execution_field_definition_status_v1::duplicate_capture;
        }
    }
    for (const auto& boundary : definition.observable_boundaries) {
        if (!contains(definition.source, boundary.source)) {
            return execution_field_definition_status_v1::boundary_outside_field;
        }
    }

    execution_field_semantics_v1 result;
    result.identity = source_identity(definition);
    result.stable_source_name = definition.stable_source_name;
    result.explicit_field_name = definition.explicit_field_name;
    result.source = definition.source;
    result.captured_values = definition.captured_values;
    result.observable_boundaries = definition.observable_boundaries;
    result.profile_environment = definition.profile_environment;
    result.semantic_effects = definition.semantic_effects;
    for (const auto& boundary : result.observable_boundaries) {
        result.semantic_effects |= boundary.effects;
    }
    *semantics = std::move(result);
    return execution_field_definition_status_v1::success;
}

bool execution_field_owns_operation_v1(
    const execution_field_semantics_v1& field,
    frontend::source::source_span_v1 operation) noexcept {
    return (field.identity.low != 0 || field.identity.high != 0) &&
        contains(field.source, operation);
}

}  // namespace Cellerator::compiler::sema::field
