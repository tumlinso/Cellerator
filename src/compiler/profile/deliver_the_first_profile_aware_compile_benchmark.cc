#include <Cellerator/compiler/profile/deliver_the_first_profile_aware_compile_benchmark_v1.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace cellerator::compiler::profile::v1 {
namespace {

std::uint64_t mix(std::uint64_t hash, std::uint64_t value) noexcept {
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31u;
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    return hash;
}

std::uint64_t bits(double value) noexcept {
    std::uint64_t result = 0u;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

bool same(profile_identity_v1 left, profile_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

bool finite_nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

}  // namespace

profile_aware_compile_status_v1 compile_profile_aware_relation_v1(
    const relation_field_source_v1 &source,
    const profile_compile_state_v1 &profile,
    profile_aware_compile_result_v1 *result) noexcept {
    if (result == nullptr || source.source_semantics_fingerprint == 0u ||
        source.operation_count == 0u) {
        return profile_aware_compile_status_v1::invalid_argument;
    }
    if (profile.contract_version != profile_environment_contract_version_v1) {
        return profile_aware_compile_status_v1::unsupported_contract;
    }
    if (!same(source.relation, profile.structure.relation) ||
        !same(source.source_domain, profile.structure.source_axis.domain) ||
        !same(source.destination_domain, profile.structure.destination_axis.domain)) {
        return profile_aware_compile_status_v1::identity_mismatch;
    }
    if (profile.structure.destination_axis.extent == 0u ||
        profile.structure.support_count == 0u ||
        !finite_nonnegative(profile.values.approximation_risk) ||
        !finite_nonnegative(profile.reuse.reuse_horizon)) {
        return profile_aware_compile_status_v1::invalid_evidence;
    }

    const auto load_begin = std::chrono::steady_clock::now();
    // Copying is intentional: it models loading the bounded, immutable state
    // into compiler-owned cold data without retaining runtime pointers.
    const profile_compile_state_v1 loaded = profile;
    const auto load_end = std::chrono::steady_clock::now();

    const auto propagation_begin = std::chrono::steady_clock::now();
    profile_candidate_search_inputs_v1 search{};
    search.support_count = loaded.structure.support_count;
    search.destination_extent = loaded.structure.destination_axis.extent;
    search.expected_density = static_cast<double>(search.support_count) /
                              static_cast<double>(search.destination_extent);
    search.expected_reuse = loaded.reuse.reuse_horizon;
    search.approximation_risk = loaded.values.approximation_risk;

    const double mean_degree = loaded.structure.degree.mean;
    search.preferred_rows_per_group = mean_degree >= 32.0 ? 32u :
                                      mean_degree >= 8.0 ? 16u : 8u;
    search.candidate_flags = profile_candidate_direct_sparse_v1;
    if (loaded.structure.ordering_stability >= 0.75) {
        search.candidate_flags |= profile_candidate_row_grouped_v1;
    }
    if (loaded.values.zero_count * 2u >= loaded.values.observation_count ||
        loaded.values.approximation_risk <= 0.05) {
        search.candidate_flags |= profile_candidate_value_specialized_v1;
    }
    if (loaded.reuse.reuse_horizon >= 4.0 &&
        loaded.reuse.structure_change.upper_95 <= 0.25) {
        search.candidate_flags |= profile_candidate_cached_projection_v1;
    }

    std::uint64_t profile_hash = mix(0x43454c5052463031ull, loaded.state.low);
    profile_hash = mix(profile_hash, loaded.state.high);
    profile_hash = mix(profile_hash, loaded.structure.evidence.low);
    profile_hash = mix(profile_hash, loaded.structure.evidence.high);
    profile_hash = mix(profile_hash, loaded.values.evidence.low);
    profile_hash = mix(profile_hash, loaded.reuse.evidence.low);
    profile_hash = mix(profile_hash, loaded.structure.support_count);
    profile_hash = mix(profile_hash, bits(loaded.reuse.reuse_horizon));

    std::uint64_t search_hash = mix(0x5345415243483031ull, search.support_count);
    search_hash = mix(search_hash, search.destination_extent);
    search_hash = mix(search_hash, search.preferred_rows_per_group);
    search_hash = mix(search_hash, search.candidate_flags);
    search_hash = mix(search_hash, bits(search.expected_density));
    search_hash = mix(search_hash, bits(search.expected_reuse));
    search_hash = mix(search_hash, bits(search.approximation_risk));
    const auto propagation_end = std::chrono::steady_clock::now();

    result->state = loaded.state;
    result->search = search;
    result->source_semantics_fingerprint = source.source_semantics_fingerprint;
    result->profile_fingerprint = profile_hash;
    result->search_fingerprint = search_hash;
    result->profile_load_nanoseconds = std::max<std::uint64_t>(
        1u, std::chrono::duration_cast<std::chrono::nanoseconds>(load_end - load_begin).count());
    result->propagation_nanoseconds = std::max<std::uint64_t>(
        1u, std::chrono::duration_cast<std::chrono::nanoseconds>(
                propagation_end - propagation_begin).count());
    result->compiler_memory_bytes = sizeof(loaded) + sizeof(search);
    result->candidate_count = 1u;
    for (std::uint32_t mask = search.candidate_flags; mask != 0u; mask &= mask - 1u) {
        ++result->candidate_count;
    }
    return profile_aware_compile_status_v1::ok;
}

}  // namespace cellerator::compiler::profile::v1
