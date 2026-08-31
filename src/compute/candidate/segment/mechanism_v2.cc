#include <Cellerator/compute/candidate/segment/mechanism_v2.hh>

#include <array>
#include <cstdint>

namespace cellerator::compute::segment {
namespace {

constexpr std::uint64_t warp_identity = 0x7365672d77617270ULL;
constexpr std::uint64_t cta_identity = 0x7365672d63746131ULL;
constexpr std::uint64_t large_identity = 0x7365672d6c617267ULL;

constexpr segment_mechanism_descriptor_v2 warp_descriptor{
    segment_mechanism_schema_version_v2,
    segment_mechanism_v2::warp_per_output,
    {},
    warp_identity,
    "cellerator_segment_v2_warp_per_output",
    32u,
    1u,
    32u,
    1u,
    0u,
    true,
    true,
    false,
    {}};

constexpr segment_mechanism_descriptor_v2 cta_descriptor{
    segment_mechanism_schema_version_v2,
    segment_mechanism_v2::cta_per_output,
    {},
    cta_identity,
    "cellerator_segment_v2_cta_per_output",
    256u,
    8u,
    4096u,
    1u,
    0u,
    true,
    true,
    false,
    {}};

constexpr segment_mechanism_descriptor_v2 large_descriptor{
    segment_mechanism_schema_version_v2,
    segment_mechanism_v2::large_segment_cta,
    {},
    large_identity,
    "cellerator_segment_v2_large_segment_cta",
    512u,
    16u,
    0xffffffffffffffffULL,
    1u,
    0u,
    true,
    true,
    false,
    {}};

constexpr std::array<segment_mechanism_descriptor_v2, 3> mechanisms{{
    warp_descriptor, cta_descriptor, large_descriptor}};

segment_result_v2 error(segment_status_v2 code, const char *message) noexcept {
    return {code, message};
}

bool terminated_name(const char *name) noexcept {
    for (std::uint32_t index = 0u;
         index < segment_mechanism_name_capacity_v2; ++index)
        if (name[index] == '\0') return index != 0u;
    return false;
}

bool same_name(const char *left, const char *right) noexcept {
    for (std::uint32_t index = 0u;
         index < segment_mechanism_name_capacity_v2; ++index) {
        if (left[index] != right[index]) return false;
        if (left[index] == '\0') return true;
    }
    return true;
}

} // namespace

segment_mechanism_portfolio_view_v2 built_in_segment_mechanisms_v2() noexcept {
    return {mechanisms.data(), static_cast<std::uint32_t>(mechanisms.size()), 0u};
}

segment_result_v2 validate_segment_mechanism_portfolio_v2(
    segment_mechanism_portfolio_view_v2 portfolio) noexcept {
    if (portfolio.mechanisms == nullptr || portfolio.count != mechanisms.size()
        || portfolio.reserved != 0u)
        return error(segment_status_v2::invalid_argument,
            "segment mechanism portfolio must contain exactly three entries");
    bool seen_warp = false;
    bool seen_cta = false;
    bool seen_large = false;
    for (std::uint32_t index = 0u; index < portfolio.count; ++index) {
        const auto &entry = portfolio.mechanisms[index];
        if (entry.schema_version != segment_mechanism_schema_version_v2
            || entry.mechanism_identity == 0u
            || !terminated_name(entry.static_name)
            || entry.threads_per_cta == 0u
            || entry.threads_per_cta % 32u != 0u
            || entry.warps_per_cta != entry.threads_per_cta / 32u
            || entry.launches_per_component == 0u
            || !entry.graph_capture_compatible
            || !entry.requires_measurement
            || entry.production_promoted)
            return error(segment_status_v2::invalid_argument,
                "segment mechanism descriptor is invalid or promoted");
        for (std::uint32_t previous = 0u; previous < index; ++previous)
            if (portfolio.mechanisms[previous].mechanism_identity
                    == entry.mechanism_identity
                || same_name(portfolio.mechanisms[previous].static_name,
                    entry.static_name))
                return error(segment_status_v2::invalid_identity,
                    "segment mechanism identity or name is duplicated");
        seen_warp |= entry.mechanism == segment_mechanism_v2::warp_per_output;
        seen_cta |= entry.mechanism == segment_mechanism_v2::cta_per_output;
        seen_large |= entry.mechanism == segment_mechanism_v2::large_segment_cta;
    }
    return seen_warp && seen_cta && seen_large ? segment_result_v2{}
        : error(segment_status_v2::invalid_argument,
            "segment mechanism portfolio is incomplete");
}

const segment_mechanism_descriptor_v2 *find_segment_mechanism_v2(
    segment_mechanism_portfolio_view_v2 portfolio,
    segment_mechanism_v2 mechanism) noexcept {
    if (!validate_segment_mechanism_portfolio_v2(portfolio)) return nullptr;
    for (std::uint32_t index = 0u; index < portfolio.count; ++index)
        if (portfolio.mechanisms[index].mechanism == mechanism)
            return &portfolio.mechanisms[index];
    return nullptr;
}

} // namespace cellerator::compute::segment
