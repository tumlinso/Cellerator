#include <Cellerator/compute/candidate/segment/mechanism_v2.hh>

#include <array>

int main() {
    using namespace cellerator::compute::segment;
    const auto portfolio = built_in_segment_mechanisms_v2();
    if (!validate_segment_mechanism_portfolio_v2(portfolio)) return 1;
    for (const auto mechanism : std::array<segment_mechanism_v2, 3>{{
            segment_mechanism_v2::warp_per_output,
            segment_mechanism_v2::cta_per_output,
            segment_mechanism_v2::large_segment_cta}}) {
        const auto *entry = find_segment_mechanism_v2(portfolio, mechanism);
        if (entry == nullptr || !entry->requires_measurement
            || entry->production_promoted)
            return 2;
    }
    auto duplicate = std::array<segment_mechanism_descriptor_v2, 3>{{
        portfolio.mechanisms[0], portfolio.mechanisms[1],
        portfolio.mechanisms[2]}};
    duplicate[2].mechanism_identity = duplicate[0].mechanism_identity;
    if (validate_segment_mechanism_portfolio_v2(
            {duplicate.data(), static_cast<std::uint32_t>(duplicate.size()), 0u}))
        return 3;
    return 0;
}
