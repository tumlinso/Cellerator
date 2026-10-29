#include <Cellerator/geometry/rectangular_support.hh>

#include <cassert>

namespace {

namespace geo = cellerator::geometry;
namespace ex = cellerator::execution;

constexpr ex::axis_identity make_axis(std::uint32_t seed) noexcept {
    return {
        {seed + 1u, 1u},
        {seed + 2u, 1u},
        {seed + 3u, 1u},
        {seed + 4u, 1u}
    };
}

geo::relation_cover_view_v1 make_cover(
    const geo::semantic_component_v1 *components) noexcept {
    geo::relation_cover_view_v1 cover{};
    cover.structure = {1u, 1u};
    cover.structure_epoch = {3u};
    cover.source_axis = make_axis(10u);
    cover.destination_axis = make_axis(20u);
    cover.logical_edge_count = 5u;
    cover.component_count = 2u;
    cover.components = components;
    return cover;
}

geo::rectangular_support_view_v1 make_support(
    const geo::relation_cover_view_v1 &cover,
    const geo::rectangular_component_membership_v1 *memberships,
    const std::uint32_t *source_members,
    const std::uint32_t *destination_members,
    const geo::portable_support_reference_v1 *references) noexcept {
    geo::rectangular_support_view_v1 support{};
    support.source_axis = cover.source_axis;
    support.destination_axis = cover.destination_axis;
    support.memberships = memberships;
    support.membership_count = 1u;
    support.source_members = source_members;
    support.source_member_count = 3u;
    support.destination_members = destination_members;
    support.destination_member_count = 2u;
    support.support_references = references;
    support.support_reference_count = 1u;
    return support;
}

void test_hardware_neutral_two_axis_membership() {
    const geo::semantic_component_v1 components[] = {
        {7u, geo::semantic_component_kind::rectangular, {}, 0u, 4u},
        {8u, geo::semantic_component_kind::unstructured, {}, 4u, 1u}
    };
    const geo::relation_cover_view_v1 cover = make_cover(components);
    const geo::rectangular_component_membership_v1 memberships[] = {
        {7u, 0u, 0u, 3u, 0u, 2u, 0u, 1u}
    };
    const std::uint32_t source_members[] = {9u, 2u, 14u};
    const std::uint32_t destination_members[] = {6u, 1u};
    const geo::portable_support_reference_v1 references[] = {
        {0xabcdefu, 5u, 0u, 12u, 3u}
    };
    const auto support = make_support(cover, memberships, source_members,
        destination_members, references);
    assert(geo::validate_rectangular_support(cover, support));
}

void test_membership_must_name_rectangular_component_once() {
    const geo::semantic_component_v1 components[] = {
        {7u, geo::semantic_component_kind::rectangular, {}, 0u, 4u},
        {8u, geo::semantic_component_kind::unstructured, {}, 4u, 1u}
    };
    const geo::relation_cover_view_v1 cover = make_cover(components);
    geo::rectangular_component_membership_v1 memberships[] = {
        {8u, 0u, 0u, 3u, 0u, 2u, 0u, 1u},
        {8u, 0u, 0u, 3u, 0u, 2u, 0u, 1u}
    };
    const std::uint32_t source_members[] = {9u, 2u, 14u};
    const std::uint32_t destination_members[] = {6u, 1u};
    const geo::portable_support_reference_v1 references[] = {
        {0xabcdefu, 5u, 0u, 12u, 3u}
    };
    auto support = make_support(cover, memberships, source_members,
        destination_members, references);
    assert(geo::validate_rectangular_support(cover, support).code
        == geo::rectangular_support_validation_code::invalid_component);

    memberships[0].component_id = 7u;
    memberships[1].component_id = 7u;
    support.membership_count = 2u;
    assert(geo::validate_rectangular_support(cover, support).code
        == geo::rectangular_support_validation_code::duplicate_component);
}

void test_membership_spans_and_values_are_exact() {
    const geo::semantic_component_v1 components[] = {
        {7u, geo::semantic_component_kind::rectangular, {}, 0u, 4u},
        {8u, geo::semantic_component_kind::unstructured, {}, 4u, 1u}
    };
    const geo::relation_cover_view_v1 cover = make_cover(components);
    geo::rectangular_component_membership_v1 membership =
        {7u, 0u, 0u, 3u, 0u, 2u, 0u, 1u};
    std::uint32_t source_members[] = {9u, 2u, 14u};
    const std::uint32_t destination_members[] = {6u, 1u};
    geo::portable_support_reference_v1 reference =
        {0xabcdefu, 5u, 0u, 12u, 3u};
    auto support = make_support(cover, &membership, source_members,
        destination_members, &reference);

    membership.source_member_count = 4u;
    assert(geo::validate_rectangular_support(cover, support).code
        == geo::rectangular_support_validation_code::member_span_out_of_bounds);

    membership.source_member_count = 3u;
    source_members[2] = 9u;
    assert(geo::validate_rectangular_support(cover, support).code
        == geo::rectangular_support_validation_code::duplicate_axis_member);

    source_members[2] = 14u;
    reference.evidence_identity = 0u;
    assert(geo::validate_rectangular_support(cover, support).code
        == geo::rectangular_support_validation_code::invalid_support_reference);
}

void test_axis_and_rectangular_coverage_are_mandatory() {
    const geo::semantic_component_v1 components[] = {
        {7u, geo::semantic_component_kind::rectangular, {}, 0u, 4u},
        {8u, geo::semantic_component_kind::unstructured, {}, 4u, 1u}
    };
    const geo::relation_cover_view_v1 cover = make_cover(components);
    geo::rectangular_support_view_v1 support{};
    support.source_axis = cover.source_axis;
    support.destination_axis = cover.destination_axis;
    assert(geo::validate_rectangular_support(cover, support).code
        == geo::rectangular_support_validation_code::missing_rectangular_component);

    support.destination_axis.order.generation += 1u;
    assert(geo::validate_rectangular_support(cover, support).code
        == geo::rectangular_support_validation_code::axis_mismatch);
}

} // namespace

int main() {
    test_hardware_neutral_two_axis_membership();
    test_membership_must_name_rectangular_component_once();
    test_membership_spans_and_values_are_exact();
    test_axis_and_rectangular_coverage_are_mandatory();
    return 0;
}
