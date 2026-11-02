#include <Cellerator/geometry/dependency/cartesian_access.hh>

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>

namespace dep = cellerator::geometry::dependency;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

dep::SourceDescriptor source(std::uint64_t id, std::uint64_t extent) {
    return {{{id, 1}, {id + 10, 1}, {id + 20, 1}},
            {{id + 10, 1}, {id + 30, 1}, extent}};
}

dep::SelectionDescriptor all(std::uint64_t id,
                             const dep::SourceDescriptor& value) {
    return {{{id, 1}, value.axis.domain, dep::SelectionSpace::coordinates},
            value.axis.logical_order, value.axis.extent};
}

dep::SelectionDescriptor range(std::uint64_t id,
                               const dep::SourceDescriptor& value,
                               std::uint64_t begin,
                               std::uint64_t count) {
    auto result = all(id, value);
    result.kind = dep::SelectionKind::range;
    result.begin = begin;
    result.count = count;
    return result;
}

dep::SelectionDescriptor explicit_selection(
    std::uint64_t id, const dep::SourceDescriptor& value,
    std::span<const std::uint64_t> coordinates) {
    auto result = all(id, value);
    result.kind = dep::SelectionKind::explicit_coordinates;
    result.coordinates = coordinates;
    return result;
}

bool matches(const dep::CoordinateParticipant& actual,
             const dep::RoleSelection& expected,
             std::uint64_t coordinate) {
    return cellerator::structure::same_identity(actual.role, expected.role)
        && dep::same_source(actual.source, expected.source.reference)
        && cellerator::structure::same_identity(actual.order,
                                         expected.source.axis.logical_order)
        && actual.coordinate == coordinate
        && actual.incidence_slot == expected.incidence_slot;
}

bool same_participant(const dep::CoordinateParticipant& a,
                      const dep::CoordinateParticipant& b) {
    return cellerator::structure::same_identity(a.role, b.role)
        && dep::same_source(a.source, b.source)
        && cellerator::structure::same_identity(a.order, b.order)
        && a.coordinate == b.coordinate
        && a.incidence_slot == b.incidence_slot;
}

template<std::size_t N>
bool same_participants(const std::array<dep::CoordinateParticipant, N>& a,
                       const std::array<dep::CoordinateParticipant, N>& b) {
    for (std::size_t i = 0; i < N; ++i)
        if (!same_participant(a[i], b[i])) return false;
    return true;
}

void implicit_huge_product_is_bounded() {
    const auto x = source(1, 1000000000ULL);
    const auto y = source(2, 1000000000ULL);
    std::array<dep::RoleSelection, 2> roles{{
        {{11, 1}, 0, x, all(21, x)},
        {{12, 1}, 1, y, all(22, y)}}};
    const dep::CartesianPattern pattern{{31, 1}, roles};

    // The implicit family contains 10^18 candidates, but only two tuples are
    // ever materialized at either end of that addressable range.
    std::array<dep::CoordinateParticipant, 4> tuples{};
    std::uint64_t written = 77;
    require(bool(dep::cartesian_batch(pattern, 0, 2, tuples, written)),
            "decode first small batch of implicit huge product");
    require(written == 2 && matches(tuples[0], roles[0], 0)
        && matches(tuples[1], roles[1], 0)
        && matches(tuples[2], roles[0], 0)
        && matches(tuples[3], roles[1], 1),
        "first batch is row-major with last role varying fastest");

    constexpr std::uint64_t total = 1000000000000000000ULL;
    written = 0;
    require(bool(dep::cartesian_batch(pattern, total - 2, 2, tuples, written)),
            "decode last small batch of implicit huge product");
    require(written == 2
        && matches(tuples[0], roles[0], 999999999ULL)
        && matches(tuples[1], roles[1], 999999998ULL)
        && matches(tuples[2], roles[0], 999999999ULL)
        && matches(tuples[3], roles[1], 999999999ULL),
        "last batch decodes from ordinal without enumerating its prefix");

    std::array<dep::CoordinateParticipant, 2> one{};
    require(bool(dep::cartesian_tuple(pattern, total - 1, one)),
            "decode the final tuple directly");
    require(matches(one[0], roles[0], 999999999ULL)
        && matches(one[1], roles[1], 999999999ULL),
        "direct tuple ordinal agrees with batch decoder");
}

void selection_coordinate_mapping() {
    const auto x = source(3, 5);
    const auto y = source(4, 6);
    const auto z = source(5, 5);
    const std::array<std::uint64_t, 2> chosen{0, 4};
    std::array<dep::RoleSelection, 3> roles{{
        {{13, 1}, 0, x, all(23, x)},
        {{14, 1}, 1, y, range(24, y, 2, 2)},
        {{15, 1}, 2, z, explicit_selection(25, z, chosen)}}};
    const dep::CartesianPattern pattern{{32, 1}, roles};

    std::array<dep::CoordinateParticipant, 6> output{};
    std::uint64_t written = 0;
    require(bool(dep::cartesian_batch(pattern, 0, 2, output, written)),
            "decode a small product with mixed selection kinds");
    require(written == 2
        && matches(output[0], roles[0], 0)
        && matches(output[1], roles[1], 2)
        && matches(output[2], roles[2], 0)
        && matches(output[3], roles[0], 0)
        && matches(output[4], roles[1], 2)
        && matches(output[5], roles[2], 4),
        "all/range/list selections map positions to their logical coordinates");
}

void empty_and_rank_zero_products() {
    const auto x = source(6, 0);
    std::array<dep::RoleSelection, 1> empty_role{{
        {{16, 1}, 0, x, all(26, x)}}};
    const dep::CartesianPattern empty{{33, 1}, empty_role};
    std::uint64_t written = 99;
    require(bool(dep::cartesian_batch(empty, 0, 0, {}, written))
        && written == 0,
        "empty product accepts an empty request at its end");
    std::array<dep::CoordinateParticipant, 1> sentinel{{
        {{99, 1}, {{98, 1}, {97, 1}, {96, 1}}, {95, 1}, 94, 93}}};
    const auto prior = sentinel;
    written = 88;
    require(dep::cartesian_tuple(empty, 0, sentinel).code
            == dep::StatusCode::invalid_pattern
        && same_participants(sentinel, prior),
        "zero-extent role rejects tuple access without writing output");
    require(dep::cartesian_batch(empty, 1, 0, {}, written).code
            == dep::StatusCode::invalid_pattern
        && written == 88,
        "begin beyond empty product fails without publishing a count");

    const dep::CartesianPattern scalar{{34, 1}, {}};
    require(bool(dep::cartesian_tuple(scalar, 0, {})),
            "rank-zero Cartesian product has one empty tuple");
    require(dep::cartesian_tuple(scalar, 1, {}).code
            == dep::StatusCode::invalid_pattern,
        "rank-zero product has no second tuple");
    written = 77;
    require(bool(dep::cartesian_batch(scalar, 0, 1, {}, written))
        && written == 1,
        "rank-zero batch reports its one empty tuple");
    written = 76;
    require(bool(dep::cartesian_batch(scalar, 1, 0, {}, written))
        && written == 0,
        "rank-zero product permits an empty request at its end");
}

void failures_preserve_outputs() {
    const auto x = source(7, 5);
    const auto y = source(8, 4);
    std::array<dep::RoleSelection, 2> roles{{
        {{17, 1}, 0, x, all(27, x)},
        {{18, 1}, 1, y, all(28, y)}}};
    const dep::CartesianPattern pattern{{35, 1}, roles};
    std::array<dep::CoordinateParticipant, 4> output{{
        {{91, 1}, {{92, 1}, {93, 1}, {94, 1}}, {95, 1}, 96, 97},
        {{81, 1}, {{82, 1}, {83, 1}, {84, 1}}, {85, 1}, 86, 87},
        {{71, 1}, {{72, 1}, {73, 1}, {74, 1}}, {75, 1}, 76, 77},
        {{61, 1}, {{62, 1}, {63, 1}, {64, 1}}, {65, 1}, 66, 67}}};
    const auto original = output;
    std::uint64_t written = 55;
    require(dep::cartesian_batch(pattern, 0, 3,
                std::span<dep::CoordinateParticipant>(output.data(), 5), written).code
            == dep::StatusCode::insufficient_capacity,
        "undersized batch buffer is reported distinctly");
    require(same_participants(output, original) && written == 55,
        "undersized batch leaves all output and count state untouched");
    require(dep::cartesian_tuple(pattern, 0,
                std::span<dep::CoordinateParticipant>(output.data(), 1)).code
            == dep::StatusCode::insufficient_capacity,
        "undersized tuple buffer is reported distinctly");
    require(same_participants(output, original),
        "undersized tuple buffer is not partially populated");
    require(dep::cartesian_batch(pattern, 19, 2, output, written).code
            == dep::StatusCode::invalid_pattern
        && same_participants(output, original) && written == 55,
        "batch that extends past family end fails transactionally");

    roles[1].selection.order.low++;
    require(dep::cartesian_batch(pattern, 0, 1, output, written).code
            == dep::StatusCode::order_mismatch
        && same_participants(output, original) && written == 55,
        "same-size wrong-order role association rejects before writing");
}

} // namespace

int main() {
    try {
        implicit_huge_product_is_bounded();
        selection_coordinate_mapping();
        empty_and_rank_zero_products();
        failures_preserve_outputs();
        std::cout << "Cartesian access tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Cartesian access test failed: " << error.what() << '\n';
        return 1;
    }
}
