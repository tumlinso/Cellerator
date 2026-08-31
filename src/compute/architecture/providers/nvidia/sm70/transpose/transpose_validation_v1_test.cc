#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_validation_v1.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_value_map_v1.hh>

#include <cstdlib>
#include <vector>

using namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose;

namespace { void require(bool value) { if (!value) std::abort(); } }

int main() {
    constexpr std::uint64_t edge_count = 131071u;
    constexpr std::uint64_t owner_count = 4096u;
    constexpr std::uint64_t destination_count = 257u;
    constexpr std::uint32_t width = 7u;
    constexpr std::uint64_t high = (std::uint64_t{1u} << 32u) + 100u;
    std::vector<global_relation_edge_v1> edges(edge_count);
    std::vector<std::uint64_t> source_order(edge_count);
    std::vector<std::uint64_t> identity_order(edge_count);
    std::uint64_t position = 0u;
    for (std::uint64_t owner = 0u; owner < owner_count; ++owner) {
        const std::uint64_t begin = edge_count * owner / owner_count;
        const std::uint64_t end = edge_count * (owner + 1u) / owner_count;
        for (std::uint64_t local = 0u; local < end - begin; ++local) {
            const std::uint64_t destination = local % destination_count;
            edges[position] = {position + 1u, high + owner + 1u,
                high + destination_count - destination};
            source_order[position] = position;
            identity_order[position] = position;
            ++position;
        }
    }
    // Each owner must be destination-major. Reverse the per-owner destination
    // construction above by sorting only fixture indices during setup; this is
    // validation data preparation, never the production cover path.
    for (std::uint64_t owner = 0u; owner < owner_count; ++owner) {
        const std::uint64_t begin = edge_count * owner / owner_count;
        const std::uint64_t end = edge_count * (owner + 1u) / owner_count;
        for (std::uint64_t left = begin, right = end - 1u; left < right;
            ++left, --right) {
            const global_relation_edge_v1 temporary = edges[left];
            edges[left] = edges[right];
            edges[right] = temporary;
        }
    }
    // Identities moved with the edge records; construct the identity order in
    // one pass from the known contiguous owner partition.
    for (std::uint64_t index = 0u; index < edge_count; ++index)
        identity_order[edges[index].logical_edge_id - 1u] = index;

    transpose_cover_input_v1 input{edges.data(), source_order.data(),
        identity_order.data(), edge_count, 0x111u, 0x222u};
    transpose_cover_requirements_v1 requirements{};
    require(query_transpose_cover_requirements_v1(input, &requirements)
        == transpose_status_v1::success);
    require(requirements.owner_count == owner_count);
    std::vector<transpose_edge_placement_v1> placements(edge_count);
    std::vector<source_owner_schedule_v1> owners(owner_count);
    transpose_cover_view_v1 cover{};
    require(build_transpose_cover_v1(input,
        {placements.data(), placements.size(), owners.data(), owners.size()},
        &cover) == transpose_status_v1::success);

    std::vector<std::uint64_t> destination_ids(destination_count);
    for (std::uint64_t index = 0u; index < destination_count; ++index)
        destination_ids[index] = high + index + 1u;
    std::vector<transpose_edge_placement_v1> bound(edge_count);
    std::vector<projection_gradient_position_v1> gradient_positions(edge_count);
    std::vector<std::uint64_t> logical_to_projection(edge_count);
    transpose_cover_view_v1 bound_cover{};
    direct_gradient_order_v1 gradient_order{};
    require(bind_transpose_local_maps_v1(
        {cover, {destination_ids.data(), destination_ids.size()},
            identity_order.data(), 7u, 9u, 11u},
        {bound.data(), bound.size(), gradient_positions.data(),
            gradient_positions.size(), logical_to_projection.data(),
            logical_to_projection.size()}, &bound_cover, &gradient_order)
        == transpose_status_v1::success);

    std::vector<float> values(edge_count);
    for (std::uint64_t index = 0u; index < edge_count; ++index)
        values[index] = static_cast<float>((index % 13u) + 1u) / 16.0f;
    std::vector<float> destination_gradient(destination_count * width);
    for (std::uint64_t index = 0u; index < destination_gradient.size(); ++index)
        destination_gradient[index] =
            static_cast<float>(static_cast<int>(index % 17u) - 8) / 32.0f;
    std::vector<float> candidate(owner_count * width);
    transpose_reference_request_v1 reference{bound_cover, values.data(),
        destination_gradient.data(), destination_count, width, candidate.data(),
        candidate.size()};
    require(execute_transpose_reference_v1(reference)
        == transpose_status_v1::success);
    std::vector<float> oracle(candidate.size());
    transpose_validation_report_v1 report{};
    require(validate_transpose_exact_v1({reference, candidate.data(),
        candidate.size(), 0.0f}, oracle.data(), oracle.size(), &report)
        == transpose_status_v1::success);
    require(report.visited_edges == edge_count
        && report.visited_owner_segments == owner_count
        && report.compared_outputs == owner_count * width
        && report.maximum_absolute_error == 0.0f);
    candidate.back() += 1.0f;
    require(validate_transpose_exact_v1({reference, candidate.data(),
        candidate.size(), 0.0f}, oracle.data(), oracle.size(), &report)
        == transpose_status_v1::invalid_cover);
    require(report.first_mismatch == candidate.size() - 1u);
    return 0;
}
