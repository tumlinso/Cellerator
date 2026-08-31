#include <Cellerator/compute/compute.hh>

#include <cassert>
#include <array>
#include <cstdint>

int main() {
    const cellerator::compute::ce_exop_operation_portfolio_v1 portfolio =
        cellerator::compute::query_ce_exop_operation_portfolio_v1();
    assert(portfolio.relation_apply_candidates >= 15u);
    assert(portfolio.residual_candidates >= 6u);
    assert(portfolio.contraction_candidates >= 4u);
    assert(portfolio.transpose_candidates >= 2u);
    assert(portfolio.gate_and_update_candidates == 14u);
    assert(portfolio.bundle_and_chain_candidates == 6u);
    assert(portfolio.segment_candidates == 162u);
    assert(portfolio.all_candidates_planner_owned);
    assert(portfolio.all_experimental_candidates_require_measurement);

    namespace edge = cellerator::compute::operation::edge;
    namespace update = cellerator::compute::operation::sparse_axis_update;
    std::size_t gate_count = 0u;
    const edge::registry_entry_v1 *gates = edge::registry_v1(&gate_count);
    assert(gates != nullptr && gate_count == 14u);
    for (std::size_t left = 0u; left < gate_count; ++left) {
        assert(gates[left].stable_id != 0u && gates[left].requires_measurement
            && !gates[left].promoted);
        for (std::size_t right = left + 1u; right < gate_count; ++right)
            assert(gates[left].stable_id != gates[right].stable_id);
    }
    edge::edge_coordinate_v1 coordinates[]{{0u, 1u, 0u}, {1u, 0u, 0u}};
    edge::validation_result_v1 validation{};
    assert(edge::validate_edge_coordinates_v1(coordinates,
        {(std::uint64_t{1} << 32) + 7u, 2u}, 2u, 2u, 1u, &validation)
        == edge::status_v1::success);
    assert(validation.valid && validation.checked_item_count == 2u);
    const float gate_input[]{2.0f, 3.0f};
    const float source_gate[]{5.0f, 7.0f};
    float gated[2]{};
    assert(edge::reference_indexed_gate_v1(coordinates, 2u, gate_input,
        source_gate, nullptr, edge::indexed_gate_kind_v1::per_source, gated)
        == edge::status_v1::success);
    assert(gated[0] == 10.0f && gated[1] == 21.0f);
    float target[]{1.0f, 2.0f};
    const std::uint64_t indices[]{(std::uint64_t{1} << 32) + 10u};
    const float updates[]{4.0f};
    assert(edge::reference_sparse_axis_update_v1(target, 2u, 1u, indices,
        (std::uint64_t{1} << 32) + 10u, updates, 1u,
        update::operation_v1::add, &validation) == update::status_v1::success);
    assert(target[0] == 5.0f && target[1] == 2.0f);

    namespace segment = cellerator::compute::segment;
    std::array<segment::segment_candidate_descriptor_v2, 162> segments{};
    segment::segment_candidate_buffer_v2 segment_buffer{
        segments.data(), static_cast<std::uint32_t>(segments.size()), 0u};
    assert(segment::enumerate_segment_candidates_v2(segment_buffer));
    assert(segment_buffer.count == segments.size());
    assert(segment::validate_segment_candidate_catalog_v2(
        segments.data(), segment_buffer.count));
}
