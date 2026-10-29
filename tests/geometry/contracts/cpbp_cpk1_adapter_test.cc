#include <Cellerator/geometry/strategy/cpbp_v1_compatibility.hh>

#include <cassert>
#include <cstring>

namespace {

namespace geo = cellerator::geometry;
namespace strategy = cellerator::geometry::strategy;
namespace ex = cellerator::execution;

constexpr ex::axis_identity make_axis(std::uint32_t seed) noexcept {
    return {
        {seed + 1u, 1u},
        {seed + 2u, 1u},
        {seed + 3u, 1u},
        {seed + 4u, 1u}
    };
}

struct cpk1_fixture {
    unsigned char image[64]{};
    std::uint32_t feature_permutation[3] = {2u, 0u, 1u};
    std::uint32_t inverse_feature_permutation[3] = {1u, 2u, 0u};
    std::uint32_t feature_group_offsets[3] = {0u, 2u, 3u};
    std::uint32_t row_group_offsets[3] = {0u, 2u, 3u};
    std::uint32_t row_permutation[3] = {2u, 0u, 1u};
    std::uint32_t inverse_row_permutation[3] = {1u, 2u, 0u};
    cellpack::persistent_packing_payload_view payload{};

    cpk1_fixture() noexcept {
        for (std::uint32_t index = 0u; index < 64u; ++index)
            image[index] = static_cast<unsigned char>(index + 1u);
        payload.payload_schema_version =
            cellpack::persistent_packing_payload_schema_version;
        payload.payload_kind = cellpack::persistent_packing_payload_kind;
        payload.payload_identity = 0x12345678u;
        payload.image_base = image;
        payload.image_bytes = sizeof(image);
        payload.plan_identity.feature_axis_fingerprint = 0x91u;
        payload.plan_identity.feature_axis_fingerprint_version = 1u;
        payload.objective_kind =
            cellpack::packing_exact_objective_kind::row_active_block_references;
        payload.cost_policy_identity = 0x92u;
        payload.maximum_feature_block_width = 2u;
        payload.row_group_width = 2u;
        payload.inverse_feature_permutation = inverse_feature_permutation;
        payload.row_group_count = 2u;
        payload.row_group_offsets = row_group_offsets;
        payload.plan.feature_count = 3u;
        payload.plan.feature_block_count = 2u;
        payload.plan.feature_permutation = feature_permutation;
        payload.plan.feature_block_offsets = feature_group_offsets;
        payload.order.row_count = 3u;
        payload.order.row_permutation = row_permutation;
        payload.order.inverse_row_permutation = inverse_row_permutation;
        payload.tiles.feature_count = 3u;
        payload.tiles.row_count = 3u;
        payload.tiles.nnz_count = 5u;
    }
};

strategy::cpbp_v1_semantic_binding_v1 make_binding(
    const std::uint32_t *window_members) noexcept {
    strategy::cpbp_v1_semantic_binding_v1 binding{};
    binding.structure = {30u, 1u};
    binding.structure_epoch = {7u};
    binding.source_feature_axis = make_axis(10u);
    binding.destination_row_axis = make_axis(20u);
    binding.work_window.identity = {0x101u, 0x202u};
    binding.work_window.axis = binding.destination_row_axis;
    binding.work_window.axis_extent = 3u;
    binding.work_window.member_count = 3u;
    binding.work_window.members = window_members;
    return binding;
}

void test_aliases_frozen_semantics_without_modifying_cpk1() {
    cpk1_fixture fixture;
    unsigned char before[64]{};
    std::memcpy(before, fixture.image, sizeof(before));
    const std::uint32_t window_members[] = {0u, 1u, 2u};
    const auto binding = make_binding(window_members);
    geo::semantic_component_v1 component{};
    std::uint64_t edge_ids[5]{};
    strategy::cpbp_v1_semantic_adapter_v1 adapter{};

    assert(strategy::adapt_validated_cpbp_v1_payload(fixture.payload, binding,
        {&component, edge_ids, 5u}, &adapter)
        == strategy::cpbp_v1_semantic_adapter_status::ok);
    assert(std::memcmp(before, fixture.image, sizeof(before)) == 0);
    assert(adapter.feature_execution_to_canonical
        == fixture.feature_permutation);
    assert(adapter.feature_canonical_to_execution
        == fixture.inverse_feature_permutation);
    assert(adapter.feature_group_offsets == fixture.feature_group_offsets);
    assert(adapter.row_group_offsets == fixture.row_group_offsets);
    assert(adapter.work_layout.execution_to_window
        == fixture.row_permutation);
    assert(adapter.work_layout.window_to_execution
        == fixture.inverse_row_permutation);
    assert(adapter.objective_kind == fixture.payload.objective_kind);
    assert(adapter.cost_policy_identity == fixture.payload.cost_policy_identity);
    assert(component.kind == geo::semantic_component_kind::unstructured);
    for (std::uint64_t edge = 0u; edge < 5u; ++edge)
        assert(edge_ids[edge] == edge);
}

void test_emitted_contracts_pass_independent_validation() {
    cpk1_fixture fixture;
    const std::uint32_t window_members[] = {0u, 1u, 2u};
    const auto binding = make_binding(window_members);
    geo::semantic_component_v1 component{};
    std::uint64_t edge_ids[5]{};
    std::uint8_t marks[5]{};
    strategy::cpbp_v1_semantic_adapter_v1 adapter{};
    assert(strategy::adapt_validated_cpbp_v1_payload(fixture.payload, binding,
        {&component, edge_ids, 5u}, &adapter)
        == strategy::cpbp_v1_semantic_adapter_status::ok);
    assert(geo::validate_work_layout(binding.work_window, adapter.work_layout));
    assert(geo::validate_relation_cover(
        adapter.relation_cover, {marks, 5u}));
}

void test_rejects_nonidentity_window_and_insufficient_capacity() {
    cpk1_fixture fixture;
    const std::uint32_t reordered_members[] = {2u, 0u, 1u};
    auto binding = make_binding(reordered_members);
    geo::semantic_component_v1 component{};
    std::uint64_t edge_ids[5]{};
    strategy::cpbp_v1_semantic_adapter_v1 adapter{};
    assert(strategy::adapt_validated_cpbp_v1_payload(fixture.payload, binding,
        {&component, edge_ids, 5u}, &adapter)
        == strategy::cpbp_v1_semantic_adapter_status::incompatible_work_window);

    const std::uint32_t identity_members[] = {0u, 1u, 2u};
    binding = make_binding(identity_members);
    assert(strategy::adapt_validated_cpbp_v1_payload(fixture.payload, binding,
        {&component, edge_ids, 4u}, &adapter)
        == strategy::cpbp_v1_semantic_adapter_status::insufficient_capacity);
}

void test_rejects_stale_binding_and_unvalidated_payload_shape() {
    cpk1_fixture fixture;
    const std::uint32_t window_members[] = {0u, 1u, 2u};
    auto binding = make_binding(window_members);
    geo::semantic_component_v1 component{};
    std::uint64_t edge_ids[5]{};
    strategy::cpbp_v1_semantic_adapter_v1 adapter{};

    binding.structure_epoch = {};
    assert(strategy::adapt_validated_cpbp_v1_payload(fixture.payload, binding,
        {&component, edge_ids, 5u}, &adapter)
        == strategy::cpbp_v1_semantic_adapter_status::invalid_binding);

    binding = make_binding(window_members);
    fixture.payload.payload_identity = 0u;
    assert(strategy::adapt_validated_cpbp_v1_payload(fixture.payload, binding,
        {&component, edge_ids, 5u}, &adapter)
        == strategy::cpbp_v1_semantic_adapter_status::invalid_payload_contract);
}

} // namespace

int main() {
    test_aliases_frozen_semantics_without_modifying_cpk1();
    test_emitted_contracts_pass_independent_validation();
    test_rejects_nonidentity_window_and_insufficient_capacity();
    test_rejects_stale_binding_and_unvalidated_payload_shape();
    return 0;
}
