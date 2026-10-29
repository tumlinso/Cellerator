#include <Cellerator/geometry/admissibility.hh>
#include <Cellerator/geometry/compiler/compile_geometry.hh>
#include <Cellerator/geometry/persistence/semantic_geometry_image_v1.hh>
#include <Cellerator/geometry/strategy/cpbp_v1_compatibility.hh>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace geo = cellerator::geometry;
namespace comp = cellerator::geometry::compiler;
namespace persist = cellerator::geometry::persistence;
namespace strategy = cellerator::geometry::strategy;
namespace exec = cellerator::execution;

namespace {

constexpr std::uint32_t maximum_items = 8u;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "semantic_compiler_property_test: " << message << '\n';
        std::exit(1);
    }
}

constexpr exec::axis_identity compact_axis(std::uint32_t seed) noexcept {
    return {{seed + 1u, 1u}, {seed + 2u, 1u}, {seed + 3u, 1u},
        {seed + 4u, 1u}};
}

exec::persistent_axis_identity persistent_axis(std::uint64_t seed) noexcept {
    exec::persistent_axis_identity result{};
    result.header = {exec::biological_abi_version,
        exec::serialized_record_kind::persistent_axis_identity,
        sizeof(exec::persistent_axis_identity)};
    result.domain = {seed + 1u, seed + 2u};
    result.order = {seed + 3u, seed + 4u};
    result.geometry = {seed + 5u, seed + 6u};
    result.partition = {seed + 7u, seed + 8u};
    return result;
}

comp::geometry_problem_v1 make_problem(
    const std::uint32_t *members,
    std::uint32_t count,
    std::uint64_t edge_count,
    const geo::admissibility_view_v1 *admissibility = nullptr) noexcept {
    comp::geometry_problem_v1 problem{};
    problem.primary_relation.source_axis = compact_axis(10u);
    problem.primary_relation.destination_axis = compact_axis(20u);
    problem.primary_relation.structure = {31u, 1u};
    problem.primary_relation.epoch = {9u};
    problem.primary_relation.logical_edge_count = edge_count;
    problem.primary_relation.location =
        {exec::residency_kind::host, {}, -1, 0u};
    problem.work_window.identity = {0x101u, 0x202u};
    problem.work_window.axis = problem.primary_relation.destination_axis;
    problem.work_window.axis_extent = maximum_items;
    problem.work_window.member_count = count;
    problem.work_window.members = members;
    problem.admissibility = admissibility;
    problem.workload.relation_value_type = exec::numeric_type::f32;
    problem.workload.dense_input_type = exec::numeric_type::f32;
    problem.workload.accumulation_type = exec::numeric_type::f32;
    problem.workload.output_type = exec::numeric_type::f32;
    problem.workload.dense_width = 4u;
    return problem;
}

struct solution_storage {
    std::uint32_t forward[maximum_items]{};
    std::uint32_t inverse[maximum_items]{};
    geo::semantic_component_v1 component{};
    std::uint64_t edges[maximum_items]{};
    std::uint8_t marks[maximum_items]{};
};

comp::geometry_solution_buffers_v1 solution_buffers(
    solution_storage &storage) noexcept {
    return {storage.forward, storage.inverse, maximum_items,
        &storage.component, 1u, storage.edges, maximum_items};
}

comp::geometry_solution_v1 compile_identity(
    const comp::geometry_problem_v1 &problem,
    solution_storage &storage) {
    const comp::geometry_strategy_descriptor_v1 &identity =
        comp::identity_geometry_strategy();
    const comp::geometry_strategy_registry_v1 registry{&identity, 1u};
    comp::geometry_search_policy_v1 policy{};
    policy.strategy_id = identity.strategy_id;
    comp::geometry_solution_v1 solution{};
    require(comp::compile_geometry(registry, problem, policy, {},
                solution_buffers(storage),
                {storage.marks, maximum_items}, &solution)
            == comp::geometry_strategy_status::ok,
        "permissive identity compilation failed");
    return solution;
}

void check_permissive_window_properties() {
    for (std::uint32_t count = 1u; count <= maximum_items; ++count) {
        std::uint32_t members[maximum_items]{};
        for (std::uint32_t index = 0u; index < count; ++index)
            members[index] = count - index - 1u;
        solution_storage storage{};
        const comp::geometry_problem_v1 problem =
            make_problem(members, count, count);
        const comp::geometry_solution_v1 solution =
            compile_identity(problem, storage);
        require(static_cast<bool>(geo::validate_work_layout(
                    problem.work_window, solution.work_layout)),
            "compiled work order did not independently validate");
        require(static_cast<bool>(geo::validate_relation_cover(
                    solution.relation_cover,
                    {storage.marks, maximum_items})),
            "compiled exact cover did not independently validate");
        for (std::uint32_t index = 0u; index < count; ++index)
            require(storage.forward[index] == index
                    && storage.inverse[index] == index
                    && geo::work_layout_axis_position(problem.work_window,
                           solution.work_layout, index) == members[index],
                "identity strategy changed selected work-window order");
    }

    const std::uint32_t members[] = {3u, 0u, 2u};
    geo::admissibility_record_v1 constraint{};
    constraint.axis = compact_axis(20u);
    constraint.subject = 3u;
    geo::admissibility_view_v1 constrained{};
    constrained.record_count = 1u;
    constrained.records = &constraint;
    solution_storage storage{};
    const comp::geometry_problem_v1 problem =
        make_problem(members, 3u, 3u, &constrained);
    const auto &identity = comp::identity_geometry_strategy();
    const comp::geometry_strategy_registry_v1 registry{&identity, 1u};
    comp::geometry_search_policy_v1 policy{};
    policy.strategy_id = identity.strategy_id;
    comp::geometry_solution_v1 solution{};
    require(comp::compile_geometry(registry, problem, policy, {},
                solution_buffers(storage), {storage.marks, maximum_items},
                &solution)
            == comp::geometry_strategy_status::requirements_failed,
        "identity strategy accepted a nonpermissive window");
}

bool same_geometry(exec::geometry_id lhs, exec::geometry_id rhs) noexcept {
    return lhs.low == rhs.low && lhs.high == rhs.high;
}

persist::semantic_geometry_image_view_v1 build_image(
    const comp::geometry_problem_v1 &problem,
    const comp::geometry_solution_v1 &solution,
    std::uint8_t *image,
    std::size_t image_capacity,
    std::uint8_t *marks,
    std::uint64_t mark_capacity) {
    persist::semantic_geometry_image_build_request_v1 request{};
    request.relation = {0x301u, 0x302u};
    request.structure = {0x401u, 0x402u};
    request.structure_epoch = problem.primary_relation.epoch;
    request.source_axis = persistent_axis(500u);
    request.destination_axis = persistent_axis(600u);
    request.work_window = problem.work_window;
    request.work_layout = solution.work_layout;
    request.relation_cover = solution.relation_cover;
    persist::semantic_geometry_image_view_v1 view{};
    require(persist::build_semantic_geometry_image_v1(request,
                {image, image_capacity}, {marks, mark_capacity}, &view)
            == persist::semantic_geometry_image_status_v1::ok,
        "CSG1 build failed");
    return view;
}

void check_round_trip_corruption_relocation_and_identity() {
    const std::uint32_t members[] = {5u, 1u, 3u};
    solution_storage first_storage{};
    const comp::geometry_problem_v1 problem =
        make_problem(members, 3u, 4u);
    const comp::geometry_solution_v1 solution =
        compile_identity(problem, first_storage);
    alignas(64) std::uint8_t first_image[4096]{};
    alignas(64) std::uint8_t relocated[4096]{};
    auto first = build_image(problem, solution, first_image,
        sizeof(first_image), first_storage.marks, maximum_items);
    std::memcpy(relocated, first_image,
        static_cast<std::size_t>(first.image_bytes));
    persist::semantic_geometry_image_view_v1 rebound{};
    require(persist::rebind_semantic_geometry_image_v1(first, relocated,
                first.image_bytes, &rebound)
            == persist::semantic_geometry_image_status_v1::ok
            && same_geometry(first.geometry_identity,
                rebound.geometry_identity),
        "valid relocation failed or changed geometry identity");
    persist::semantic_geometry_image_view_v1 round_trip{};
    require(persist::validate_semantic_geometry_image_v1(relocated,
                first.image_bytes,
                {first_storage.marks, maximum_items}, &round_trip)
            == persist::semantic_geometry_image_status_v1::ok,
        "relocated CSG1 round trip failed");

    relocated[first.image_bytes - 1u] ^= 1u;
    require(persist::rebind_semantic_geometry_image_v1(first, relocated,
                first.image_bytes, &rebound)
            == persist::semantic_geometry_image_status_v1::incompatible_relocation,
        "relocation accepted corrupted bytes");
    require(persist::validate_semantic_geometry_image_v1(relocated,
                first.image_bytes,
                {first_storage.marks, maximum_items}, &round_trip)
            != persist::semantic_geometry_image_status_v1::ok,
        "CSG1 corruption validated");

    std::uint32_t reordered_forward[3]{1u, 2u, 0u};
    std::uint32_t reordered_inverse[3]{2u, 0u, 1u};
    comp::geometry_solution_v1 reordered = solution;
    reordered.work_layout.execution_to_window = reordered_forward;
    reordered.work_layout.window_to_execution = reordered_inverse;
    alignas(64) std::uint8_t reordered_image[4096]{};
    const auto second = build_image(problem, reordered, reordered_image,
        sizeof(reordered_image), first_storage.marks, maximum_items);
    require(!same_geometry(first.geometry_identity,
                second.geometry_identity),
        "execution-order change did not change persistent geometry identity");

    std::uint64_t duplicate_edges[4]{0u, 1u, 1u, 3u};
    geo::relation_cover_view_v1 malformed = solution.relation_cover;
    malformed.logical_edge_ids = duplicate_edges;
    require(!geo::validate_relation_cover(malformed,
                {first_storage.marks, maximum_items}),
        "duplicate exact-cover ownership validated");
}

struct cpk1_fixture {
    std::uint8_t image[64]{};
    std::uint32_t feature_forward[3]{2u, 0u, 1u};
    std::uint32_t feature_inverse[3]{1u, 2u, 0u};
    std::uint32_t feature_offsets[3]{0u, 2u, 3u};
    std::uint32_t row_offsets[3]{0u, 2u, 3u};
    std::uint32_t row_forward[3]{2u, 0u, 1u};
    std::uint32_t row_inverse[3]{1u, 2u, 0u};
    cellpack::persistent_packing_payload_view payload{};

    cpk1_fixture() noexcept {
        payload.payload_schema_version =
            cellpack::persistent_packing_payload_schema_version;
        payload.payload_kind = cellpack::persistent_packing_payload_kind;
        payload.payload_identity = 0x1234u;
        payload.image_base = image;
        payload.image_bytes = sizeof(image);
        payload.maximum_feature_block_width = 2u;
        payload.row_group_width = 2u;
        payload.inverse_feature_permutation = feature_inverse;
        payload.row_group_count = 2u;
        payload.row_group_offsets = row_offsets;
        payload.plan.feature_count = 3u;
        payload.plan.feature_block_count = 2u;
        payload.plan.feature_permutation = feature_forward;
        payload.plan.feature_block_offsets = feature_offsets;
        payload.order.row_count = 3u;
        payload.order.row_permutation = row_forward;
        payload.order.inverse_row_permutation = row_inverse;
        payload.tiles.feature_count = 3u;
        payload.tiles.row_count = 3u;
        payload.tiles.nnz_count = 5u;
    }
};

void check_cpbp_compatibility() {
    cpk1_fixture fixture{};
    const std::uint32_t members[3]{0u, 1u, 2u};
    strategy::cpbp_v1_semantic_binding_v1 binding{};
    binding.structure = {30u, 1u};
    binding.structure_epoch = {7u};
    binding.source_feature_axis = compact_axis(10u);
    binding.destination_row_axis = compact_axis(20u);
    binding.work_window.identity = {0x101u, 0x202u};
    binding.work_window.axis = binding.destination_row_axis;
    binding.work_window.axis_extent = 3u;
    binding.work_window.member_count = 3u;
    binding.work_window.members = members;
    geo::semantic_component_v1 component{};
    std::uint64_t edge_ids[5]{};
    std::uint8_t marks[5]{};
    strategy::cpbp_v1_semantic_adapter_v1 adapter{};
    require(strategy::adapt_validated_cpbp_v1_payload(fixture.payload,
                binding, {&component, edge_ids, 5u}, &adapter)
            == strategy::cpbp_v1_semantic_adapter_status::ok,
        "frozen CP-BP payload did not adapt");
    require(adapter.work_layout.execution_to_window == fixture.row_forward
            && geo::validate_work_layout(
                binding.work_window, adapter.work_layout)
            && geo::validate_relation_cover(
                adapter.relation_cover, {marks, 5u}),
        "CP-BP compatibility contracts failed independent validation");
    require(strategy::adapt_validated_cpbp_v1_payload(fixture.payload,
                binding, {&component, edge_ids, 4u}, &adapter)
            == strategy::cpbp_v1_semantic_adapter_status::insufficient_capacity,
        "CP-BP adapter accepted insufficient exact-cover capacity");
}

} // namespace

int main() {
    check_permissive_window_properties();
    check_round_trip_corruption_relocation_and_identity();
    check_cpbp_compatibility();
    std::cout << "semantic_compiler_property_test: ok\n";
    return 0;
}
