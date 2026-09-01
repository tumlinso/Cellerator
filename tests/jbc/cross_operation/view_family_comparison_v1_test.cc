#include <Cellerator/compute/projection_family/view_family_comparison_v1.hh>

#include <cassert>
#include <cstdint>
#include <limits>

namespace family = cellerator::compute::projection_family;
namespace execution = cellerator::execution;

namespace {

execution::persistent_axis_identity axis(std::uint64_t base) {
    return {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            {base + 1, base + 2}, {base + 3, base + 4},
            {base + 5, base + 6}, {base + 7, base + 8}};
}

family::view_family_measurement_v1 measurement(
    family::view_family_kind_v1 kind) {
    family::view_family_measurement_v1 value{};
    value.candidate_identity = kind == family::view_family_kind_v1::specialized
        ? cellerator::compute::operation::v2::stable_id{1, 2}
        : cellerator::compute::operation::v2::stable_id{3, 4};
    value.evidence_identity = {5, 6};
    value.family.family_identity = {7, 8};
    value.family.exact_support_identity = {9, 10};
    value.family.structure_identity = {11, 12};
    value.family.structure_epoch = {13};
    value.family.source_axis = axis(20);
    value.family.destination_axis = axis(40);
    value.family.logical_edge_order = {60, 61};
    value.family.logical_edge_count = 100;
    value.kind = kind;
    value.supported_operations = family::support_relation_apply_v1
        | family::support_relation_apply_transpose_v1;
    value.preparation_ns = 100;
    value.persistent_preprocess_ns = 50;
    value.input_pack_ns = 25;
    value.kernel_ns = 10;
    value.epilogue_ns = 2;
    value.output_transform_ns = 3;
    value.synchronization_ns = 1;
    value.communication_ns = 4;
    value.persistent_bytes = 1000;
    value.transient_bytes = 200;
    value.launch_count = 2;
    value.warmup_count = 5;
    value.repeat_count = 20;
    return value;
}

} // namespace

int main() {
    const auto specialized = measurement(family::view_family_kind_v1::specialized);
    auto generalized = measurement(family::view_family_kind_v1::generalized);
    generalized.preparation_ns = 10;
    generalized.persistent_preprocess_ns = 5;
    generalized.input_pack_ns = 5;
    generalized.kernel_ns = 25;
    generalized.persistent_bytes = 600;
    generalized.transient_bytes = 300;

    const auto low_reuse =
        family::compare_view_families_v1(specialized, generalized, 1);
    assert(low_reuse.compared());
    assert(low_reuse.latency_winner == family::view_family_winner_v1::right);
    assert(low_reuse.persistent_memory_winner
           == family::view_family_winner_v1::right);
    assert(low_reuse.transient_memory_winner
           == family::view_family_winner_v1::left);

    const auto high_reuse =
        family::compare_view_families_v1(specialized, generalized, 100);
    assert(high_reuse.latency_winner == family::view_family_winner_v1::left);

    auto self_certified = generalized;
    self_certified.correctness =
        family::correctness_evidence_kind_v1::provider_self_report;
    assert(family::compare_view_families_v1(
               specialized, self_certified, 10).code
           == family::view_family_comparison_code_v1::invalid_right);

    auto incomplete = generalized;
    incomplete.kernel_ns = 0;
    assert(!family::validate_view_family_measurement_v1(incomplete).valid());

    auto other_support = generalized;
    other_support.family.structure_epoch.value = 14;
    assert(family::compare_view_families_v1(
               specialized, other_support, 10).code
           == family::view_family_comparison_code_v1::family_mismatch);

    auto overflow = specialized;
    overflow.kernel_ns = std::numeric_limits<std::uint64_t>::max();
    assert(family::compare_view_families_v1(overflow, generalized, 2).code
           == family::view_family_comparison_code_v1::arithmetic_overflow);
}
