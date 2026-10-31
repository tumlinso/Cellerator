#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Cellerator/nn/relations.hh>

#include <c10/core/InferenceMode.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

cellerator::biology::AxisDescriptor axis(std::uint64_t order, std::int64_t extent) {
    return {
        cellerator::execution::domain_id{1, 0},
        cellerator::execution::order_id{order, 0},
        cellerator::execution::geometry_id{3, 0},
        cellerator::execution::partition_id{4, 0},
        extent};
}

template<class Callback>
bool throws_invalid_argument(const Callback& callback) {
    try {
        callback();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    using cellerator::biology::OccurrenceCoordinates;
    using cellerator::nn::CsrSupport;
    using cellerator::nn::PreparedRelation;

    auto source = OccurrenceCoordinates<std::string>::from_ids(
        {"gene-A", "gene-A", "gene-B", "gene-C", "gene-C"},
        {0, 3, 5}, axis(2, 5), 11, {0, 1, 2, 3, 4});
    auto target = OccurrenceCoordinates<std::string>::from_ids(
        {"program-A", "program-B", "program-C", "program-C"},
        {0, 2, 4}, axis(5, 4), 11, {10, 11, 12, 13});

    // CSR rows address target occurrences and columns address source
    // occurrences.  Duplicate edges and an empty target row are explicit.
    auto relation = PreparedRelation::from_csr(
        CsrSupport{{0, 2, 4, 6, 6}, {0, 0, 1, 2, 3, 4}},
        source.order_token(), target.order_token(), 7);
    assert(relation.source_count() == 5);
    assert(relation.target_count() == 4);
    assert(relation.edge_count() == 6);
    assert(relation.generation() == 7);
    assert(relation.device().is_cpu());
    assert(relation.source_axis() && relation.source_axis()->extent == 5);
    assert(relation.target_axis() && relation.target_axis()->extent == 4);
    assert((relation.source_offsets() == std::vector<std::int64_t>{0, 3, 5}));
    assert((relation.target_offsets() == std::vector<std::int64_t>{0, 2, 4}));
    assert((relation.host_rows() == std::vector<std::int64_t>{0, 0, 1, 1, 2, 2}));
    assert((relation.host_col_indices() == std::vector<std::int64_t>{0, 0, 1, 2, 3, 4}));
    assert((relation.host_causal_mask() == std::vector<std::uint8_t>{1, 1, 1, 0, 1, 0}));
    assert(relation.has_future_edges());

    relation.validate_for(source.order_token(), target.order_token(), 5, 4,
                          7, at::Device(at::kCPU));
    assert(relation.row_ptr().device().is_cpu());
    assert(relation.col_indices().numel() == 6);
    assert(relation.rows().numel() == 6);
    assert(relation.causal_mask().numel() == 6);

    const auto causal = relation.causal();
    assert(!causal.has_future_edges());
    assert((causal.host_row_ptr() == std::vector<std::int64_t>{0, 2, 3, 4, 4}));
    assert((causal.host_col_indices() == std::vector<std::int64_t>{0, 0, 1, 3}));
    assert(causal.source_order_token().same_order(source.order_token()));
    assert(causal.target_order_token().same_order(target.order_token()));

    const auto reversed = relation.transpose();
    assert(reversed.source_count() == 4);
    assert(reversed.target_count() == 5);
    assert((reversed.host_row_ptr()
        == std::vector<std::int64_t>{0, 2, 3, 4, 5, 6}));
    assert((reversed.host_col_indices() == std::vector<std::int64_t>{0, 0, 1, 1, 2, 2}));
    assert(reversed.source_order_token().same_order(target.order_token()));
    assert(reversed.target_order_token().same_order(source.order_token()));

    const auto cpu_copy = relation.to(at::Device(at::kCPU));
    assert(cpu_copy.host_row_ptr() == relation.host_row_ptr());
    assert(cpu_copy.source_order_token().same_order(source.order_token()));

    // A rebuilt declaration with identical IDs and extents is still stale:
    // the order token is structural identity, not a shape check.
    auto rebuilt_source = OccurrenceCoordinates<std::string>::from_ids(
        {"gene-A", "gene-A", "gene-B", "gene-C", "gene-C"},
        {0, 3, 5}, axis(2, 5), 11, {0, 1, 2, 3, 4});
    assert(throws_invalid_argument([&] {
        relation.validate_for(rebuilt_source.order_token(), target.order_token(),
                              5, 4, at::Device(at::kCPU));
    }));
    assert(throws_invalid_argument([&] {
        relation.validate_for(source.order_token(), target.order_token(),
                              5, 4, 8, at::Device(at::kCPU));
    }));
    assert(throws_invalid_argument([&] {
        relation.validate_for(source.order_token(), target.order_token(),
                              6, 4, at::Device(at::kCPU));
    }));

    assert(throws_invalid_argument([&] {
        (void)PreparedRelation::from_csr(
            CsrSupport{{0, 1, 1, 1, 1}, {3}},
            source.order_token(), target.order_token(), 7);
    }));

    auto mutable_relation = PreparedRelation::from_csr(
        CsrSupport{{0, 2, 4, 6, 6}, {0, 0, 1, 2, 3, 4}},
        source.order_token(), target.order_token(), 7);
    mutable_relation.col_indices().fill_(0);
    assert(throws_invalid_argument([&] {
        mutable_relation.validate_for(source.order_token(), target.order_token(),
                                      5, 4, at::Device(at::kCPU));
    }));

    // ATen inference tensors do not carry mutation versions. Preparation
    // explicitly disables inference mode while it creates owned structure.
    std::optional<PreparedRelation> inference_prepared;
    {
        c10::InferenceMode inference_mode;
        inference_prepared.emplace(PreparedRelation::from_csr(
            CsrSupport{{0, 1, 1, 1, 1}, {0}},
            source.order_token(), target.order_token(), 7));
    }
    inference_prepared->validate_storage();
}
