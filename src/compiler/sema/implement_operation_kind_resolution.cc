#include <Cellerator/compiler/sema/implement_operation_kind_resolution_v1.hh>

#include <array>

namespace cellerator::compiler::sema::v1 {
namespace {
using core_kind = compute::operation::v2::operation_kind;
constexpr std::array<operation_kind_resolution, 14> coverage{{
    {source_operation_kind::relation_apply, "-[relation]->", core_kind::relation_apply, false},
    {source_operation_kind::relation_transpose, "transpose", core_kind::relation_apply_transpose, false},
    {source_operation_kind::support_contraction, "contract", core_kind::contract_on_support, false},
    {source_operation_kind::segment_statistics, "segment_reduce", core_kind::segment_reduce, false},
    {source_operation_kind::normalization, "normalize", core_kind::segment_normalize, false},
    {source_operation_kind::edge_map_or_gate, "edge_map", core_kind::edge_map_or_gate, false},
    {source_operation_kind::sparse_update, "sparse_update", core_kind::sparse_axis_update, false},
    {source_operation_kind::relation_bundle, "bundle", core_kind::relation_bundle_apply, false},
    {source_operation_kind::relation_chain, "chain", core_kind::relation_apply, true},
    {source_operation_kind::moments, "moments", core_kind::segment_reduce, true},
    {source_operation_kind::hierarchy, "hierarchy", core_kind::segment_reduce, true},
    {source_operation_kind::exchange, "exchange", core_kind::sparse_axis_update, true},
    {source_operation_kind::gradient, "gradient", core_kind::relation_apply_transpose, true},
    {source_operation_kind::publication, "publish", core_kind::sparse_axis_update, true},
}};
}  // namespace

const operation_kind_resolution *operation_kind_coverage_table() noexcept {
    return coverage.data();
}
std::uint32_t operation_kind_coverage_count() noexcept {
    return static_cast<std::uint32_t>(coverage.size());
}
const operation_kind_resolution *resolve_operation_kind(
    source_operation_kind kind) noexcept {
    for (const auto &entry : coverage) {
        if (entry.source == kind)
            return &entry;
    }
    return nullptr;
}

}  // namespace cellerator::compiler::sema::v1
