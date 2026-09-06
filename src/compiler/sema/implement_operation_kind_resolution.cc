#include <Cellerator/compiler/sema/implement_operation_kind_resolution_v1.hh>

#include <array>

namespace cellerator::compiler::sema::v1 {
namespace {
using core_kind = compute::operation::v2::operation_kind;
constexpr std::array<operation_kind_resolution, 14> coverage{{
    {source_operation_kind::relation_apply, "-[relation]->", core_kind::relation_apply},
    {source_operation_kind::relation_transpose, "transpose", core_kind::relation_apply_transpose},
    {source_operation_kind::support_contraction, "contract", core_kind::contract_on_support},
    {source_operation_kind::segment_statistics, "segment_reduce", core_kind::segment_reduce},
    {source_operation_kind::normalization, "normalize", core_kind::segment_normalize},
    {source_operation_kind::edge_map_or_gate, "edge_map", core_kind::edge_map_or_gate},
    {source_operation_kind::sparse_update, "sparse_update", core_kind::sparse_axis_update},
    {source_operation_kind::relation_bundle, "bundle", core_kind::relation_bundle_apply},
    {source_operation_kind::relation_chain, "chain", composition_kind::relation_chain},
    {source_operation_kind::moments, "moments", composition_kind::moments},
    {source_operation_kind::hierarchy, "hierarchy", composition_kind::hierarchy},
    {source_operation_kind::exchange, "exchange", composition_kind::exchange},
    {source_operation_kind::gradient, "gradient", composition_kind::gradient},
    {source_operation_kind::publication, "publish", effect_kind::publication},
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
