#include <Cellerator/compat/cp_math_v1/operation.hh>

namespace cellerator::compute::math {

namespace {

constexpr u64 splitmix64(u64 value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

void mix(u64 value, u64 *low, u64 *high) noexcept {
    *low = splitmix64(*low ^ value);
    *high = splitmix64(*high + value + 0x9e3779b97f4a7c15ull);
}

void mix_order(const feature_order_identity &order, u64 *low, u64 *high) noexcept {
    mix(order.schema_version, low, high);
    mix(static_cast<u32>(order.kind), low, high);
    mix(order.feature_count, low, high);
    mix(order.feature_axis_identity_version, low, high);
    mix(order.feature_axis_identity, low, high);
    mix(order.packing_geometry_identity, low, high);
}

} // namespace

operation_signature make_operation_signature(const spmm_request &request) noexcept {
    u64 low = 0x43504d4154484f50ull;
    u64 high = 0x53504d4d00000001ull;
    mix(request.schema_version, &low, &high);
    mix(static_cast<u32>(request.operation), &low, &high);
    mix(request.m, &low, &high);
    mix(request.k, &low, &high);
    mix(request.n, &low, &high);
    mix(request.sparse_nnz, &low, &high);
    mix(request.sparse_structure.schema_version, &low, &high);
    mix(request.sparse_structure.identity_version, &low, &high);
    mix(request.sparse_structure.value, &low, &high);
    mix(static_cast<u32>(request.transpose_sparse), &low, &high);
    mix(static_cast<u32>(request.transpose_dense), &low, &high);
    mix(static_cast<u32>(request.dense_rhs_layout), &low, &high);
    mix(static_cast<u32>(request.output_layout), &low, &high);
    mix(request.dense_rhs_leading_dimension, &low, &high);
    mix(request.output_leading_dimension, &low, &high);
    mix(request.sparse_storage_type_code, &low, &high);
    mix(request.dense_storage_type_code, &low, &high);
    mix(request.output_storage_type_code, &low, &high);
    mix(request.compute_type_code, &low, &high);
    mix(request.accumulation_type_code, &low, &high);
    mix(request.alpha.type_code, &low, &high);
    mix(request.alpha.bits, &low, &high);
    mix(request.beta.type_code, &low, &high);
    mix(request.beta.bits, &low, &high);
    mix(static_cast<u32>(request.determinism), &low, &high);
    mix(static_cast<u32>(request.workspace.kind), &low, &high);
    mix(request.workspace.byte_limit, &low, &high);
    mix(static_cast<u32>(request.reuse.kind), &low, &high);
    mix(request.reuse.expected_run_count, &low, &high);
    mix(static_cast<u32>(request.epilogue.kind), &low, &high);
    mix(request.epilogue.bias_type_code, &low, &high);
    mix(request.epilogue.bias_element_count, &low, &high);
    mix_order(request.sparse_feature_order, &low, &high);
    mix_order(request.dense_feature_order, &low, &high);
    return {operation_contract_schema_version, operation_kind::spmm, low, high};
}

} // namespace cellerator::compute::math
