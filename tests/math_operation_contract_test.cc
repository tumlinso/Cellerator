#include <Cellerator/compute/math/execution_plan.hh>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>

namespace cm = cellerator::compute::math;
namespace cr = cellerator::real;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "mathOperationContractTest: " << message << '\n';
        std::exit(1);
    }
}

cm::feature_order_identity canonical_order(cm::u32 features) {
    cm::feature_order_identity order;
    order.feature_count = features;
    order.feature_axis_identity_version = 1u;
    order.feature_axis_identity = 0x12345678ull;
    return order;
}

cm::spmm_request request_fixture() {
    cm::spmm_request request;
    request.m = 8u;
    request.k = 16u;
    request.n = 32u;
    request.sparse_nnz = 24u;
    request.sparse_structure.identity_version = 1u;
    request.sparse_structure.value = 0xc511e7a70ull;
    request.dense_rhs_leading_dimension = 32u;
    request.output_leading_dimension = 32u;
    request.sparse_storage_type_code = cr::value_f16;
    request.dense_storage_type_code = cr::value_f16;
    request.output_storage_type_code = cr::value_f32;
    request.compute_type_code = cr::value_f32;
    request.accumulation_type_code = cr::value_f32;
    request.alpha = cm::make_scalar(1.0f);
    request.beta = cm::make_scalar(0.0f);
    request.sparse_feature_order = canonical_order(16u);
    request.dense_feature_order = request.sparse_feature_order;
    return request;
}

cm::math_request bound_request_fixture() {
    static int sparse = 0;
    static int dense = 0;
    static int output = 0;
    cm::math_request request;
    request.operation = request_fixture();
    request.bindings.sparse_matrix = &sparse;
    request.bindings.dense_rhs = &dense;
    request.bindings.output = &output;
    return request;
}

void test_signature_and_order_identity() {
    cm::spmm_request request = request_fixture();
    require(static_cast<bool>(cm::validate_spmm_request(request)),
        "valid request rejected");
    const cm::operation_signature first = cm::make_operation_signature(request);
    const cm::operation_signature second = cm::make_operation_signature(request);
    require(first.low == second.low && first.high == second.high,
        "operation signature is unstable");
    require(first.low == 0xa7e9d18214be8880ull
            && first.high == 0xc02e69bb5dae6efbull,
        "operation signature schema changed without a version revision");

    request.dense_rhs_layout = cm::dense_layout_kind::column_major;
    const cm::operation_signature column_major = cm::make_operation_signature(request);
    require(first.low != column_major.low || first.high != column_major.high,
        "dense layout is absent from operation signature");
    request = request_fixture();
    request.sparse_structure.value += 1u;
    const cm::operation_signature other_structure =
        cm::make_operation_signature(request);
    require(first.low != other_structure.low || first.high != other_structure.high,
        "sparse structure is absent from operation signature");

    request = request_fixture();
    request.dense_feature_order.kind = cm::feature_order_kind::packed;
    request.dense_feature_order.packing_geometry_identity = 0xabcdu;
    const auto mismatch = cm::validate_spmm_request(request);
    require(mismatch.code == cm::request_validation_code::feature_order_mismatch,
        "canonical and packed operands were silently mixed");
}

void test_binding_and_policy_validation() {
    cm::math_request request = bound_request_fixture();
    require(static_cast<bool>(cm::validate_math_request(request)),
        "valid bound request rejected");

    request.bindings.dense_rhs = nullptr;
    require(cm::validate_math_request(request).code
            == cm::request_validation_code::missing_binding,
        "missing dense input binding was accepted");

    request = bound_request_fixture();
    request.operation.workspace.kind = cm::workspace_policy_kind::caller_limit;
    require(cm::validate_math_request(request).code
            == cm::request_validation_code::invalid_workspace_policy,
        "zero-byte caller workspace limit was accepted");

    request = bound_request_fixture();
    request.operation.reuse.kind = cm::expected_reuse_kind::persistent;
    require(cm::validate_math_request(request).code
            == cm::request_validation_code::invalid_reuse,
        "persistent reuse with a finite run count was accepted");
    request.operation.reuse.expected_run_count = 0u;
    require(static_cast<bool>(cm::validate_math_request(request)),
        "canonical persistent reuse was rejected");
}

void test_shape_and_enum_validation() {
    cm::spmm_request request = request_fixture();
    request.sparse_nnz = request.m * request.k + 1u;
    require(cm::validate_spmm_request(request).code
            == cm::request_validation_code::invalid_shape,
        "sparse_nnz larger than M*K was accepted");

    request = request_fixture();
    request.m = std::numeric_limits<cm::u64>::max();
    request.n = 2u;
    require(cm::validate_spmm_request(request).code
            == cm::request_validation_code::invalid_shape,
        "overflowing output shape was accepted");

    request = request_fixture();
    request.k = std::numeric_limits<cm::u64>::max();
    request.n = 2u;
    require(cm::validate_spmm_request(request).code
            == cm::request_validation_code::invalid_shape,
        "overflowing dense RHS shape was accepted");

    request = request_fixture();
    request.dense_rhs_layout = static_cast<cm::dense_layout_kind>(99u);
    require(cm::validate_spmm_request(request).code
            == cm::request_validation_code::invalid_layout,
        "unknown dense layout was accepted");

    request = request_fixture();
    request.dense_rhs_leading_dimension = request.n - 1u;
    require(cm::validate_spmm_request(request).code
            == cm::request_validation_code::invalid_layout,
        "undersized dense RHS leading dimension was accepted");

    request = request_fixture();
    request.determinism = static_cast<cm::determinism_requirement>(99u);
    require(cm::validate_spmm_request(request).code
            == cm::request_validation_code::invalid_determinism,
        "unknown determinism requirement was accepted");
}

void test_bias_contract() {
    static float bias[32]{};
    cm::math_request request = bound_request_fixture();
    request.operation.epilogue.kind = cm::epilogue_kind::bias_relu;
    request.operation.epilogue.bias_type_code = cr::value_f32;
    request.operation.epilogue.bias_element_count = request.operation.n;
    require(cm::validate_math_request(request).code
            == cm::request_validation_code::missing_bias,
        "bias metadata without a bias binding was accepted");
    request.bindings.bias = bias;
    require(static_cast<bool>(cm::validate_math_request(request)),
        "valid bias epilogue rejected");
}

void test_explicit_gelu_semantics() {
    cm::spmm_request exact = request_fixture();
    cm::spmm_request approximate = request_fixture();
    exact.epilogue.kind = cm::epilogue_kind::gelu_exact_erf;
    approximate.epilogue.kind = cm::epilogue_kind::gelu_tanh_approximate;
    require(static_cast<bool>(cm::validate_spmm_request(exact))
            && static_cast<bool>(cm::validate_spmm_request(approximate)),
        "explicit GELU modes were rejected");
    const cm::operation_signature exact_signature =
        cm::make_operation_signature(exact);
    const cm::operation_signature approximate_signature =
        cm::make_operation_signature(approximate);
    require(exact_signature.low != approximate_signature.low
            || exact_signature.high != approximate_signature.high,
        "exact and approximate GELU share an operation signature");

    cm::math_request bias_exact = bound_request_fixture();
    static float bias[32]{};
    bias_exact.operation.epilogue.kind =
        cm::epilogue_kind::bias_gelu_exact_erf;
    bias_exact.operation.epilogue.bias_type_code = cr::value_f32;
    bias_exact.operation.epilogue.bias_element_count = bias_exact.operation.n;
    bias_exact.bindings.bias = bias;
    require(static_cast<bool>(cm::validate_math_request(bias_exact)),
        "bias plus exact GELU was rejected");
}

void test_packed_plan_identity_mismatch() {
    cm::spmm_request request = request_fixture();
    request.sparse_feature_order.kind = cm::feature_order_kind::packed;
    request.dense_feature_order.kind = cm::feature_order_kind::packed;
    request.sparse_feature_order.packing_geometry_identity = 0x100u;
    request.dense_feature_order.packing_geometry_identity = 0x200u;
    require(cm::validate_spmm_request(request).code
            == cm::request_validation_code::feature_order_mismatch,
        "packed dense operands from different plans were accepted");
}

void test_trivial_semantics() {
    cm::spmm_request request = request_fixture();
    request.m = 0u;
    request.sparse_nnz = 0u;
    require(cm::classify_trivial_operation(request)
            == cm::trivial_operation_kind::no_output,
        "M=0 should have no output work");

    request = request_fixture();
    request.alpha = cm::make_scalar(0.0f);
    require(cm::classify_trivial_operation(request)
            == cm::trivial_operation_kind::epilogue_only,
        "alpha=0 should skip sparse multiplication");
    cm::math_request bound;
    bound.operation = request;
    static int output = 0;
    bound.bindings.output = &output;
    require(static_cast<bool>(cm::validate_math_request(bound)),
        "alpha=0 incorrectly requires sparse or dense bindings");

    request = request_fixture();
    request.k = 0u;
    request.sparse_nnz = 0u;
    request.sparse_feature_order = canonical_order(0u);
    request.dense_feature_order = request.sparse_feature_order;
    require(cm::classify_trivial_operation(request)
            == cm::trivial_operation_kind::epilogue_only,
        "K=0 should use epilogue-only semantics");
}

} // namespace

int main() {
    static_assert(std::is_trivially_copyable<cm::execution_plan>::value,
        "execution plan serialization contract changed");
    test_signature_and_order_identity();
    test_packed_plan_identity_mismatch();
    test_binding_and_policy_validation();
    test_shape_and_enum_validation();
    test_bias_contract();
    test_explicit_gelu_semantics();
    test_trivial_semantics();
    std::cout << "mathOperationContractTest passed\n";
    return 0;
}
