#include <Cellerator/compute/math/planner.hh>
#include <Cellerator/types.cuh>

#include <cstdlib>
#include <iostream>

namespace cm = cellerator::compute::math;
namespace cr = cellerator::real;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cpMathPlannerTest: " << message << '\n';
        std::exit(1);
    }
}

enum class mock_behavior {
    native,
    compose_epilogue,
    reject_determinism,
    unimplemented
};

class MockBackend final : public cm::SpMMBackend {
public:
    MockBackend(cm::u64 identity, cm::u64 algorithm, cm::u64 kernel,
        cm::u64 workspace, mock_behavior behavior,
        cm::preprocessing_kind preprocessing = cm::preprocessing_kind::none)
        : identity_(identity), algorithm_(algorithm), kernel_(kernel),
          workspace_(workspace), behavior_(behavior),
          preprocessing_(preprocessing) {}

    cm::u64 identity() const noexcept override { return identity_; }
    const char *name() const noexcept override { return "planner-mock"; }

    cm::backend_capability query(
        const cm::spmm_request &request,
        const cm::DeviceCapabilities &) const noexcept override {
        if (behavior_ == mock_behavior::reject_determinism
            && request.determinism == cm::determinism_requirement::deterministic) {
            return {cm::capability_code::unsupported_determinism,
                cm::request_validation_code::ok,
                "mock backend is nondeterministic"};
        }
        if (behavior_ == mock_behavior::compose_epilogue
            && request.epilogue.kind != cm::epilogue_kind::none) {
            return {cm::capability_code::unsupported_epilogue,
                cm::request_validation_code::ok,
                "mock backend requires a composed epilogue"};
        }
        cm::backend_capability result;
        result.algorithm_identity = behavior_ == mock_behavior::unimplemented
            ? 0u : algorithm_;
        result.kernel_variant_identity = kernel_;
        result.workspace_bytes = workspace_;
        result.preprocessing = preprocessing_;
        result.epilogue_strategy = request.epilogue.kind == cm::epilogue_kind::none
            ? cm::epilogue_strategy_kind::none
            : cm::epilogue_strategy_kind::backend_fused;
        return result;
    }

    cm::backend_status prepare(cm::PreparedExecution *) noexcept override {
        return {};
    }
    cm::backend_status run(cm::PreparedExecution *) noexcept override {
        return {};
    }
    void release(cm::PreparedExecution *) noexcept override {}

private:
    cm::u64 identity_, algorithm_, kernel_, workspace_;
    mock_behavior behavior_;
    cm::preprocessing_kind preprocessing_;
};

cm::feature_order_identity canonical_order(cm::u32 count) {
    cm::feature_order_identity order;
    order.feature_count = count;
    order.feature_axis_identity_version = 1u;
    order.feature_axis_identity = 0x43504d4154483039ull;
    return order;
}

cm::math_request request_fixture(cm::epilogue_kind epilogue =
    cm::epilogue_kind::none) {
    static int sparse = 0, dense = 0, output = 0, bias = 0;
    cm::math_request request;
    request.operation.m = 8u;
    request.operation.k = 16u;
    request.operation.n = 32u;
    request.operation.sparse_nnz = 64u;
    request.operation.sparse_structure.identity_version = 1u;
    request.operation.sparse_structure.value = 0x09u;
    request.operation.dense_rhs_leading_dimension = 32u;
    request.operation.output_leading_dimension = 32u;
    request.operation.sparse_storage_type_code = cr::value_f32;
    request.operation.dense_storage_type_code = cr::value_f32;
    request.operation.output_storage_type_code = cr::value_f32;
    request.operation.compute_type_code = cr::value_f32;
    request.operation.accumulation_type_code = cr::value_f32;
    request.operation.alpha = cm::make_scalar(1.0f);
    request.operation.beta = cm::make_scalar(0.0f);
    request.operation.epilogue.kind = epilogue;
    if (epilogue == cm::epilogue_kind::bias_relu) {
        request.operation.epilogue.bias_type_code = cr::value_f32;
        request.operation.epilogue.bias_element_count = request.operation.n;
        request.bindings.bias = &bias;
    }
    request.operation.sparse_feature_order = canonical_order(16u);
    request.operation.dense_feature_order = request.operation.sparse_feature_order;
    request.bindings.sparse_matrix = &sparse;
    request.bindings.dense_rhs = &dense;
    request.bindings.output = &output;
    return request;
}

cm::DeviceCapabilities device_fixture() {
    cm::DeviceCapabilities device;
    device.device_ordinal = 0;
    device.compute_capability_major = 7;
    device.compute_capability_minor = 0;
    device.total_global_memory_bytes = 1ull << 30u;
    device.driver_version = 12000;
    device.runtime_version = 12000;
    device.toolkit_version = 12000;
    return device;
}

cm::DeviceFingerprint fingerprint_fixture() {
    cm::DeviceFingerprint fingerprint;
    fingerprint.device_ordinal = 0;
    fingerprint.compute_capability_major = 7;
    fingerprint.total_global_memory_bytes = 1ull << 30u;
    fingerprint.driver_version = 12000;
    fingerprint.runtime_version = 12000;
    fingerprint.toolkit_version = 12000;
    fingerprint.uuid[0] = 9u;
    return fingerprint;
}

cm::planner_status plan(
    const cm::math_request &request,
    const cm::SpMMBackendRegistry &registry,
    cm::planner_result *result,
    cm::plan_cache_lookup_hook cache = {},
    cm::planner_policy policy = {}) {
    static const cm::DeviceCapabilities device = device_fixture();
    static const cm::DeviceFingerprint fingerprint = fingerprint_fixture();
    return cm::plan_spmm(
        {&request, &device, &fingerprint, &registry, cache, policy}, result);
}

void test_registry_contract() {
    cm::SpMMBackendRegistry registry;
    MockBackend valid(10u, 20u, 30u, 0u, mock_behavior::native);
    MockBackend duplicate(10u, 21u, 31u, 0u, mock_behavior::native);
    MockBackend invalid(0u, 22u, 32u, 0u, mock_behavior::native);
    require(static_cast<bool>(registry.add(&valid)),
        "valid backend registration failed");
    require(registry.add(&duplicate).code
            == cm::backend_registration_code::duplicate_identity,
        "duplicate backend identity was accepted");
    require(registry.add(&invalid).code
            == cm::backend_registration_code::invalid_backend,
        "zero backend identity was accepted");
    require(registry.size() == 1u && registry.at(0u) == &valid,
        "registry lookup is inconsistent");
    require(static_cast<bool>(registry.remove(valid.identity()))
            && registry.size() == 0u,
        "registered backend removal failed");
}

void test_trivial_interception() {
    cm::SpMMBackendRegistry registry;
    cm::math_request request = request_fixture();
    request.operation.m = 0u;
    request.operation.sparse_nnz = 0u;
    request.bindings = {};
    cm::planner_result result;
    require(static_cast<bool>(plan(request, registry, &result)),
        "zero-output request was not intercepted");
    require(result.trivial == cm::trivial_operation_kind::no_output
            && result.candidate_count == 0u
            && result.selected_backend == nullptr,
        "zero-output request emitted a backend candidate");

    request = request_fixture();
    request.operation.sparse_nnz = 0u;
    request.bindings.sparse_matrix = nullptr;
    request.bindings.dense_rhs = nullptr;
    require(static_cast<bool>(plan(request, registry, &result)),
        "epilogue-only request was not intercepted");
    require(result.trivial == cm::trivial_operation_kind::epilogue_only
            && result.candidate_count == 0u,
        "epilogue-only request emitted a backend candidate");
}

void test_filtering_and_composition() {
    cm::SpMMBackendRegistry registry;
    MockBackend deterministic_reject(
        10u, 100u, 1000u, 0u, mock_behavior::reject_determinism);
    MockBackend workspace_reject(
        20u, 200u, 2000u, 4096u, mock_behavior::native);
    MockBackend unimplemented(
        30u, 300u, 3000u, 0u, mock_behavior::unimplemented);
    MockBackend composed(
        40u, 400u, 4000u, 0u, mock_behavior::compose_epilogue);
    require(registry.add(&deterministic_reject)
            && registry.add(&workspace_reject)
            && registry.add(&unimplemented)
            && registry.add(&composed),
        "filtering fixture registration failed");

    cm::math_request request = request_fixture(cm::epilogue_kind::bias_relu);
    request.operation.determinism = cm::determinism_requirement::deterministic;
    request.operation.workspace.kind = cm::workspace_policy_kind::caller_limit;
    request.operation.workspace.byte_limit = 1024u;
    cm::planner_result result;
    require(static_cast<bool>(plan(request, registry, &result)),
        "legal composed candidate was not selected");
    require(result.candidate_count == 1u
            && result.candidates[0].backend == &composed
            && result.candidates[0].origin
                == cm::candidate_origin::generic_epilogue_composed
            && result.plan.epilogue_strategy
                == cm::epilogue_strategy_kind::generic_unfused,
        "epilogue composition did not produce the sole legal candidate");
    require(result.diagnostics.rejection_count[
                static_cast<std::size_t>(
                    cm::capability_code::unsupported_determinism)] == 1u
            && result.diagnostics.rejection_count[
                static_cast<std::size_t>(
                    cm::capability_code::workspace_policy_rejected)] == 1u
            && result.diagnostics.malformed_candidate_count == 1u,
        "hard-filter diagnostics are incomplete");
}

struct cache_fixture {
    bool found = false;
    cm::execution_plan plan{};
};

bool cache_lookup(void *context, const cm::operation_signature &,
    cm::u64, cm::u64, cm::execution_plan *out) noexcept {
    const auto *fixture = static_cast<const cache_fixture *>(context);
    if (!fixture->found) return false;
    *out = fixture->plan;
    return true;
}

void test_authoritative_selection_and_cache() {
    MockBackend cheap(10u, 100u, 1000u, 0u, mock_behavior::native);
    MockBackend expensive(20u, 200u, 2000u, 8192u, mock_behavior::native,
        cm::preprocessing_kind::backend_preprocess);
    cm::math_request request = request_fixture(cm::epilogue_kind::relu);

    cm::SpMMBackendRegistry expensive_only;
    require(static_cast<bool>(expensive_only.add(&expensive)),
        "cache fixture registration failed");
    cm::planner_result expensive_result;
    require(static_cast<bool>(plan(request, expensive_only, &expensive_result)),
        "cache fixture plan failed");

    cm::SpMMBackendRegistry registry;
    require(registry.add(&expensive) && registry.add(&cheap),
        "selection fixture registration failed");
    cm::planner_result result;
    require(static_cast<bool>(plan(request, registry, &result)),
        "uncached selection failed");
    require(result.selected_backend == &cheap,
        "cheap structural ordering was not authoritative");

    cm::planner_policy shortlist;
    shortlist.candidate_limit = 1u;
    require(static_cast<bool>(plan(request, registry, &result, {}, shortlist)),
        "bounded shortlist selection failed");
    require(result.candidate_count == 1u && result.selected_backend == &cheap
            && result.diagnostics.structurally_pruned_count == 1u,
        "shortlist pruning depended on backend registration order");

    cache_fixture cached{true, expensive_result.plan};
    require(static_cast<bool>(plan(request, registry, &result,
        {&cached, cache_lookup})), "cached selection failed");
    require(result.cache_state == cm::cache_lookup_state::hit
            && result.selected_backend == &expensive,
        "legal cache hit did not override structural ordering");

    cached.plan.workspace_bytes += 1u;
    require(static_cast<bool>(plan(request, registry, &result,
        {&cached, cache_lookup})), "stale cache fallback failed");
    require(result.cache_state == cm::cache_lookup_state::stale_or_illegal
            && result.selected_backend == &cheap,
        "stale cache entry bypassed authoritative legal selection");
}

void test_no_legal_candidate() {
    cm::SpMMBackendRegistry registry;
    MockBackend unavailable(
        10u, 100u, 1000u, 0u, mock_behavior::unimplemented);
    require(static_cast<bool>(registry.add(&unavailable)),
        "unavailable fixture registration failed");
    cm::planner_result result;
    const cm::planner_status status = plan(request_fixture(), registry, &result);
    require(status.code == cm::planner_status_code::no_legal_candidate
            && status.capability == cm::capability_code::backend_unavailable
            && result.candidate_count == 0u,
        "unimplemented backend leaked into the legal candidate set");
}

void test_invalid_input_clears_output() {
    cm::SpMMBackendRegistry registry;
    cm::math_request request = request_fixture();
    request.bindings.output = nullptr;
    cm::planner_result result;
    result.candidate_count = 7u;
    const cm::planner_status status = plan(request, registry, &result);
    require(status.code == cm::planner_status_code::invalid_request
            && status.validation == cm::request_validation_code::missing_binding
            && result.candidate_count == 0u,
        "invalid request retained a stale planner result");
}

} // namespace

int main() {
    test_registry_contract();
    test_trivial_interception();
    test_filtering_and_composition();
    test_authoritative_selection_and_cache();
    test_no_legal_candidate();
    test_invalid_input_clears_output();
    std::cout << "cpMathPlannerTest passed\n";
    return 0;
}
