#include <Cellerator/runtime/runtime.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

using cellerator::runtime::execution_session;
using cellerator::runtime::execution_session_options;
using cellerator::runtime::persistent_lifetime;
using cellerator::runtime::session_cache_key;
using cellerator::runtime::session_cache_kind;
using cellerator::runtime::session_status;

int require(bool condition, const char *message) {
    if (condition) return 0;
    std::cerr << "execution session test failed: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    cudaStream_t external[2]{};
    if (cudaStreamCreateWithFlags(&external[0], cudaStreamNonBlocking) != cudaSuccess
        || cudaStreamCreateWithFlags(&external[1], cudaStreamNonBlocking) != cudaSuccess) {
        std::cerr << "execution session test failed: external stream creation\n";
        return 1;
    }

    execution_session session{};
    execution_session_options options{};
    options.device = 0;
    options.external_streams = external;
    options.external_stream_count = 2;
    options.owned_stream_count = 0;
    if (require(cellerator::runtime::init_session(&session, options) == session_status::success,
                "session initialization")) return 1;
    if (require(session.performance.compute_major >= 7 && session.performance.warp_size == 32,
                "device performance class")) return 1;

    void *structure = nullptr, *plan = nullptr, *graph = nullptr;
    void *workspace0 = nullptr, *workspace1 = nullptr;
    if (require(cellerator::runtime::reserve_persistent(
                    &session, persistent_lifetime::structure, 256, &structure)
                    == session_status::success,
                "structure reserve")
        || require(cellerator::runtime::reserve_persistent(
                       &session, persistent_lifetime::plan, 512, &plan)
                       == session_status::success,
                   "plan reserve")
        || require(cellerator::runtime::reserve_persistent(
                       &session, persistent_lifetime::graph_stable, 1024, &graph)
                       == session_status::success,
                   "graph reserve")
        || require(cellerator::runtime::reserve_transient(&session, 0, 2048, &workspace0)
                       == session_status::success,
                   "stream zero workspace reserve")
        || require(cellerator::runtime::reserve_transient(&session, 1, 4096, &workspace1)
                       == session_status::success,
                   "stream one workspace reserve")) return 1;

    int plan_state = 11, projection_state = 12, transform_state = 13;
    if (require(cellerator::runtime::insert_session_cache(
                    &session, session_cache_kind::plan, {1, 1}, &plan_state, 4, 9)
                    == session_status::success,
                "plan cache insert")
        || require(cellerator::runtime::insert_session_cache(
                       &session, session_cache_kind::projection, {2, 2},
                       &projection_state, 4, 9) == session_status::success,
                   "projection cache insert")
        || require(cellerator::runtime::insert_session_cache(
                       &session, session_cache_kind::order_transform, {3, 3},
                       &transform_state, 4, 9) == session_status::success,
                   "order transform cache insert")) return 1;

    if (require(cellerator::runtime::prepare_stream_libraries(&session, 0)
                    == session_status::success,
                "stream zero handles")
        || require(cellerator::runtime::prepare_stream_libraries(&session, 1)
                       == session_status::success,
                   "stream one handles")
        || require(cellerator::runtime::seal_session(&session) == session_status::success,
                   "session seal")) return 1;

    const auto accounting_before = session.accounting;
    const auto binding0 = cellerator::runtime::bind_launch(&session, 0, 1024);
    const auto binding1 = cellerator::runtime::bind_launch(&session, 1, 3072);
    if (require(binding0.status == session_status::success
                    && binding1.status == session_status::success,
                "two stream launch bindings")
        || require(binding0.execution.stream == external[0]
                       && binding1.execution.stream == external[1],
                   "caller stream preservation")
        || require(binding0.workspace == workspace0 && binding1.workspace == workspace1,
                   "per-stream workspace isolation")
        || require(binding0.cublas != nullptr && binding0.cusparse != nullptr
                       && binding1.cublas != nullptr && binding1.cusparse != nullptr,
                   "prepared library bindings")
        || require(session.accounting.structure.allocation_count
                       == accounting_before.structure.allocation_count
                       && session.accounting.plan.allocation_count
                           == accounting_before.plan.allocation_count
                       && session.accounting.transient.allocation_count
                           == accounting_before.transient.allocation_count
                       && session.accounting.device_query_count
                           == accounting_before.device_query_count
                       && session.accounting.synchronization_count
                           == accounting_before.synchronization_count,
                   "steady launch has no allocation, discovery, or synchronization")) return 1;

    const auto exhausted = cellerator::runtime::bind_launch(&session, 0, 4096);
    if (require(exhausted.status == session_status::workspace_exhausted,
                "workspace exhaustion is explicit")) return 1;
    void *late = reinterpret_cast<void *>(std::uintptr_t{1});
    if (require(cellerator::runtime::reserve_transient(&session, 0, 8192, &late)
                    == session_status::invalid_state && late == nullptr,
                "sealed session cannot grow workspace")
        || require(cellerator::runtime::graph_stable_address(session, graph, 1024),
                   "graph-stable address classification")
        || require(!cellerator::runtime::graph_stable_address(session, workspace0, 1),
                   "transient address is not graph-stable")) return 1;

    const auto *cached = cellerator::runtime::find_session_cache(
        session, session_cache_kind::projection, session_cache_key{2, 2});
    const auto fleet = cellerator::runtime::single_device_fleet(session);
    if (require(cached != nullptr && cached->state == &projection_state
                    && cached->structure_epoch == 4 && cached->generation == 9,
                "cache identity and generations")
        || require(fleet.device_count == 1 && fleet.devices == &session.performance,
                   "single-device fleet view")) return 1;

    cellerator::runtime::clear_session(&session);
    if (require(cudaStreamQuery(external[0]) == cudaSuccess
                    && cudaStreamQuery(external[1]) == cudaSuccess,
                "teardown preserves external streams")) return 1;
    cudaStreamDestroy(external[0]);
    cudaStreamDestroy(external[1]);

    std::cout << "celleratorExecutionSessionTest passed"
              << " structure_bytes=256 plan_bytes=512 graph_bytes=1024"
              << " transient_bytes=6144 launch_binds=2\n";
    return 0;
}
