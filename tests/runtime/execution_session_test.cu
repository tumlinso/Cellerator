#include <Cellerator/runtime/runtime.cuh>
#include <Cellerator/execution/launch_bindings.hh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

namespace execution = cellerator::execution;
namespace runtime = cellerator::runtime;

using runtime::execution_session;
using runtime::execution_session_options;
using runtime::persistent_lifetime;
using runtime::session_cache_key;
using runtime::session_cache_kind;
using runtime::session_status;

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
    if (require(runtime::init_session(&session, options) == session_status::success,
                "session initialization")) return 1;
    if (require(session.performance.compute_major >= 7 && session.performance.warp_size == 32,
                "device performance class")) return 1;

    void *structure = nullptr, *structure_larger = nullptr, *plan = nullptr;
    void *graph = nullptr, *graph_second = nullptr;
    void *workspace0 = nullptr, *workspace1 = nullptr;
    if (require(runtime::reserve_persistent(
                    &session, persistent_lifetime::structure, 256, &structure)
                    == session_status::success,
                "structure reserve")
        || require(runtime::reserve_persistent(
                       &session, persistent_lifetime::structure, 4096,
                       &structure_larger) == session_status::success,
                   "second structure reserve")
        || require(runtime::reserve_persistent(
                       &session, persistent_lifetime::plan, 512, &plan)
                       == session_status::success,
                   "plan reserve")
        || require(runtime::reserve_persistent(
                       &session, persistent_lifetime::graph_stable, 1024, &graph)
                       == session_status::success,
                   "graph reserve")
        || require(runtime::reserve_persistent(
                       &session, persistent_lifetime::graph_stable, 128,
                       &graph_second) == session_status::success,
                   "second graph reserve")
        || require(runtime::reserve_transient(&session, 0, 2048, &workspace0)
                       == session_status::success,
                   "stream zero workspace reserve")
        || require(runtime::reserve_transient(&session, 1, 4096, &workspace1)
                       == session_status::success,
                   "stream one workspace reserve")) return 1;

    if (require(structure != structure_larger && structure != plan
                    && structure != graph && plan != graph
                    && graph != graph_second,
                "persistent allocations must not alias")) return 1;
    const std::uint32_t sentinel = 0x51a7c0deu;
    if (require(cudaMemcpy(structure, &sentinel, sizeof(sentinel),
                    cudaMemcpyHostToDevice) == cudaSuccess,
                "write first persistent allocation")) return 1;
    std::uint32_t recovered = 0u;
    if (require(cudaMemcpy(&recovered, structure, sizeof(recovered),
                    cudaMemcpyDeviceToHost) == cudaSuccess
                    && recovered == sentinel,
                "later persistent allocation invalidated earlier pointer")) return 1;

    while (session.persistent_allocation_count
           < runtime::execution_session_max_persistent_allocations) {
        void *extra = nullptr;
        if (require(runtime::reserve_persistent(
                        &session, persistent_lifetime::plan, 1u, &extra)
                        == session_status::success && extra != nullptr,
                    "fill persistent allocation table")) return 1;
    }
    void *overflow = reinterpret_cast<void *>(std::uintptr_t{1});
    if (require(runtime::reserve_persistent(
                    &session, persistent_lifetime::plan, 1u, &overflow)
                    == session_status::capacity_exceeded && overflow == nullptr,
                "persistent allocation table exhaustion is explicit")) return 1;
    if (require(session.accounting.structure.current_bytes == 4352u
                    && session.accounting.structure.high_water_bytes == 4352u
                    && session.accounting.structure.allocation_count == 2u,
                "structure allocation accounting")
        || require(session.accounting.graph_stable.current_bytes == 1152u
                       && session.accounting.graph_stable.high_water_bytes == 1152u
                       && session.accounting.graph_stable.allocation_count == 2u,
                   "graph allocation accounting")
        || require(session.accounting.structure.allocation_count
                       + session.accounting.plan.allocation_count
                       + session.accounting.graph_stable.allocation_count
                       == runtime::execution_session_max_persistent_allocations,
                   "persistent allocation count accounting")) return 1;

    int plan_state = 11, projection_state = 12, transform_state = 13;
    if (require(runtime::insert_session_cache(
                    &session, session_cache_kind::plan, {1, 1}, &plan_state, 4, 9)
                    == session_status::success,
                "plan cache insert")
        || require(runtime::insert_session_cache(
                       &session, session_cache_kind::projection, {2, 2},
                       &projection_state, 4, 9) == session_status::success,
                   "projection cache insert")
        || require(runtime::insert_session_cache(
                       &session, session_cache_kind::order_transform, {3, 3},
                       &transform_state, 4, 9) == session_status::success,
                   "order transform cache insert")) return 1;

    if (require(runtime::prepare_stream_libraries(&session, 0)
                    == session_status::success,
                "stream zero handles")
        || require(runtime::prepare_stream_libraries(&session, 1)
                       == session_status::success,
                   "stream one handles")
        || require(runtime::seal_session(&session) == session_status::success,
                   "session seal")) return 1;

    const auto accounting_before = session.accounting;
    const auto binding0 = runtime::bind_launch(&session, 0, 1024);
    const auto binding1 = runtime::bind_launch(&session, 1, 3072);
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

    execution::relation_structure alignment_relation{};
    alignment_relation.identity = {91u, 1u};
    alignment_relation.epoch = {7u};
    alignment_relation.source_axis = {{1u, 1u}, {2u, 1u}, {3u, 1u}, {4u, 1u}};
    alignment_relation.destination_axis = alignment_relation.source_axis;
    alignment_relation.projections = {92u, 1u};
    execution::prepared_binding_contract alignment_contract{};
    alignment_contract.structures[0] = {
        alignment_relation.identity, alignment_relation.epoch};
    alignment_contract.structure_count = 1u;
    alignment_contract.workspace = {8u, 16u, 0u};
    execution::launch_bindings misaligned{};
    misaligned.structures = &alignment_relation;
    misaligned.structure_count = 1u;
    misaligned.stream = {external[0], 0, 0u};
    misaligned.workspace = {
        static_cast<void *>(static_cast<unsigned char *>(workspace0) + 1u),
        8u, {execution::residency_kind::device, {}, 0, 0u}};
    if (require(execution::validate_launch_bindings(
                    alignment_contract, misaligned)
                    == execution::binding_validation_code::insufficient_workspace,
                "misaligned launch workspace was accepted")) return 1;

    const auto exhausted = runtime::bind_launch(&session, 0, 4096);
    if (require(exhausted.status == session_status::workspace_exhausted,
                "workspace exhaustion is explicit")) return 1;
    void *late = reinterpret_cast<void *>(std::uintptr_t{1});
    if (require(runtime::reserve_transient(&session, 0, 8192, &late)
                    == session_status::invalid_state && late == nullptr,
                "sealed session cannot grow workspace")
        || require(runtime::reserve_persistent(
                       &session, persistent_lifetime::plan, 8u, &late)
                       == session_status::invalid_state && late == nullptr,
                   "sealed session cannot add persistent allocation")
        || require(runtime::graph_stable_address(session, graph, 1024),
                   "graph-stable address classification")
        || require(runtime::graph_stable_address(
                       session, static_cast<unsigned char *>(graph) + 32u, 64u),
                   "graph-stable subrange classification")
        || require(runtime::graph_stable_address(
                       session, graph_second, 128u),
                   "second graph-stable allocation classification")
        || require(!runtime::graph_stable_address(session, structure, 1u),
                   "structure allocation is not graph-stable")
        || require(!runtime::graph_stable_address(session, plan, 1u),
                   "plan allocation is not graph-stable")
        || require(!runtime::graph_stable_address(session, workspace0, 1),
                   "transient address is not graph-stable")) return 1;

    const auto *cached = runtime::find_session_cache(
        session, session_cache_kind::projection, session_cache_key{2, 2});
    const auto fleet = runtime::single_device_fleet(session);
    if (require(cached != nullptr && cached->state == &projection_state
                    && cached->structure_epoch == 4 && cached->generation == 9,
                "cache identity and generations")
        || require(fleet.device_count == 1 && fleet.devices == &session.performance,
                   "single-device fleet view")) return 1;

    runtime::clear_session(&session);
    if (require(!session.initialized && session.persistent_allocation_count == 0u
                    && session.accounting.structure.current_bytes == 0u
                    && session.accounting.plan.current_bytes == 0u
                    && session.accounting.graph_stable.current_bytes == 0u,
                "session clear did not reset persistent allocation records")
        || require(cudaStreamQuery(external[0]) == cudaSuccess
                    && cudaStreamQuery(external[1]) == cudaSuccess,
                "teardown preserves external streams")) return 1;
    runtime::clear_session(&session);
    cudaStreamDestroy(external[0]);
    cudaStreamDestroy(external[1]);

    std::cout << "celleratorExecutionSessionTest passed"
              << " persistent_allocations="
              << runtime::execution_session_max_persistent_allocations
              << " transient_bytes=6144 launch_binds=2\n";
    return 0;
}
