#include "learning/ste_chain.hh"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cgs {
namespace {

void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template <class T> struct Device {
    T* p = nullptr;
    explicit Device(std::size_t n) {
        if (n) check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "cudaMalloc");
    }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
template <class T> void upload(Device<T>& d, const std::vector<T>& v, cudaStream_t stream) {
    check(cudaMemcpyAsync(d.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
}
template <class T> void download(std::vector<T>& v, Device<T>& d, cudaStream_t stream) {
    check(cudaMemcpyAsync(v.data(), d.p, v.size() * sizeof(T), cudaMemcpyDeviceToHost, stream), "download");
}
std::uint32_t float_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}
bool close(float a, float b) { return std::fabs(a - b) <= 2e-6f * (1.0f + std::fabs(b)); }

} // namespace

bool run_ste_tests(cudaStream_t stream, std::string* error) {
    try {
        constexpr std::uint64_t B = 35, C = 3, W = (B + 31) / 32, N = B * C;
        constexpr float temperature = 2.0f, threshold = 0.5f, rate = 0.3f;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> scores(N), dq(N), host_p(N), host_q(N), host_ds(N), expected_ds(N);
        std::vector<std::uint8_t> host_admit(N);
        std::vector<std::uint32_t> candidate(C * W, 0), primal(C * W), derivative(C * W);
        for (std::uint64_t cell = 0; cell < B; ++cell) {
            for (std::uint64_t c = 0; c < C; ++c) {
                const auto i = cell * C + c;
                const bool included = (cell + c) % 7 != 0;
                if (included) candidate[c * W + cell / 32] |= 1u << (cell % 32);
                scores[i] = included ? (c == 0 ? -0.01f : c == 1 ? 0.0f : 0.01f) : nan;
                dq[i] = included ? -4.0f : nan;
            }
        }
        Device<float> d_scores(N), d_dq(N), d_p(N), d_q(N), d_ds(N), d_reduced(C);
        Device<std::uint8_t> d_admit(N);
        Device<std::uint32_t> d_candidate(C * W), d_primal(C * W), d_derivative(C * W), d_status(1);
        upload(d_scores, scores, stream);
        upload(d_dq, dq, stream);
        upload(d_candidate, candidate, stream);
        check(cudaMemsetAsync(d_status.p, 0, sizeof(std::uint32_t), stream), "clear status");

        GateRevisions rev{};
        rev.temperature_bits = float_bits(temperature);
        rev.threshold_bits = float_bits(threshold);
        SteBuffers buffers{d_scores.p, d_candidate.p, d_p.p, d_q.p, d_admit.p,
                           d_primal.p, d_derivative.p, d_status.p};
        SteSnapshot snap;
        check(ste_forward(B, C, temperature, threshold, rev, buffers, &snap, stream), "forward");
        check(ste_backward(snap, rev, d_p.p, d_derivative.p, d_dq.p, d_ds.p, stream), "backward");
        check(ste_reduce_per_condition(B, C, d_ds.p, d_reduced.p, d_status.p, stream), "reduce");
        download(host_p, d_p, stream);
        download(host_q, d_q, stream);
        download(host_ds, d_ds, stream);
        download(host_admit, d_admit, stream);
        download(primal, d_primal, stream);
        download(derivative, d_derivative, stream);
        std::vector<float> reduced(C);
        download(reduced, d_reduced, stream);
        std::uint32_t status = ~0u;
        check(cudaMemcpyAsync(&status, d_status.p, sizeof(status), cudaMemcpyDeviceToHost, stream), "status");
        check(cudaStreamSynchronize(stream), "synchronize");
        require(status == 0, "noncandidate poison must not set status");
        require(derivative == candidate, "derivative membership or tail mismatch");
        std::vector<std::uint32_t> expected_primal(C * W, 0);
        std::vector<float> expected_reduced(C, 0);
        bool saw_dormant = false, saw_equality = false;
        for (std::uint64_t cell = 0; cell < B; ++cell) {
            for (std::uint64_t c = 0; c < C; ++c) {
                const auto i = cell * C + c;
                const bool included = ((candidate[c * W + cell / 32] >> (cell % 32)) & 1u) != 0;
                const float p = included ? static_cast<float>(1.0 / (1.0 + std::exp(-double(scores[i]) / temperature))) : 0.0f;
                const bool active = included && p >= threshold;
                const float ds = included ? dq[i] * (p * (1.0f - p) / temperature) : 0.0f;
                expected_ds[i] = ds;
                expected_reduced[c] += ds;
                if (active) expected_primal[c * W + cell / 32] |= 1u << (cell % 32);
                require(close(host_p[i], p) && close(host_q[i], active ? p : 0.0f), "forward numerical mismatch");
                require(host_admit[i] == std::uint8_t(active), "threshold admission mismatch");
                require(close(host_ds[i], ds), "declared STE estimator mismatch");
                if (included && !active && std::fabs(host_ds[i]) > 0) saw_dormant = true;
                if (included && scores[i] == 0 && active) saw_equality = true;
            }
        }
        require(primal == expected_primal, "primal membership or tail mismatch");
        require(saw_dormant && saw_equality, "dormant gradient or threshold equality not exercised");
        for (std::uint64_t c = 0; c < C; ++c)
            require(close(reduced[c], expected_reduced[c]), "condition reduction mismatch");

        // Every revision class invalidates the saved forward independently.
        using Member64 = std::uint64_t GateRevisions::*;
        const Member64 fields[] = {&GateRevisions::structure_epoch, &GateRevisions::mapping_revision,
            &GateRevisions::accessibility_generation, &GateRevisions::score_generation,
            &GateRevisions::parameter_revision, &GateRevisions::teacher_revision,
            &GateRevisions::temporal_generation, &GateRevisions::primal_generation,
            &GateRevisions::derivative_generation};
        for (auto field : fields) {
            auto stale = rev;
            ++(stale.*field);
            require(ste_backward(snap, stale, d_p.p, d_derivative.p, d_dq.p, d_ds.p, stream) == cudaErrorInvalidValue,
                    "changed revision must reject backward");
        }
        auto stale = rev;
        stale.threshold_bits ^= 1u;
        require(ste_backward(snap, stale, d_p.p, d_derivative.p, d_dq.p, d_ds.p, stream) == cudaErrorInvalidValue,
                "changed threshold must reject backward");
        stale = rev;
        stale.temperature_bits ^= 1u;
        require(ste_backward(snap, stale, d_p.p, d_derivative.p, d_dq.p, d_ds.p, stream) == cudaErrorInvalidValue,
                "changed temperature must reject backward");
        require(ste_backward(snap, rev, d_q.p, d_derivative.p, d_dq.p, d_ds.p, stream) == cudaErrorInvalidValue,
                "changed probability buffer must reject backward");

        check(ste_sgd_update(N, d_scores.p, d_ds.p, rate, &snap, stream), "SGD");
        require(!snap.valid, "SGD must invalidate saved forward");
        require(ste_backward(snap, rev, d_p.p, d_derivative.p, d_dq.p, d_ds.p, stream) == cudaErrorInvalidValue,
                "updated score must reject stale forward");
        std::vector<float> updated(N);
        download(updated, d_scores, stream);
        check(cudaStreamSynchronize(stream), "sync updated");
        for (std::uint64_t i = 0; i < N; ++i)
            if (std::isfinite(scores[i])) require(close(updated[i], scores[i] - rate * expected_ds[i]), "SGD mismatch");
        ++rev.score_generation;
        check(ste_forward(B, C, temperature, threshold, rev, buffers, &snap, stream), "forward after SGD");
        download(host_admit, d_admit, stream);
        download(host_p, d_p, stream);
        check(cudaStreamSynchronize(stream), "sync crossing");
        require(host_admit[C] == 1, "SGD must cross threshold for dormant candidate");
        const std::size_t stable_active = C + 2;
        require(host_admit[stable_active] == 1 &&
                host_p[stable_active] > static_cast<float>(1.0 / (1.0 + std::exp(-0.01 / temperature))),
                "saved gate value was not refreshed when membership stayed active");

        // Null candidate plane means all logical pairs, including tail bits;
        // out-of-range bits must remain zero. Exactly 0.5 must be admitted.
        std::vector<float> zeros(N, 0.0f);
        upload(d_scores, zeros, stream);
        buffers.candidates = nullptr;
        ++rev.score_generation;
        ++rev.derivative_generation;
        check(ste_forward(B, C, temperature, threshold, rev, buffers, &snap, stream), "all-candidates forward");
        download(primal, d_primal, stream);
        download(derivative, d_derivative, stream);
        check(cudaStreamSynchronize(stream), "sync all candidates");
        for (std::uint64_t c = 0; c < C; ++c) {
            require(primal[c * W] == 0xffffffffu && derivative[c * W] == 0xffffffffu,
                    "all-candidates full word mismatch");
            require(primal[c * W + 1] == 7u && derivative[c * W + 1] == 7u,
                    "all-candidates tail mismatch");
        }

        zeros[0] = nan;
        upload(d_scores, zeros, stream);
        ++rev.score_generation;
        check(cudaMemsetAsync(d_status.p, 0, sizeof(std::uint32_t), stream), "clear nonfinite status");
        Device<float> d_guarded_parameters(N);
        std::vector<float> guarded_parameters(N, 1.0f);
        upload(d_guarded_parameters, guarded_parameters, stream);
        check(ste_forward(B, C, temperature, threshold, rev, buffers, &snap, stream), "nonfinite forward");
        // These phases are queued before the host can observe status. No
        // parameter may change, including finite unrelated parameters.
        check(ste_backward(snap, rev, d_p.p, d_derivative.p, d_dq.p, d_ds.p, stream), "nonfinite backward");
        check(ste_reduce_per_condition(B, C, d_ds.p, d_reduced.p, d_status.p, stream), "nonfinite reduction");
        check(ste_sgd_update(N, d_guarded_parameters.p, d_ds.p, rate, &snap, stream), "guarded failed update");
        std::vector<float> guarded_after(N);
        download(guarded_after, d_guarded_parameters, stream);
        check(cudaMemcpyAsync(&status, d_status.p, sizeof(status), cudaMemcpyDeviceToHost, stream), "nonfinite status");
        check(cudaStreamSynchronize(stream), "sync nonfinite");
        require((status & status_nonfinite) != 0, "candidate nonfinite was not reported");
        require(guarded_after == guarded_parameters, "failed forward committed parameters");

        // A nonfinite gradient must also abort the entire update even when the
        // forward succeeds and another parameter has a finite gradient.
        zeros[0] = 0.0f;
        upload(d_scores, zeros, stream);
        std::vector<float> failing_dq(N, 1.0f);
        failing_dq[0] = nan;
        upload(d_dq, failing_dq, stream);
        ++rev.score_generation;
        check(cudaMemsetAsync(d_status.p, 0, sizeof(std::uint32_t), stream), "clear gradient status");
        check(ste_forward(B, C, temperature, threshold, rev, buffers, &snap, stream), "gradient guard forward");
        check(ste_backward(snap, rev, d_p.p, d_derivative.p, d_dq.p, d_ds.p, stream), "gradient guard backward");
        check(ste_sgd_update(N, d_guarded_parameters.p, d_ds.p, rate, &snap, stream), "guarded gradient update");
        download(guarded_after, d_guarded_parameters, stream);
        check(cudaMemcpyAsync(&status, d_status.p, sizeof(status), cudaMemcpyDeviceToHost, stream), "gradient status");
        check(cudaStreamSynchronize(stream), "sync gradient guard");
        require((status & status_nonfinite) != 0 && guarded_after == guarded_parameters,
                "nonfinite gradient committed a parameter");

        std::vector<float> overflow_gradient(N, 0.0f);
        overflow_gradient[0] = std::numeric_limits<float>::max();
        overflow_gradient[1] = 1.0f;
        upload(d_ds, overflow_gradient, stream);
        check(cudaMemsetAsync(d_status.p, 0, sizeof(std::uint32_t), stream), "clear update status");
        ++rev.score_generation;
        check(ste_forward(B, C, temperature, threshold, rev, buffers, &snap, stream), "update guard forward");
        check(ste_sgd_update(N, d_guarded_parameters.p, d_ds.p, 2.0f, &snap, stream), "overflow guarded update");
        download(guarded_after, d_guarded_parameters, stream);
        check(cudaMemcpyAsync(&status, d_status.p, sizeof(status), cudaMemcpyDeviceToHost, stream), "overflow status");
        check(cudaStreamSynchronize(stream), "sync overflow guard");
        require((status & status_nonfinite) != 0 && guarded_after == guarded_parameters,
                "overflowed candidate committed a partial update");

        // Exercise the adapter with a native body-provided dq. Shared-score
        // mode reduces all cell derivatives before the guarded update.
        std::vector<float> unit_dq(N, 1.0f), shared_parameters(C, 0.0f);
        upload(d_dq, unit_dq, stream);
        Device<float> d_shared(C);
        upload(d_shared, shared_parameters, stream);
        check(cudaMemsetAsync(d_status.p, 0, sizeof(std::uint32_t), stream), "clear chain status");
        SteChainBuffers chain_buffers{buffers, d_dq.p, d_ds.p, d_reduced.p, d_shared.p,
                                       SteParameters::shared_condition};
        chain_buffers.shared_scores = d_scores.p;
        SteChain chain;
        ++rev.score_generation;
        check(ste_chain_forward(B, C, temperature, threshold, rev, chain_buffers,
                                &chain, {}, stream), "chain forward");
        ResultView body_result{};
        body_result.dq = d_dq.p;
        WorkspaceView body_workspace{};
        body_workspace.status = d_status.p;
        auto stale_chain = rev;
        ++stale_chain.mapping_revision;
        require(ste_chain_backward_update(&chain, stale_chain, body_result, body_workspace,
                                          rate, {}, stream) == cudaErrorInvalidValue,
                "chain accepted stale mapping");
        check(ste_chain_backward_update(&chain, rev, body_result, body_workspace,
                                        rate, {}, stream), "chain update");
        require(!chain.snapshot.valid, "chain did not invalidate snapshot");
        download(shared_parameters, d_shared, stream);
        check(cudaMemcpyAsync(&status, d_status.p, sizeof(status), cudaMemcpyDeviceToHost, stream), "chain status");
        check(cudaStreamSynchronize(stream), "sync shared chain");
        require(status == 0, "valid chain reported failure");
        for (float p : shared_parameters)
            require(close(p, -rate * float(B) * 0.25f / temperature), "shared-chain SGD mismatch");

        // Per-cell parameters use ds without the shared-condition reduction.
        upload(d_scores, zeros, stream);
        check(cudaMemsetAsync(d_status.p, 0, sizeof(std::uint32_t), stream), "clear cell chain status");
        chain_buffers.parameters = d_scores.p;
        chain_buffers.parameterization = SteParameters::per_cell_condition;
        ++rev.score_generation;
        check(ste_chain_forward(B, C, temperature, threshold, rev, chain_buffers,
                                &chain, {}, stream), "cell chain forward");
        check(ste_chain_backward_update(&chain, rev, body_result, body_workspace,
                                        rate, {}, stream), "cell chain update");
        download(updated, d_scores, stream);
        check(cudaStreamSynchronize(stream), "sync cell chain");
        for (float p : updated)
            require(close(p, -rate * 0.25f / temperature), "per-cell-chain SGD mismatch");

        SteSnapshot empty;
        check(ste_forward(0, 3, temperature, threshold, rev, {}, &empty, stream), "empty forward");
        check(ste_backward(empty, rev, nullptr, nullptr, nullptr, nullptr, stream), "empty backward");
        require(ste_forward(B, C, 0.0f, threshold, rev, buffers, &empty, stream) == cudaErrorInvalidValue,
                "zero temperature accepted");
        if (error) error->clear();
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace cgs
