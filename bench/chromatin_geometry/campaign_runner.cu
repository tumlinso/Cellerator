#include "campaign_runner.hh"
#include "device_fixture.hh"
#include "teacher_fixture.hh"
#include "backends/conditions.hh"
#include "backends/descriptor_debug.hh"
#include "backends/grouped_linear_library.hh"
#include "data/replay.hh"
#include "learning/model_update.hh"
#include "learning/ste_chain.hh"
#include "reference/reference.hh"
#include "reference/teacher_reference.hh"
#include "workloads.hh"
#include "grouping_probe.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <openssl/evp.h>
#include <type_traits>
#include <tuple>
#include <memory>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace cgs {
namespace {
using Clock = std::chrono::steady_clock;
constexpr float learning_rate = 0.01f;

__global__ void merge_condition_status(const std::uint32_t* source,
                                       std::uint32_t* destination) {
    if (!blockIdx.x && !threadIdx.x) atomicOr(destination, *source);
}

struct Stream {
    cudaStream_t value{};
    Stream() { cuda_check(cudaStreamCreate(&value), "create campaign stream"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};

enum Phase : std::size_t { condition_phase, clear_phase, gate_phase, prepare_phase,
                           forward_phase, backward_phase, score_phase, model_phase, phase_count };
struct PhaseRange {
    cudaEvent_t begin = nullptr, end = nullptr;
    Clock::time_point host_start{};
    double host_ms = 0.0;
    bool recorded = false;
    PhaseRange() {
        cuda_check(cudaEventCreate(&begin), "create phase event");
        cuda_check(cudaEventCreate(&end), "create phase event");
    }
    ~PhaseRange() { if (begin) cudaEventDestroy(begin); if (end) cudaEventDestroy(end); }
    void start(cudaStream_t stream) {
        recorded = true; cuda_check(cudaEventRecord(begin, stream), "phase begin");
        host_start = Clock::now();
    }
    void stop(cudaStream_t stream) {
        host_ms = std::chrono::duration<double, std::milli>(Clock::now() - host_start).count();
        cuda_check(cudaEventRecord(end, stream), "phase end");
    }
    double device_ms() const {
        if (!recorded) return 0.0;
        float value = 0; cuda_check(cudaEventElapsedTime(&value, begin, end), "phase duration");
        return value;
    }
};
struct PhaseProbe {
    std::array<PhaseRange, phase_count> phases;
    PhaseRange teacher_fit, teacher_harden, teacher_circuit;
    bool teacher_recorded = false;
    teacher::ChainEvents teacher_events() {
        teacher_recorded = true;
        teacher::ChainEvents result{};
        result.training = {teacher_fit.begin, teacher_fit.end};
        result.hardening = {teacher_harden.begin, teacher_harden.end};
        result.circuit = {teacher_circuit.begin, teacher_circuit.end};
        teacher_fit.recorded = teacher_harden.recorded = teacher_circuit.recorded = true;
        return result;
    }
};

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

Backend parse_backend(const std::string& name) {
    if (name == "scalar") return Backend::scalar;
    if (name == "bitmap") return Backend::bitmap;
    if (name == "direct") return Backend::direct;
    if (name == "cohort-direct") return Backend::cohort_direct;
    if (name == "cohort-direct-two-tile") return Backend::cohort_direct_two_tile;
    if (name == "masked-direct-two-tile") return Backend::masked_direct_two_tile;
    if (name == "cohort-shared") return Backend::cohort_shared;
    if (name == "library" || name == "library-batched") return Backend::scalar; // separate library dispatch
    throw std::invalid_argument("unknown campaign backend");
}

bool is_cohort(Backend backend) noexcept {
    return backend == Backend::cohort_direct || backend == Backend::cohort_shared ||
           backend == Backend::cohort_direct_two_tile;
}

bool uses_program_rows(Backend backend) noexcept {
    return is_cohort(backend) || backend == Backend::masked_direct_two_tile;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void check_product(const std::vector<float>& got, const std::vector<float>& want,
                   const char* name) {
    if (got.size() != want.size()) throw std::runtime_error(std::string(name) + " size mismatch");
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float bound = reference::product_abs_tolerance * (1.0f + std::fabs(want[i]));
        if (!std::isfinite(got[i]) || std::fabs(got[i] - want[i]) > bound)
            throw std::runtime_error(std::string(name) + " exceeds product bound at " + std::to_string(i));
    }
}

void check_bound(const std::vector<float>& got, const std::vector<float>& want,
                 const std::vector<float>& bound, const char* name) {
    if (got.size() != want.size() || got.size() != bound.size())
        throw std::runtime_error(std::string(name) + " bound size mismatch");
    for (std::size_t i = 0; i < got.size(); ++i)
        if (!reference::within_absolute_bound(got[i], want[i], bound[i]))
            throw std::runtime_error(std::string(name) + " exceeds compound bound at " + std::to_string(i));
}

std::vector<float> decode(const DeviceBuffer<std::uint16_t>& buffer) {
    auto bits = buffer.download();
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = reference::half_bits_to_float(bits[i]);
    return out;
}

template<class T> void restore(DeviceBuffer<T>& target, const std::vector<T>& values, cudaStream_t stream) {
    if (target.size != values.size()) throw std::runtime_error("reset buffer size mismatch");
    if (target.size) cuda_check(cudaMemcpyAsync(target.ptr, values.data(), values.size() * sizeof(T),
                                              cudaMemcpyHostToDevice, stream), "reset values");
}

void clear_outputs(DeviceFixture& d, cudaStream_t stream) {
    d.y.clear(stream); d.dx.clear(stream); d.dtheta.clear(stream);
    d.dq.clear(stream); d.dweights.clear(stream);
}

// A host view of exactly one step's independently reconstructed gate.  The
// primary STE contract declares every model entry as a derivative candidate.
// Score thresholds determine forward admission; no accessibility observation
// is treated as an exclusion or used to omit a dormant derivative.
void reference_gate(ConsumerHostFixture& h, const std::vector<std::uint8_t>& eligible,
                    const std::vector<float>& scores, GateProfile profile) {
    if (eligible.size() != h.batch * h.conditions) throw std::runtime_error("eligible extent mismatch");
    h.admitted.assign(eligible.size(), 0);
    h.probabilities.assign(eligible.size(), 0);
    const auto words = (h.batch + 31) / 32;
    h.primal.assign(h.conditions * words, 0);
    h.derivative.assign(h.conditions * words, 0);
    if (profile == GateProfile::ste && scores.size() != eligible.size())
        throw std::runtime_error("score extent mismatch");
    for (std::uint64_t cell = 0; cell < h.batch; ++cell) {
        for (std::uint64_t c = 0; c < h.conditions; ++c) {
            const auto i = cell * h.conditions + c;
            const bool candidate = profile == GateProfile::ste || eligible[i] != 0;
            const float p = profile == GateProfile::ste ? sigmoid(scores[i]) : 0.0f;
            const bool admitted = candidate && (profile != GateProfile::ste || p >= 0.5f);
            h.probabilities[i] = p;
            h.admitted[i] = admitted;
            const auto wi = c * words + cell / 32;
            const auto bit = std::uint32_t(1) << (cell % 32);
            if (admitted) h.primal[wi] |= bit;
            if (profile == GateProfile::ste ? candidate : admitted) h.derivative[wi] |= bit;
        }
    }
}

void check_gate(const ConsumerHostFixture& host, const GateView& gate,
                DeviceFixture& d, cudaStream_t stream) {
    std::vector<std::uint32_t> p(host.primal.size()), deriv(host.derivative.size());
    std::vector<std::uint8_t> admitted(host.admitted.size());
    if (!p.empty() && gate.primal_words) cuda_check(cudaMemcpyAsync(p.data(), gate.primal_words, p.size()*sizeof(p[0]), cudaMemcpyDeviceToHost, stream), "copy primal");
    if (!deriv.empty() && gate.derivative_words) cuda_check(cudaMemcpyAsync(deriv.data(), gate.derivative_words, deriv.size()*sizeof(deriv[0]), cudaMemcpyDeviceToHost, stream), "copy derivative");
    if (!admitted.empty() && gate.admitted) cuda_check(cudaMemcpyAsync(admitted.data(), gate.admitted, admitted.size(), cudaMemcpyDeviceToHost, stream), "copy admission");
    cuda_check(cudaStreamSynchronize(stream), "synchronize gate validation");
    if ((!gate.primal_words && !gate.admitted && !host.admitted.empty()) ||
        (gate.profile == GateProfile::ste && !gate.derivative_words && !host.derivative.empty()) ||
        (gate.primal_words && p != host.primal) ||
        (gate.derivative_words && deriv != host.derivative) ||
        (gate.admitted && admitted != host.admitted))
        throw std::runtime_error("gate decisions or derivative membership differ from scalar reference");
    if (gate.profile == GateProfile::ste) {
        const auto actual = d.probs.download();
        for (std::size_t i = 0; i < actual.size(); ++i)
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - host.probabilities[i]) > 2.0e-6f)
                throw std::runtime_error("STE probability differs from scalar reference at " + std::to_string(i));
    }
}

struct Snapshot {
    std::vector<float> scores, theta, weights;
    std::vector<std::uint16_t> weights16;
};

Snapshot snapshot_before(DeviceFixture& d, const DeviceBuffer<float>& scores,
                         BodyKind body, DType dtype, GateProfile gate, cudaStream_t stream) {
    cuda_check(cudaStreamSynchronize(stream), "synchronize pre-step snapshot");
    Snapshot s;
    if (gate == GateProfile::ste) s.scores = scores.download();
    if (body == BodyKind::ordered_product) s.theta = d.theta.download();
    else if (dtype == DType::fp16) s.weights16 = d.w16.download();
    else s.weights = d.w32.download();
    return s;
}

void assign_snapshot(ConsumerHostFixture& h, const Snapshot& s) {
    if (h.body == BodyKind::ordered_product) h.theta = s.theta;
    else if (h.dtype == DType::fp16) h.weights16 = s.weights16;
    else h.weights32 = s.weights;
}

// Independent teacher check is performed outside timed execution. It proves
// that the numerically trained logits and per-row BCE/gradients changed as
// declared and that the circuit equals the thresholded resulting teacher for
// every one of its eight rows, before deriving the observed cell decisions.
std::vector<std::uint8_t> validate_teacher(
    TeacherFixture& t, const std::vector<float>& before, bool trained,
    const std::vector<std::uint8_t>& predicates, cudaStream_t stream,
    std::uint64_t* crossing_count) {
    cuda_check(cudaStreamSynchronize(stream), "finish teacher validation");
    const auto logits = t.logits.download();
    const auto tables = t.luts.download();
    const auto stats = trained ? t.statistics.download() : std::vector<TeacherFixture::Stat>{};
    if (before.size() != t.conditions * 8 || logits.size() != before.size() ||
        tables.size() != t.conditions || predicates.size() != t.batch * t.conditions ||
        (trained && stats.size() != before.size()))
        throw std::runtime_error("teacher validation extent mismatch");
    const auto labels_vector = TeacherFixture::make_targets(t.conditions);
    std::vector<std::uint8_t> expected_tables(t.conditions);
    for (std::size_t table = 0; table < t.conditions; ++table) {
        std::array<float, 8> before_row{}, labels{}, expected_row{};
        for (unsigned row = 0; row < 8; ++row) {
            before_row[row] = before[table * 8 + row];
            labels[row] = labels_vector[table * 8 + row];
        }
        const auto old_reference = reference::teacher_reference(before_row, labels);
        for (unsigned row = 0; row < 8; ++row) {
            const auto i = table * 8 + row;
            expected_row[row] = before_row[row] - (trained ? learning_rate * old_reference.mean_logit_gradient[row] : 0.0f);
            const float bound = 2e-6f * (1.0f + std::fabs(expected_row[row]));
            if (!std::isfinite(logits[i]) || std::fabs(logits[i] - expected_row[row]) > bound)
                throw std::runtime_error("teacher logit update differs from scalar reference at " + std::to_string(i));
            if (trained) {
                const auto& actual = stats[i];
                const float gradient = old_reference.mean_logit_gradient[row];
                const float loss = old_reference.per_row_bce[row];
                if (actual.count != 1 || !std::isfinite(actual.gradient) || !std::isfinite(actual.loss) ||
                    std::fabs(actual.gradient - gradient) > 2e-6f * (1.0f + std::fabs(gradient)) ||
                    std::fabs(actual.loss - loss) > 2e-6f * (1.0f + std::fabs(loss)))
                    throw std::runtime_error("teacher BCE/gradient differs from scalar reference at " + std::to_string(i));
                if ((before_row[row] >= 0.0f) != (logits[i] >= 0.0f)) ++*crossing_count;
            }
        }
        const auto hardened_reference = reference::teacher_reference(expected_row, labels);
        expected_tables[table] = hardened_reference.table;
        if (tables[table] != hardened_reference.table)
            throw std::runtime_error("hardened teacher LUT differs from thresholded scalar teacher at table " + std::to_string(table));
    }
    std::vector<std::uint8_t> result(predicates.size());
    for (std::uint64_t cell = 0; cell < t.batch; ++cell) {
        for (std::uint64_t c = 0; c < t.conditions; ++c) {
            const bool a = predicates[cell*t.conditions+c] != 0;
            const bool b = predicates[cell*t.conditions+(c+1)%t.conditions] != 0;
            const bool cc = predicates[cell*t.conditions+(c+2)%t.conditions] != 0;
            result[cell*t.conditions+c] = reference::teacher_lookup(expected_tables[c], a, b, cc);
        }
    }
    return result;
}

float weight_rounding_bound(float x, DType type) {
    float bound = 1e-6f * (1.0f + std::fabs(x));
    if (type == DType::fp16) {
        const float a = std::fabs(x);
        bound += a < std::ldexp(1.0f, -14) ? std::ldexp(1.0f, -24)
                                           : std::ldexp(1.0f, std::ilogb(a) - 10);
    }
    return bound;
}

void validate_step(ConsumerHostFixture& host, const Snapshot& before,
                   const std::vector<std::uint8_t>& eligible, const GateView& gates,
                   DeviceFixture& d, const DeviceBuffer<float>& scores,
                   const DeviceBuffer<float>& ds,
                   const DeviceBuffer<std::uint16_t>& local_dx16,
                   const DeviceBuffer<std::uint16_t>& local_dw16,
                   const DeviceBuffer<float>& local_dx32, const DeviceBuffer<float>& local_dw32,
                   bool update, cudaStream_t stream) {
    assign_snapshot(host, before);
    reference_gate(host, eligible, before.scores, host.profile);
    check_gate(host, gates, d, stream);
    const auto hp = host.problem();
    const auto hg = host.gates();
    std::vector<float> ey, edx, eq, eg;
    reference::LinearValidationBounds bounds;
    reference::LinearBackward linear;
    if (host.body == BodyKind::ordered_product) {
        auto f = reference::product_forward(hp, hg);
        auto b = reference::product_backward(hp, hg, host.dy.data());
        ey = std::move(f.y); edx = std::move(b.dx); eq = std::move(b.dq); eg = std::move(b.dtheta);
        check_product(d.y.download(), ey, "y");
        check_product(d.dx.download(), edx, "dx");
        check_product(d.dq.download(), eq, "dq");
        check_product(d.dtheta.download(), eg, "dtheta");
    } else {
        auto f = reference::grouped_linear_forward(hp, hg);
        linear = reference::grouped_linear_backward(hp, hg, host.dy.data(), f);
        bounds = reference::grouped_linear_validation_bounds(hp, hg, host.dy.data(), f, linear);
        ey = f.y; edx = linear.dx; eq = linear.dq; eg = linear.dweights;
        check_bound(d.y.download(), ey, bounds.y, "y");
        check_bound(d.dx.download(), edx, bounds.dx, "dx");
        check_bound(d.dq.download(), eq, bounds.dq, "dq");
        check_bound(d.dweights.download(), eg, bounds.dweights, "dweights");
        check_bound(host.dtype == DType::fp16 ? decode(local_dx16) : local_dx32.download(),
                    linear.group_dx, bounds.group_dx, "local dx");
        check_bound(host.dtype == DType::fp16 ? decode(local_dw16) : local_dw32.download(),
                    linear.group_dw, bounds.group_dw, "local dweights");
    }
    if (host.profile == GateProfile::ste) {
        const auto got_ds = ds.download();
        const auto got_scores = scores.download();
        for (std::size_t i = 0; i < eq.size(); ++i) {
            const float expected = eq[i] * host.probabilities[i] * (1.0f - host.probabilities[i]);
            const float error = host.body == BodyKind::ordered_product
                ? reference::product_abs_tolerance * (1.0f + std::fabs(eq[i])) : bounds.dq[i];
            const float bound = 0.25f * error + 2e-6f * (1.0f + std::fabs(expected));
            if (!std::isfinite(got_ds[i]) || std::fabs(got_ds[i] - expected) > bound)
                throw std::runtime_error("STE derivative outside propagated bound at " + std::to_string(i));
            const float score_expected = before.scores[i] - (update ? learning_rate : 0.0f) * expected;
            if (!std::isfinite(got_scores[i]) || std::fabs(got_scores[i] - score_expected) >
                (update ? learning_rate * bound : 0.0f) + 2e-6f * (1.0f + std::fabs(score_expected)))
                throw std::runtime_error("STE update outside propagated bound at " + std::to_string(i));
        }
    }
    if (update) {
        const auto got = host.body == BodyKind::ordered_product ? d.theta.download()
                        : host.dtype == DType::fp16 ? decode(d.w16) : d.w32.download();
        if (got.size() != eg.size()) throw std::runtime_error("updated parameter size mismatch");
        for (std::size_t i = 0; i < eg.size(); ++i) {
            const float old = host.body == BodyKind::ordered_product ? before.theta[i]
                : host.dtype == DType::fp16 ? reference::half_bits_to_float(before.weights16[i]) : before.weights[i];
            const DType type = host.body == BodyKind::ordered_product ? DType::fp32 : host.dtype;
            const float want = reference::round_value(old - learning_rate * eg[i], type);
            const float gbound = host.body == BodyKind::ordered_product
                ? reference::product_abs_tolerance * (1.0f + std::fabs(eg[i])) : bounds.dweights[i];
            if (!std::isfinite(got[i]) || std::fabs(got[i] - want) > learning_rate * gbound + weight_rounding_bound(want, type))
                throw std::runtime_error("numerical update outside propagated bound at " + std::to_string(i));
        }
    }
}

template<class T>
std::vector<T> read_descriptor(const T* pointer, std::size_t count, cudaStream_t stream) {
    std::vector<T> result(count);
    if (count) {
        if (!pointer) throw std::runtime_error("missing execution descriptor array");
        cuda_check(cudaMemcpyAsync(result.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
                   "copy execution descriptor");
        cuda_check(cudaStreamSynchronize(stream), "finish execution descriptor copy");
    }
    return result;
}

void validate_descriptor(const ConsumerHostFixture& h, Backend backend,
                         const ProblemView& problem, BackendOptions options,
                         WorkspaceView workspace, cudaStream_t stream) {
    if (backend != Backend::bitmap && !uses_program_rows(backend)) return;
    DescriptorDebugView d{};
    if (!inspect_descriptor(backend, h.body, problem, options, workspace, &d))
        throw std::runtime_error("cannot inspect execution descriptor");
    const auto derivative = [&](std::uint64_t cell, std::uint64_t condition) {
        return h.profile == GateProfile::ste &&
            ((h.derivative[condition * ((h.batch + 31) / 32) + cell / 32] >> (cell % 32)) & 1u);
    };
    if (backend == Backend::bitmap) {
        const auto p = read_descriptor(d.primal_bitmap, h.batch*d.bitmap_words_per_cell, stream);
        const auto q = read_descriptor(d.derivative_bitmap, h.batch*d.bitmap_words_per_cell, stream);
        const auto pc = read_descriptor(d.primal_count, h.batch, stream);
        const auto dc = read_descriptor(d.derivative_count, h.batch, stream);
        // Copy only initialized worklist prefixes. Unused capacity is not a
        // declared output and may never have been written by the producer.
        std::vector<std::uint64_t> pw(h.batch*d.work_capacity_per_cell);
        std::vector<std::uint64_t> dw(h.batch*d.work_capacity_per_cell);
        for (std::uint64_t cell = 0; cell < h.batch; ++cell) {
            if (pc[cell] > d.work_capacity_per_cell || dc[cell] > d.work_capacity_per_cell)
                throw std::runtime_error("workset count exceeds capacity");
            const auto offset = cell*d.work_capacity_per_cell;
            if (pc[cell]) cuda_check(cudaMemcpyAsync(pw.data()+offset, d.primal_work+offset,
                pc[cell]*sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream), "copy primal workset");
            if (dc[cell]) cuda_check(cudaMemcpyAsync(dw.data()+offset, d.derivative_work+offset,
                dc[cell]*sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream), "copy derivative workset");
        }
        cuda_check(cudaStreamSynchronize(stream), "finish workset copies");
        for (std::uint64_t cell = 0; cell < h.batch; ++cell) {
            for (std::uint64_t word = 0; word < d.bitmap_words_per_cell; ++word) {
                std::uint64_t ep = 0, ed = 0;
                for (unsigned bit = 0; bit < 64 && word*64 + bit < h.conditions; ++bit) {
                    const auto c = word*64 + bit;
                    if (h.admitted[cell*h.conditions+c]) ep |= std::uint64_t(1) << bit;
                    if (derivative(cell,c)) ed |= std::uint64_t(1) << bit;
                }
                if (p[cell*d.bitmap_words_per_cell+word] != ep ||
                    q[cell*d.bitmap_words_per_cell+word] != ed)
                    throw std::runtime_error("bitmap descriptor membership differs from scalar reference");
            }
            std::vector<std::uint64_t> ep, ed;
            for (std::uint64_t group = 0; group < h.groups.size(); ++group) {
                const auto c = h.groups[group].condition_index;
                const bool active = h.admitted[cell*h.conditions+c] != 0;
                if (active) ep.push_back(group);
                if (active || derivative(cell,c)) ed.push_back(group);
            }
            if (pc[cell] != ep.size() || dc[cell] != ed.size() ||
                pc[cell] > d.work_capacity_per_cell || dc[cell] > d.work_capacity_per_cell)
                throw std::runtime_error("workset count differs from scalar reference");
            const auto offset = cell*d.work_capacity_per_cell;
            for (std::size_t i = 0; i < ep.size(); ++i)
                if (pw[offset+i] != ep[i] || h.groups[pw[offset+i]].logical_id != h.groups[ep[i]].logical_id)
                    throw std::runtime_error("primal workset identity differs");
            for (std::size_t i = 0; i < ed.size(); ++i)
                if (dw[offset+i] != ed[i] || h.groups[dw[offset+i]].logical_id != h.groups[ed[i]].logical_id)
                    throw std::runtime_error("derivative workset identity differs");
        }
    } else {
        const auto order = read_descriptor(d.cohort_order, d.cohort_entries, stream);
        const auto p = read_descriptor(d.primal_signature, d.cohort_entries, stream);
        const auto q = read_descriptor(d.derivative_signature, d.cohort_entries, stream);
        const auto neighborhoods = h.neighborhood_group_offsets.size() - 1;
        if (d.cohort_entries != neighborhoods*h.batch)
            throw std::runtime_error("cohort descriptor extent differs");
        using Record = std::tuple<std::uint32_t,std::uint32_t,std::uint64_t>;
        for (std::uint64_t nh=0; nh<neighborhoods; ++nh) {
            const auto gb=h.neighborhood_group_offsets[nh], ge=h.neighborhood_group_offsets[nh+1];
            if (ge < gb || ge-gb > 32) throw std::runtime_error("invalid cohort group extent");
            for (std::uint64_t base=0; base<h.batch; base+=options.cell_tile) {
                const auto end=std::min(h.batch,base+options.cell_tile);
                std::vector<Record> expected;
                for (auto cell=base; cell<end; ++cell) {
                    std::uint32_t ep=0, ed=0;
                    for (auto k=gb; k<ge; ++k) {
                        const auto g=h.neighborhood_groups[k], c=h.groups[g].condition_index;
                        if (h.admitted[cell*h.conditions+c]) ep |= 1u << (k-gb);
                        if (derivative(cell,c)) ed |= 1u << (k-gb);
                    }
                    expected.emplace_back(ep,ed,cell);
                }
                if (backend != Backend::masked_direct_two_tile) std::stable_sort(expected.begin(),expected.end(),[](const Record& a,const Record& b) {
                    return std::tie(std::get<0>(a),std::get<1>(a)) < std::tie(std::get<0>(b),std::get<1>(b));
                });
                for (std::size_t j=0; j<expected.size(); ++j) {
                    const auto at=nh*h.batch+base+j;
                    if (p[at] != std::get<0>(expected[j]) || q[at] != std::get<1>(expected[j]) ||
                        order[at] != std::get<2>(expected[j]))
                        throw std::runtime_error("cell order or local work membership differs");
                }
            }
        }
    }
}

// Fixture generation includes synthetic values and independent reference work;
// neither is execution preparation. Preserve the generated logical topology,
// rebuild the actual address maps consumed below in a separately timed region,
// and compare them to the independently retained factory maps outside timing.
struct LayoutReference {
    std::vector<cellerator::ops::mechanism::GatedProductGroup> groups;
    std::vector<std::uint64_t> arguments, weight_offsets, group_weight_offsets;
    std::vector<cellerator::ops::mechanism::OutputIncidence> outputs;
    std::vector<std::uint64_t> neighborhood_group_offsets, neighborhood_groups;
    std::vector<std::uint64_t> neighborhood_feature_offsets, neighborhood_features;
    std::vector<std::uint32_t> feature_group_masks, argument_local_indices;
    std::uint64_t max_groups=0, max_features=0, max_group_weights=0;
};

LayoutReference take_layout(ConsumerHostFixture& h) {
    LayoutReference old;
    old.groups=std::move(h.groups); old.arguments=std::move(h.arguments);
    old.weight_offsets=std::move(h.weight_offsets); old.group_weight_offsets=std::move(h.group_weight_offsets);
    old.outputs=std::move(h.outputs);
    old.neighborhood_group_offsets=std::move(h.neighborhood_group_offsets);
    old.neighborhood_groups=std::move(h.neighborhood_groups);
    old.neighborhood_feature_offsets=std::move(h.neighborhood_feature_offsets);
    old.neighborhood_features=std::move(h.neighborhood_features);
    old.feature_group_masks=std::move(h.feature_group_masks);
    old.argument_local_indices=std::move(h.argument_local_indices);
    old.max_groups=h.max_groups; old.max_features=h.max_features; old.max_group_weights=h.max_group_weights;
    // Moved-from containers are valid but need not be empty by contract.
    h.groups.clear(); h.arguments.clear(); h.weight_offsets.clear(); h.group_weight_offsets.clear();
    h.outputs.clear(); h.neighborhood_group_offsets.clear(); h.neighborhood_groups.clear();
    h.neighborhood_feature_offsets.clear(); h.neighborhood_features.clear();
    h.feature_group_masks.clear(); h.argument_local_indices.clear();
    h.max_groups=h.max_features=h.max_group_weights=0;
    return old;
}

void prepare_layout(ConsumerHostFixture& h, const WorkloadFixture& w, bool cohort) {
    h.group_weight_offsets.push_back(0);
    std::vector<std::pair<std::size_t,std::size_t>> shapes(h.parameters,{0,0});
    for (std::uint64_t gid=0; gid<w.spec.groups; ++gid) {
        const auto ab=w.group_feature_offsets.at(gid), ae=w.group_feature_offsets.at(gid+1);
        const auto ob=w.group_output_offsets.at(gid), oe=w.group_output_offsets.at(gid+1);
        if (ae<ab || oe<ob || ae>w.group_features.size() || oe>w.group_outputs.size())
            throw std::invalid_argument("invalid supplied group spans");
        const auto inputs=h.body==BodyKind::grouped_linear ? std::min<std::uint64_t>(8,ae-ab) : ae-ab;
        const auto outputs=h.body==BodyKind::grouped_linear ? std::min<std::uint64_t>(4,oe-ob) : oe-ob;
        if (gid/2>=h.parameters || w.group_program.at(gid)>=h.conditions)
            throw std::invalid_argument("group parameter or condition binding outside extent");
        h.groups.push_back({h.arguments.size(), inputs, h.outputs.size(), outputs,
                            gid/2, w.group_program[gid], gid});
        for (std::uint64_t i=0;i<inputs;++i) h.arguments.push_back(w.group_features[ab+i]);
        for (std::uint64_t i=0;i<outputs;++i) h.outputs.push_back({w.group_outputs[ob+i],1.0f});
        h.group_weight_offsets.push_back(h.group_weight_offsets.back()+inputs*outputs);
        const auto shape=std::make_pair(std::size_t(inputs),std::size_t(outputs));
        if (gid%2 && shapes[gid/2]!=shape) throw std::invalid_argument("shared parameter shape differs");
        shapes[gid/2]=shape;
    }
    std::uint64_t weight_offset=0;
    for (const auto& shape:shapes) {
        h.weight_offsets.push_back(weight_offset);
        weight_offset += shape.first*shape.second;
    }
    if (weight_offset != h.weights32.size() || h.weights16.size()!=h.weights32.size())
        throw std::invalid_argument("declared weight storage differs from parameter shapes");
    // Contiguous, bitmap and cuBLAS arms consume no cohort neighborhoods.
    // Their measured preparation must not include or upload these maps.
    if (!cohort) return;
    h.argument_local_indices.resize(h.arguments.size());
    h.neighborhood_group_offsets.push_back(0); h.neighborhood_feature_offsets.push_back(0);
    for (std::size_t begin=0; begin<h.groups.size(); begin+=8) {
        const auto end=std::min(begin+8,h.groups.size());
        std::map<std::uint64_t,std::uint32_t> masks;
        std::uint64_t weights=0;
        for (auto gid=begin; gid<end; ++gid) {
            h.neighborhood_groups.push_back(gid); const auto& g=h.groups[gid];
            weights+=g.argument_count*g.output_count;
            for (std::uint64_t a=0;a<g.argument_count;++a)
                masks[h.arguments[g.argument_begin+a]] |= 1u << (gid-begin);
        }
        std::map<std::uint64_t,std::uint32_t> local;
        for (const auto& item:masks) {
            local[item.first]=local.size(); h.neighborhood_features.push_back(item.first);
            h.feature_group_masks.push_back(item.second);
        }
        for (auto gid=begin;gid<end;++gid) {
            const auto& g=h.groups[gid];
            for (std::uint64_t a=0;a<g.argument_count;++a)
                h.argument_local_indices[g.argument_begin+a]=local.at(h.arguments[g.argument_begin+a]);
        }
        h.neighborhood_group_offsets.push_back(h.neighborhood_groups.size());
        h.neighborhood_feature_offsets.push_back(h.neighborhood_features.size());
        h.max_groups=std::max<std::uint64_t>(h.max_groups,end-begin);
        h.max_features=std::max<std::uint64_t>(h.max_features,masks.size());
        h.max_group_weights=std::max(h.max_group_weights,weights);
    }
}

void verify_layout(const ConsumerHostFixture& h, const LayoutReference& old, bool cohort) {
    if (h.groups.size()!=old.groups.size() || h.outputs.size()!=old.outputs.size())
        throw std::runtime_error("reprepared layout shape differs from independent fixture");
    for (std::size_t i=0;i<h.groups.size();++i) {
        const auto& a=h.groups[i]; const auto& b=old.groups[i];
        if (a.argument_begin!=b.argument_begin || a.argument_count!=b.argument_count ||
            a.output_begin!=b.output_begin || a.output_count!=b.output_count ||
            a.parameter_index!=b.parameter_index || a.condition_index!=b.condition_index ||
            a.logical_id!=b.logical_id)
            throw std::runtime_error("reprepared group binding differs from independent fixture");
    }
    for (std::size_t i=0;i<h.outputs.size();++i)
        if (h.outputs[i].coordinate!=old.outputs[i].coordinate ||
            std::memcmp(&h.outputs[i].scale,&old.outputs[i].scale,sizeof(float))!=0)
            throw std::runtime_error("reprepared output binding differs from independent fixture");
    if (h.arguments!=old.arguments || h.weight_offsets!=old.weight_offsets ||
        h.group_weight_offsets!=old.group_weight_offsets)
        throw std::runtime_error("reprepared common address maps differ from independent fixture");
    if (cohort) {
        if (h.neighborhood_group_offsets!=old.neighborhood_group_offsets ||
            h.neighborhood_groups!=old.neighborhood_groups ||
            h.neighborhood_feature_offsets!=old.neighborhood_feature_offsets ||
            h.neighborhood_features!=old.neighborhood_features ||
            h.feature_group_masks!=old.feature_group_masks ||
            h.argument_local_indices!=old.argument_local_indices ||
            h.max_groups!=old.max_groups || h.max_features!=old.max_features ||
            h.max_group_weights!=old.max_group_weights)
            throw std::runtime_error("reprepared cohort maps differ from independent fixture");
    } else if (!h.neighborhood_group_offsets.empty() || !h.neighborhood_groups.empty() ||
               !h.neighborhood_feature_offsets.empty() || !h.neighborhood_features.empty() ||
               !h.feature_group_masks.empty() || !h.argument_local_indices.empty() ||
               h.max_groups || h.max_features || h.max_group_weights) {
        throw std::runtime_error("non-cohort layout contains unused cohort maps");
    }
}

struct WorkloadHash {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    WorkloadHash() {
        if (!context || EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("initialize workload SHA-256");
    }
    ~WorkloadHash() { EVP_MD_CTX_free(context); }
    void add_bytes(const void* data, std::size_t size) {
        if (size && EVP_DigestUpdate(context, data, size) != 1)
            throw std::runtime_error("update workload SHA-256");
    }
    template<class T> void scalar(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        add_bytes(&value, sizeof(value));
    }
    template<class T> void vector(const std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        scalar(std::uint64_t(values.size())); add_bytes(values.data(), values.size() * sizeof(T));
    }
    void string(const std::string& value) { scalar(std::uint64_t(value.size())); add_bytes(value.data(), value.size()); }
    std::string finish() {
        unsigned char digest[32]; unsigned length = 0;
        if (EVP_DigestFinal_ex(context, digest, &length) != 1 || length != 32)
            throw std::runtime_error("finalize workload SHA-256");
        static const char hex[] = "0123456789abcdef";
        std::string result; result.reserve(64);
        for (auto byte : digest) { result.push_back(hex[byte >> 4]); result.push_back(hex[byte & 15]); }
        return result;
    }
};

std::string workload_digest(const CampaignOptions& o, const WorkloadFixture& w,
                            const ConsumerHostFixture& h, bool content_only = false) {
    WorkloadHash hash;
    hash.string(content_only ? "cgs.workload-content.v1" : "cgs.complete-workload.v1");
    if (!content_only) hash.string(w.spec.id);
    // Update strategy is an implementation arm, not part of the semantic
    // workload. The applied change fraction *is* part of the workload.
    hash.scalar(std::uint8_t(o.condition_update == "none" ? 0 : w.spec.predicate_change_percent));
    hash.scalar(std::uint8_t(o.gate));
    hash.scalar(std::uint8_t(o.body)); hash.scalar(std::uint8_t(o.dtype));
    hash.scalar(w.spec.seed); if (!content_only) hash.scalar(w.spec.reuse_horizon);
    hash.scalar(w.spec.predicate_change_percent); hash.scalar(w.spec.cells);
    hash.scalar(w.spec.features); hash.scalar(w.spec.groups);
    hash.vector(w.accessibility); hash.vector(w.accessibility_after);
    hash.vector(w.edge_roots); hash.vector(w.group_program);
    hash.vector(w.group_feature_offsets); hash.vector(w.group_features);
    hash.vector(w.group_output_offsets); hash.vector(w.group_outputs);
    for (const auto& bank : w.programs) {
        hash.scalar(std::uint64_t(bank.nodes.size()));
        for (const auto& n : bank.nodes) {
            hash.scalar(std::uint8_t(n.op)); hash.scalar(n.feature);
            hash.scalar(n.first_child); hash.scalar(n.child_count);
        }
    }
    if (o.dtype == DType::fp16) hash.vector(h.x16); else hash.vector(h.x32);
    hash.vector(h.dy);
    if (o.body == BodyKind::ordered_product) hash.vector(h.theta);
    else if (o.dtype == DType::fp16) hash.vector(h.weights16);
    else hash.vector(h.weights32);
    // The teacher truth table/targets and STE initialization are fixture rules.
    hash.string("teacher:adjacent3:passthrough-first:logits+-2:table0-crossing;ste:all-D:context-init+-ln3");
    return hash.finish();
}

struct GroupingProbeCounts {
    std::uint64_t active = 0, total = 0, tile_cohorts = 0, tiles = 0;
};

GroupingProbeCounts grouping_probe_counts(const ConsumerHostFixture& h) {
    GroupingProbeCounts counts;
    counts.total = h.admitted.size();
    counts.active = std::count(h.admitted.begin(), h.admitted.end(), std::uint8_t{1});
    for (std::size_t n = 0; n + 1 < h.neighborhood_group_offsets.size(); ++n) {
        for (std::uint64_t base = 0; base < h.batch; base += 32) {
            std::set<std::uint32_t> signatures;
            for (auto cell = base; cell < std::min<std::uint64_t>(h.batch, base + 32); ++cell) {
                std::uint32_t signature = 0;
                const auto begin = h.neighborhood_group_offsets[n], end = h.neighborhood_group_offsets[n+1];
                for (auto k = begin; k < end; ++k) {
                    const auto gid = h.neighborhood_groups[k];
                    if (h.admitted[cell * h.conditions + h.groups[gid].condition_index])
                        signature |= std::uint32_t{1} << (k - begin);
                }
                signatures.insert(signature);
            }
            counts.tile_cohorts += signatures.size(); ++counts.tiles;
        }
    }
    return counts;
}

std::string current_gpu_uuid() {
    int device = 0; cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "current CUDA device");
    cuda_check(cudaGetDeviceProperties(&properties, device), "CUDA device properties");
    const auto& uuid = properties.uuid;
    static const char hex[] = "0123456789abcdef";
    std::string result = "GPU-";
    for (unsigned i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) result.push_back('-');
        const auto c = static_cast<unsigned char>(uuid.bytes[i]);
        result.push_back(hex[c >> 4]); result.push_back(hex[c & 15]);
    }
    return result;
}

void json_array(std::ostream& out, const char* name, const std::vector<double>& values) {
    out << ",\"" << name << "\":[";
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}

} // namespace

namespace {
__global__ void initialize_scores(float* scores, std::uint64_t batch,
                                  std::uint64_t conditions,
                                  const std::uint8_t* condition_bytes,
                                  const std::uint32_t* condition_words,
                                  std::uint64_t words_per_condition) {
    const std::uint64_t count = batch * conditions;
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::uint64_t(blockDim.x) * gridDim.x) {
        const auto cell = i / conditions, condition = i % conditions;
        const bool observed = condition_bytes ? condition_bytes[i] != 0 :
            ((condition_words[condition * words_per_condition + cell / 32] >> (cell % 32)) & 1u);
        scores[i] = observed ? 1.0986122886681098f : -1.0986122886681098f;
    }
}
}

int run_campaign(const CampaignOptions& options, const std::string& output_path) {
    try {
        const Backend backend = parse_backend(options.backend);
        const bool library = options.backend == "library" || options.backend == "library-batched";
        if (library && options.body != BodyKind::grouped_linear)
            throw std::invalid_argument("library control supports grouped-linear only");
        if (options.condition_update != "none" && options.condition_update != "rebuild" &&
            options.condition_update != "incremental")
            throw std::invalid_argument("unknown condition update mode");
        if (options.gate == GateProfile::ste && options.condition_update != "none")
            throw std::invalid_argument("primary STE scores use frozen input context; condition updates are inapplicable");
        if (!options.repeats || options.repeats > 1000 || options.warmups > 1000)
            throw std::invalid_argument("invalid campaign repeat counts");
        if ((options.cell_tile != 32 && options.cell_tile != 64 && options.cell_tile != 128) ||
            (options.stage_count != 1 && options.stage_count != 2))
            throw std::invalid_argument("unsupported tile or staging count");
        const auto& manifest = workload_manifest();
        const auto probes = grouping_probe_specs();
        const auto manifest_it = std::find_if(manifest.begin(), manifest.end(),
            [&](const auto& s) { return s.id == options.case_id; });
        const auto probe_it = std::find_if(probes.begin(), probes.end(),
            [&](const auto& s) { return s.id == options.case_id; });
        const bool grouping_probe = probe_it != probes.end();
        if (manifest_it == manifest.end() && !grouping_probe)
            throw std::invalid_argument("unknown frozen workload case");
        const WorkloadSpec* it = grouping_probe ? &*probe_it : &*manifest_it;
        if (grouping_probe && (options.gate != GateProfile::supplied ||
                options.body != BodyKind::ordered_product || options.dtype != DType::fp16 ||
                (backend != Backend::masked_direct_two_tile && backend != Backend::cohort_direct_two_tile) ||
                options.cell_tile != 32 || options.stage_count != 1 ||
                options.condition_update != "none" || !options.replay_path.empty()))
            throw std::invalid_argument("grouping probe requires supplied product fp16, masked/cohort two-tile, tile 32, stages 1, unchanged synthetic input");
        if (it->role != WorkloadRole::complete_consumer)
            throw std::invalid_argument("producer-only fixture cannot be a complete-cost comparison");
        if (it->reuse_horizon != 1 && it->reuse_horizon != 16 && it->reuse_horizon != 256)
            throw std::invalid_argument("unsupported reuse horizon");
        const bool update_model = it->parameter_pattern == ParameterPattern::update_every_step;
        const bool make_bytes = backend == Backend::scalar || library;
        // Teacher consumes input planes and emits final admission bytes only
        // for byte-based consumers. Other scalar/library paths need bytes only;
        // plane consumers need no intermediate byte representation.
        const bool condition_bytes = make_bytes && options.gate != GateProfile::teacher;
        const bool condition_planes = !condition_bytes;
        // The scalar control uses prepared root maps. Bitmap, direct, and
        // cohorts consume condition planes; the library uses the strong
        // root-mapped control and charges its own activity/readback costs.
        const auto producer = (backend == Backend::scalar || library)
            ? ConditionProducer::root_mapped : ConditionProducer::bitplanes;
        const auto cold_start = Clock::now();
        auto workload = materialize_workload(*it);
        ReplayPreparationTimes replay_times{};
        if (!options.replay_path.empty()) apply_atac_replay(workload, options.replay_path, &replay_times);
        auto host = make_consumer_fixture(workload, options.body, options.dtype, options.gate);
        const auto digest = workload_digest(options, workload, host);
        const auto content_digest = workload_digest(options, workload, host, true);
        const auto grouping_counts = grouping_probe ? grouping_probe_counts(host) : GroupingProbeCounts{};
        if (grouping_probe && (!grouping_counts.active || grouping_counts.active == grouping_counts.total ||
                grouping_counts.tile_cohorts <= grouping_counts.tiles))
            throw std::runtime_error("grouping probe gates or local programs are degenerate");
        const auto fixture_end = Clock::now();
        auto reference_layout = take_layout(host);
        const bool neighborhood_layout = uses_program_rows(backend);
        const auto layout_begin = Clock::now();
        prepare_layout(host, workload, neighborhood_layout);
        const auto layout_end = Clock::now();
        verify_layout(host, reference_layout, neighborhood_layout);
        reference_layout = LayoutReference{}; // release test-only baseline maps
        const auto layout_verified = Clock::now();
        // Preserve deterministic initial numerical inputs for experiment resets.
        // The measured address maps are retained and used by every trial.
        const auto initial_theta = host.theta;
        const auto initial_w16 = host.weights16;
        const auto initial_w32 = host.weights32;
        const auto restore_host_parameters = [&]() {
            host.theta=initial_theta; host.weights16=initial_w16; host.weights32=initial_w32;
        };
        const auto device_preparation_begin = Clock::now();
        Stream stream;
        const auto device_uuid = current_gpu_uuid();
        std::size_t device_free_before = 0, device_total_before = 0;
        cuda_check(cudaMemGetInfo(&device_free_before, &device_total_before), "initial device memory");
        DeviceFixture d(host);
        ConditionDeviceFixture condition(workload, stream.value);
        std::unique_ptr<TeacherFixture> teacher;
        if (options.gate == GateProfile::teacher)
            teacher = std::make_unique<TeacherFixture>(d.p.batch, d.p.topology.conditions, make_bytes);
        const std::size_t score_count = options.gate == GateProfile::ste ? host.batch * host.conditions : 0;
        DeviceBuffer<float> scores(score_count), q(score_count), ds(score_count);
        const std::size_t local_dx_size = options.body == BodyKind::grouped_linear ? host.batch * host.arguments.size() : 0;
        const std::size_t local_dw_size = options.body == BodyKind::grouped_linear ? host.group_weight_offsets.back() : 0;
        DeviceBuffer<std::uint16_t> local_dx16(options.dtype == DType::fp16 ? local_dx_size : 0);
        DeviceBuffer<std::uint16_t> local_dw16(options.dtype == DType::fp16 ? local_dw_size : 0);
        DeviceBuffer<float> local_dx32(options.dtype == DType::fp32 ? local_dx_size : 0);
        DeviceBuffer<float> local_dw32(options.dtype == DType::fp32 ? local_dw_size : 0);
        d.result.dx_group = options.dtype == DType::fp16 ? static_cast<void*>(local_dx16.ptr) : local_dx32.ptr;
        d.result.dweights_group = options.dtype == DType::fp16 ? static_cast<void*>(local_dw16.ptr) : local_dw32.ptr;
        BackendOptions kernel_options{options.cell_tile, options.stage_count};
        std::unique_ptr<backends::GroupedLinearLibraryControl> library_control;
        std::size_t workspace_size = 0;
        if (library) {
            int device = 0; cuda_check(cudaGetDevice(&device), "current device");
            library_control = std::make_unique<backends::GroupedLinearLibraryControl>(
                backends::make_linear_library_layout(host, device), d.p, options.dtype,
                options.backend == "library-batched" ? backends::LinearLibraryArm::batched
                                                     : backends::LinearLibraryArm::per_group);
            workspace_size = library_control->workspace_bytes();
        } else if (!workspace_bytes(backend, options.body, d.p, kernel_options, &workspace_size)) {
            throw std::runtime_error("backend cannot represent requested tile/workspace");
        }
        DeviceBuffer<std::uint8_t> storage(workspace_size);
        WorkspaceView workspace{storage.ptr, storage.size, d.status.ptr};
        SteChain chain{};
        SteChainBuffers ste{};
        if (options.gate == GateProfile::ste) {
            ste.gates = {scores.ptr, nullptr, d.probs.ptr, q.ptr, d.admitted.ptr,
                         d.primal.ptr, d.derivative.ptr, d.status.ptr};
            ste.dq = d.dq.ptr; ste.ds = ds.ptr; ste.parameters = scores.ptr;
        }
        // Initial condition preparation and score initialization are part of
        // cold preparation. Subsequent identical resets are measurement-harness
        // bookkeeping and are measured separately from recurrent execution.
        auto teacher_initial_logits = teacher
            ? TeacherFixture::make_logits(d.p.topology.conditions) : std::vector<float>{};
        if (teacher && !teacher_initial_logits.empty()) {
            // Both values cross 0.5 under one declared SGD step. The resulting
            // target truth table is still the controlled pass-through-first rule.
            teacher_initial_logits[0] = 0.001f;
            teacher_initial_logits[7] = -0.001f;
        }
        auto reset = [&]() {
            const auto start = Clock::now();
            d.status.clear(stream.value);
            if (options.body == BodyKind::ordered_product) restore(d.theta, host.theta, stream.value);
            else if (options.dtype == DType::fp16) restore(d.w16, host.weights16, stream.value);
            else restore(d.w32, host.weights32, stream.value);
            cuda_check(condition.rebuild(producer, false, GateProfile::supplied, condition_bytes, stream.value, condition_planes), "reset condition");
            if (teacher) {
                restore(teacher->logits, teacher_initial_logits, stream.value);
                teacher->teacher_generation = 0;
            }
            if (options.gate == GateProfile::ste && score_count) {
                const auto gate = condition.gates(GateProfile::supplied);
                const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>((score_count + 255) / 256, 65535));
                initialize_scores<<<blocks,256,0,stream.value>>>(scores.ptr, host.batch, host.conditions,
                                                                 gate.admitted, gate.primal_words, gate.words_per_condition);
                cuda_check(cudaPeekAtLastError(), "initialize score context");
            }
            cuda_check(cudaStreamSynchronize(stream.value), "finish trial reset");
            if (condition.status(stream.value)) throw std::runtime_error("condition reset failed");
            return elapsed_ms(start, Clock::now());
        };
        reset();
        const auto cold_end = Clock::now();
        std::size_t device_free_after = 0, device_total_after = 0;
        cuda_check(cudaMemGetInfo(&device_free_after, &device_total_after), "prepared device memory");
        if (device_total_before != device_total_after) throw std::runtime_error("device changed during preparation");

        // All state for a single sequence lives on one stream. Status is sticky
        // across its steps. Full rebuild and incremental updates both toggle
        // the same real input leaves, beginning with before -> after on step 0.
        std::uint64_t teacher_validation_crossings = 0;
        auto sequence = [&](bool validate, unsigned steps, PhaseProbe* probe = nullptr) {
            bool after = false, prepared = false, gate_ready = false, teacher_ready = false;
            std::uint64_t parameter_revision = 1, score_generation = 1;
            GateView current{};
            for (unsigned step = 0; step < steps; ++step) {
                const bool check = validate && (step == 0 || step + 1 == steps);
                std::vector<float> teacher_before;
                if (check && teacher) {
                    cuda_check(cudaStreamSynchronize(stream.value), "capture teacher before fit");
                    teacher_before = teacher->logits.download();
                }
                bool teacher_trained = false;
                const bool probe_step = probe && step == 0;
                const auto start_phase = [&](Phase p, bool active = true) {
                    if (probe_step && active) probe->phases[p].start(stream.value);
                };
                const auto end_phase = [&](Phase p, bool active = true) {
                    if (probe_step && active) probe->phases[p].stop(stream.value);
                };
                // Full rebuild is deliberately forced even for a zero-change
                // ablation. A dependency-driven stream with an empty change
                // list performs no update and keeps its prepared descriptor.
                const bool changed = options.condition_update == "rebuild" ||
                    (options.condition_update == "incremental" && !workload.changed_input_features.empty());
                start_phase(condition_phase, changed);
                if (changed) {
                    after = !after;
                    if (options.condition_update == "incremental")
                        cuda_check(condition.update(producer, GateProfile::supplied, condition_bytes, stream.value, condition_planes), "dependency update");
                    else
                        cuda_check(condition.rebuild(producer, after, GateProfile::supplied, condition_bytes, stream.value, condition_planes), "full condition rebuild");
                    merge_condition_status<<<1,1,0,stream.value>>>(condition.device_status(), d.status.ptr);
                    cuda_check(cudaPeekAtLastError(), "merge condition status");
                }
                end_phase(condition_phase, changed);
                auto predicate = condition.gates(GateProfile::supplied);
                predicate.revisions.parameter_revision = parameter_revision;
                predicate.revisions.temperature_bits = host.revisions.temperature_bits;
                predicate.revisions.threshold_bits = host.revisions.threshold_bits;
                start_phase(clear_phase);
                clear_outputs(d, stream.value);
                if (options.body == BodyKind::grouped_linear) {
                    if (options.dtype == DType::fp16) { local_dx16.clear(stream.value); local_dw16.clear(stream.value); }
                    else { local_dx32.clear(stream.value); local_dw32.clear(stream.value); }
                }
                end_phase(clear_phase);
                const bool build_gate = options.gate == GateProfile::teacher
                    ? (!teacher_ready || changed || update_model)
                    : options.gate == GateProfile::ste ? (!gate_ready || update_model) : false;
                start_phase(gate_phase, build_gate);
                if (options.gate == GateProfile::supplied) {
                    current = predicate;
                    if (!make_bytes) current.admitted = nullptr;
                } else if (options.gate == GateProfile::teacher) {
                    if (!teacher_ready || changed || update_model) {
                        const bool train = !teacher_ready || update_model;
                        teacher_trained = train;
                        current = teacher->enqueue(predicate, d.status.ptr, parameter_revision,
                                                   train, stream.value,
                                                   probe_step ? probe->teacher_events() : teacher::ChainEvents{},
                                                   make_bytes);
                        teacher_ready = true;
                    }
                    if (!make_bytes) current.admitted = nullptr;
                } else {
                    if (!gate_ready || update_model) {
                        auto revisions = predicate.revisions;
                        revisions.score_generation = score_generation;
                        cuda_check(ste_chain_forward(d.p.batch, d.p.topology.conditions, 1.0f, 0.5f,
                                                     revisions, ste, &chain, {}, stream.value), "STE forward");
                        current = chain.gates;
                        gate_ready = true;
                    }
                }
                end_phase(gate_phase, build_gate);
                const auto& eligible = after ? workload.predicates_after : workload.predicates_before;
                Snapshot before;
                if (check) before = snapshot_before(d, scores, options.body, options.dtype, options.gate, stream.value);
                // Every model/gate/accessibility revision invalidates the
                // prepared execution descriptor; frozen schedules are reused.
                const bool need_prepare = !prepared || changed || update_model;
                start_phase(prepare_phase, !library && need_prepare);
                if (!library && need_prepare) {
                    cuda_check(prepare(backend, options.body, d.p, current, kernel_options, workspace, stream.value), "prepare execution");
                    prepared = true;
                }
                end_phase(prepare_phase, !library && need_prepare);
                start_phase(forward_phase);
                if (library) cuda_check(library_control->forward(d.p, current, d.result, workspace, stream.value), "library forward");
                else cuda_check(forward(backend, options.body, d.p, current, kernel_options, d.result, workspace, stream.value), "forward");
                end_phase(forward_phase);
                start_phase(backward_phase);
                if (library) cuda_check(library_control->backward(d.p, current, d.dy.ptr, d.result, workspace, stream.value), "library backward");
                else cuda_check(backward(backend, options.body, d.p, current, d.dy.ptr, kernel_options, d.result, workspace, stream.value), "backward");
                end_phase(backward_phase);
                start_phase(score_phase, options.gate == GateProfile::ste);
                if (options.gate == GateProfile::ste) {
                    if (update_model) {
                        cuda_check(ste_chain_backward_update(&chain, current.revisions, d.result, workspace,
                                                             learning_rate, {}, stream.value), "STE gradient and update");
                        ++score_generation;
                    } else {
                        // The surrogate is required even for a frozen score,
                        // but an unnecessary zero-rate SGD is not executed.
                        cuda_check(ste_backward(chain.snapshot, current.revisions, d.probs.ptr,
                                                d.derivative.ptr, d.dq.ptr, ds.ptr, stream.value), "frozen STE gradient");
                    }
                }
                end_phase(score_phase, options.gate == GateProfile::ste);
                start_phase(model_phase, update_model);
                if (update_model) {
                    void* values = options.body == BodyKind::ordered_product ? static_cast<void*>(d.theta.ptr)
                                                                             : const_cast<void*>(d.p.weights);
                    const float* gradient = options.body == BodyKind::ordered_product ? d.dtheta.ptr : d.dweights.ptr;
                    const auto count = options.body == BodyKind::ordered_product ? d.p.topology.parameters : d.p.weight_count;
                    const auto dtype = options.body == BodyKind::ordered_product ? DType::fp32 : options.dtype;
                    cuda_check(model_sgd(values, gradient, count, dtype, learning_rate, d.status.ptr, stream.value), "numerical SGD");
                    ++parameter_revision;
                }
                end_phase(model_phase, update_model);
                if (check) {
                    cuda_check(cudaStreamSynchronize(stream.value), "finish validation step");
                    const auto status = d.status.download();
                    if (!status.empty() && status[0]) throw std::runtime_error("device status during validation=" + std::to_string(status[0]));
                    std::vector<std::uint8_t> teacher_expected;
                    if (teacher) teacher_expected = validate_teacher(*teacher, teacher_before, teacher_trained,
                                                                     eligible, stream.value, &teacher_validation_crossings);
                    validate_step(host, before, teacher ? teacher_expected : eligible, current, d, scores, ds,
                                  local_dx16, local_dw16, local_dx32, local_dw32,
                                  update_model, stream.value);
                    if (!library) validate_descriptor(host, backend, d.p, kernel_options, workspace, stream.value);
                }
            }
            cuda_check(cudaStreamSynchronize(stream.value), "finish complete sequence");
            const auto status = d.status.download();
            if (!status.empty() && status[0]) throw std::runtime_error("device status after sequence=" + std::to_string(status[0]));
            if (condition.status(stream.value)) throw std::runtime_error("condition status after sequence");
        };

        // Validation is performed outside measured sequences and includes the
        // first and last full producer/body/backward/update steps. Capture the
        // actual pre-step numerical parameters and independently recompute each
        // scalar reference, avoiding accumulation of oracle differences.
        const auto validation_begin = Clock::now();
        sequence(true, it->reuse_horizon);
        if (teacher && teacher_validation_crossings < 2)
            throw std::runtime_error("teacher validation did not observe both required threshold crossings");
        const auto validation_end = Clock::now();
        std::vector<double> resets, recurrent, recovery;
        std::vector<std::string> output_digests;
        for (unsigned trial = 0; trial < options.warmups + options.repeats; ++trial) {
            // Validation mutates the host numerical copy for its independent
            // oracle. Restore values but retain the measured address maps.
            restore_host_parameters();
            const double reset_ms = reset();
            const auto begin = Clock::now();
            sequence(false, it->reuse_horizon);
            const auto end = Clock::now();
            // Include canonical outputs/gradients/score updates in recovery;
            // their copies are separate from resident execution latency.
            const auto recover_start = Clock::now();
            const auto y = d.y.download(), dx = d.dx.download(), dq = d.dq.download();
            const auto gradient = options.body == BodyKind::ordered_product ? d.dtheta.download() : d.dweights.download();
            const auto score = options.gate == GateProfile::ste ? scores.download() : std::vector<float>{};
            const double recovery_ms = elapsed_ms(recover_start, Clock::now());
            std::string output_digest;
            if (grouping_probe) {
                WorkloadHash output_hash;
                output_hash.string("cgs.output-and-vjp.v1");
                output_hash.vector(y); output_hash.vector(dx); output_hash.vector(dq);
                output_hash.vector(gradient); output_hash.vector(score);
                output_digest = output_hash.finish();
            }
            if (trial >= options.warmups) {
                resets.push_back(reset_ms);
                recurrent.push_back(elapsed_ms(begin, end));
                recovery.push_back(recovery_ms);
                output_digests.push_back(output_digest);
            }
        }
        float teacher_uniform = 0.0f, teacher_weighted = 0.0f;
        std::array<std::uint64_t, 8> teacher_rows{};
        if (teacher) {
            const auto logits = teacher->logits.download();
            const auto changes = teacher->changes.download();
            for (float value : changes) teacher_uniform += value;
            if (!changes.empty()) teacher_uniform /= changes.size();
            const bool last_after = options.condition_update != "none" && (it->reuse_horizon % 2 == 1);
            const auto& eligible = last_after ? workload.predicates_after : workload.predicates_before;
            double total = 0.0;
            for (std::uint64_t cell = 0; cell < host.batch; ++cell) {
                for (std::uint64_t c = 0; c < host.conditions; ++c) {
                    const bool active = eligible[cell * host.conditions + c] != 0;
                    const bool b = eligible[cell * host.conditions + (c + 1) % host.conditions] != 0;
                    const bool cc = eligible[cell * host.conditions + (c + 2) % host.conditions] != 0;
                    const unsigned row = 4u * unsigned(active) + 2u * unsigned(b) + unsigned(cc);
                    ++teacher_rows[row];
                    const float p = sigmoid(logits[c * 8 + row]);
                    total += std::fabs(p - (p >= 0.5f ? 1.0f : 0.0f));
                }
            }
            if (!eligible.empty()) teacher_weighted = static_cast<float>(total / eligible.size());
        }
        // Probe one freshly reset step separately, with event timestamps and
        // no additional synchronizations between phases. The samples are
        // explanatory instrumentation and are not part of confirmation time.
        restore_host_parameters();
        reset();
        PhaseProbe probe;
        sequence(false, 1, &probe);
        const auto net_memory = device_free_before >= device_free_after ? device_free_before - device_free_after : 0;
        std::ostringstream out;
        out << std::setprecision(12)
            << "{\"schema\":\"cgs.campaign.v1\",\"status\":\"ok\",\"accuracy_accepted\":true"
            << ",\"validation_scope\":\"first-and-last-complete-steps\""
            << ",\"descriptor_validation\":\"" << (library ? "library-gate-planes-and-independent-control-tests" :
                backend == Backend::bitmap || uses_program_rows(backend)
                    ? "exact-work-membership-and-raw-group-identity" : "exact-gate-planes-no-materialized-workset") << "\""
            << ",\"workload_sha256\":\"" << digest << "\",\"workload_content_sha256\":\"" << content_digest << "\",\"device_uuid\":\"" << device_uuid << "\""
            << ",\"case\":\"" << options.case_id << "\",\"backend\":\"" << options.backend << "\""
            << ",\"gate\":\"" << (options.gate == GateProfile::ste ? "ste" : options.gate == GateProfile::teacher ? "teacher" : "supplied") << "\""
            << ",\"body\":\"" << (options.body == BodyKind::ordered_product ? "product" : "linear") << "\""
            << ",\"precision\":\"" << (options.dtype == DType::fp16 ? "fp16" : "fp32") << "\""
            << ",\"numerical_contract\":{\"input_storage\":\"" << (options.dtype == DType::fp16 ? "fp16" : "fp32")
            << "\",\"multiplication\":\"" << (library && options.dtype == DType::fp16 ? "cublas-fp16-operands-compute32" : "fp32")
            << "\",\"accumulation\":\"fp32\",\"output_and_canonical_vjp\":\"fp32\",\"local_linear_F_dF_dX_dW\":\""
            << (options.body == BodyKind::grouped_linear ? (options.dtype == DType::fp16 ? "fp16" : "fp32") : "not-applicable") << "\"}"
            << ",\"condition_update\":\"" << options.condition_update << "\""
            << ",\"input\":\"" << (options.replay_path.empty() ? "synthetic" : "10x-pbmc-cgatac01") << "\""
            << ",\"seed\":" << it->seed << ",\"cells\":" << host.batch << ",\"features\":" << host.coordinates
            << ",\"groups\":" << host.groups.size() << ",\"condition_edges\":" << host.conditions
            << ",\"source_dag_nodes\":" << condition.source_node_count() << ",\"evaluated_dag_nodes\":" << condition.evaluated_node_count()
            << ",\"nominal_input_change_percent\":" << unsigned(it->predicate_change_percent)
            << ",\"applied_input_change_percent\":" << (options.condition_update == "none" ? 0u : unsigned(it->predicate_change_percent))
            << ",\"nominal_root_changes\":" << workload.root_decision_changes
            << ",\"applied_root_changes_per_toggle\":" << (options.condition_update == "none" ? 0u : workload.root_decision_changes)
            << ",\"reuse_horizon\":" << it->reuse_horizon << ",\"tile\":" << options.cell_tile << ",\"stages\":" << options.stage_count
            << ",\"cell_tiles_per_cta\":" << (neighborhood_layout ? (backend == Backend::cohort_direct ? 1 : 2) : 0)
            << ",\"cell_order\":\"" << (library ? "library-specific" : backend == Backend::masked_direct_two_tile ? "contiguous" : is_cohort(backend) ? "exact-program-cohorts" : "per-cell-group-warps") << "\""
            << ",\"per_lane_group_mask\":" << (backend == Backend::masked_direct_two_tile ? "true" : "false")
            << ",\"program_sorting\":" << (is_cohort(backend) ? "true" : "false")
            << ",\"cohort_feature_staging\":" << (backend == Backend::cohort_shared ? "true" : "false")
            << ",\"cohort_staging_buffers\":" << (backend == Backend::cohort_shared ? options.stage_count : 0)
            << ",\"parameter_update\":" << (update_model ? "true" : "false")
            << ",\"ste_derivative_candidates\":\"" << (options.gate == GateProfile::ste ? "all-model-entries" : "not-applicable") << "\""
            << ",\"ste_score_initialization\":\"" << (options.gate == GateProfile::ste ? "cuda-context-plus-minus-ln3" : "not-applicable") << "\""
            << ",\"predicate_updates_applicable\":" << (options.gate == GateProfile::ste ? "false" : "true")
            << ",\"teacher_input_rule\":\"" << (teacher ? "three-neighboring-root-ordinals" : "not-applicable") << "\""
            << ",\"teacher_initial_fit_in_recurrent_horizon\":" << (teacher ? "true" : "false")
            << ",\"teacher_fit_every_step\":" << (teacher && update_model ? "true" : "false")
            << ",\"teacher_validation_threshold_crossings\":" << teacher_validation_crossings
            << ",\"teacher_uniform_soft_hard_mean\":" << teacher_uniform
            << ",\"teacher_workload_weighted_soft_hard_mean\":" << teacher_weighted
            << ",\"host_fixture_ms\":" << elapsed_ms(cold_start, fixture_end)
            << ",\"replay_input_preparation_ms\":" << replay_times.operational_input_load_ms
            << ",\"replay_reference_recompute_ms\":" << replay_times.reference_recompute_ms
            << ",\"replay_payload_bytes_hashed\":" << replay_times.payload_bytes_hashed
            << ",\"replay_payload_bytes_read\":" << replay_times.payload_bytes_read
            << ",\"host_layout_preparation_ms\":" << elapsed_ms(layout_begin, layout_end)
            << ",\"host_layout_validation_ms\":" << elapsed_ms(layout_end, layout_verified)
            << ",\"trial_reset_setup_ms\":" << elapsed_ms(layout_verified, device_preparation_begin)
            << ",\"cold_device_preparation_ms\":" << elapsed_ms(device_preparation_begin, cold_end)
            << ",\"cold_device_preparation_includes_host_condition_maps\":true"
            << ",\"cold_execution_preparation_ms\":" << (replay_times.operational_input_load_ms + elapsed_ms(layout_begin, layout_end) + elapsed_ms(device_preparation_begin, cold_end))
            << ",\"cold_total_ms\":" << elapsed_ms(cold_start, cold_end)
            << ",\"cold_total_includes_fixture_and_reference_preparation\":true"
            << ",\"validation_ms\":" << elapsed_ms(validation_begin, validation_end)
            << ",\"neighborhood_layout_prepared\":" << (neighborhood_layout ? "true" : "false")
            << ",\"cohort_layout_prepared\":" << (is_cohort(backend) ? "true" : "false")
            << ",\"workspace_bytes\":" << workspace_size << ",\"condition_device_bytes\":" << condition.device_bytes()
            << ",\"library_owned_device_bytes\":" << (library ? library_control->device_bytes() : 0)
            << ",\"library_pinned_host_bytes\":" << (library ? library_control->pinned_host_bytes() : 0)
            << ",\"net_device_allocated_bytes\":" << net_memory << ",\"device_total_bytes\":" << device_total_after;
        out << ",\"teacher_observed_rows\":[";
        for (std::size_t i = 0; i < teacher_rows.size(); ++i) { if (i) out << ','; out << teacher_rows[i]; }
        out << ']';
        json_array(out, "trial_reset_ms", resets);
        json_array(out, "recurrent_complete_horizon_ms", recurrent);
        if (grouping_probe) {
            out << ",\"probe_active_cell_conditions\":" << grouping_counts.active
                << ",\"probe_total_cell_conditions\":" << grouping_counts.total
                << ",\"probe_tile_cohorts\":" << grouping_counts.tile_cohorts
                << ",\"probe_neighborhood_tiles\":" << grouping_counts.tiles;
            json_array(out, "resident_complete_horizon_ms", recurrent);
            out << ",\"resident_horizon_scope\":\"one-prepare-clears-forward-backward-completion-status\"";
            out << ",\"output_and_vjp_sha256\":[";
            for (std::size_t i = 0; i < output_digests.size(); ++i) {
                if (i) out << ','; out << '"' << output_digests[i] << '"';
            }
            out << ']';
        }
        json_array(out, "output_recovery_ms", recovery);
        static constexpr const char* phase_names[phase_count] = {
            "condition", "clear", "gate", "prepare", "forward", "backward", "score_gradient_update", "model_update"};
        out << ",\"phase_probe_scope\":\"first-step-after-reset-separate-from-timing\"";
        out << ",\"phase_probe_host_ms\":{";
        for (std::size_t i = 0; i < phase_count; ++i) {
            if (i) out << ','; out << '"' << phase_names[i] << "\":[" << probe.phases[i].host_ms << ']';
        }
        out << "},\"phase_probe_device_ms\":{";
        for (std::size_t i = 0; i < phase_count; ++i) {
            if (i) out << ','; out << '"' << phase_names[i] << "\":[" << probe.phases[i].device_ms() << ']';
        }
        if (probe.teacher_recorded) {
            out << ",\"teacher_fit\":[" << probe.teacher_fit.device_ms() << ']'
                << ",\"teacher_harden\":[" << probe.teacher_harden.device_ms() << ']'
                << ",\"teacher_circuit\":[" << probe.teacher_circuit.device_ms() << ']';
        }
        out << "}}\n";
        if (output_path.empty()) std::fputs(out.str().c_str(), stdout);
        else {
            std::ofstream file(output_path);
            if (!file) throw std::runtime_error("cannot open campaign output");
            file << out.str();
            if (!file) throw std::runtime_error("cannot write campaign output");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cgs campaign: %s\n", e.what());
        return 1;
    }
}

bool run_campaign_layout_checks(const std::string& case_id, std::string* error) {
    try {
        if (parse_backend("cohort-direct-two-tile") != Backend::cohort_direct_two_tile ||
            !is_cohort(Backend::cohort_direct_two_tile) || is_cohort(Backend::direct))
            throw std::logic_error("cached two-tile backend classification is invalid");
        if (parse_backend("masked-direct-two-tile") != Backend::masked_direct_two_tile ||
            is_cohort(Backend::masked_direct_two_tile) || !uses_program_rows(Backend::masked_direct_two_tile))
            throw std::logic_error("masked two-tile backend classification is invalid");
        std::size_t cases = 0;
        for (const auto& spec : workload_manifest()) {
            if (!case_id.empty() && case_id != spec.id) continue;
            auto workload = materialize_workload(spec);
            for (const auto body : {BodyKind::ordered_product, BodyKind::grouped_linear}) {
                for (const auto dtype : {DType::fp16, DType::fp32}) {
                    for (const bool cohort : {false, true}) {
                        auto host = make_consumer_fixture(workload, body, dtype, GateProfile::supplied);
                        auto expected = take_layout(host);
                        prepare_layout(host, workload, cohort);
                        verify_layout(host, expected, cohort);
                    }
                }
            }
            ++cases;
        }
        // Prove both private probe cases are the same actual data and maps,
        // differing only in their name and evaluation horizon, and that they
        // preserve the coherent source case's before-state and numeric inputs.
        const auto probes = grouping_probe_specs();
        const auto a = materialize_workload(probes[0]);
        const auto b = materialize_workload(probes[1]);
        auto ah = make_consumer_fixture(a, BodyKind::ordered_product, DType::fp16, GateProfile::supplied);
        auto bh = make_consumer_fixture(b, BodyKind::ordered_product, DType::fp16, GateProfile::supplied);
        CampaignOptions identity_options;
        if (a.spec.reuse_horizon != 1 || b.spec.reuse_horizon != 16 ||
            a.spec.parameter_pattern != ParameterPattern::frozen || b.spec.parameter_pattern != ParameterPattern::frozen ||
            !a.changed_input_features.empty() || !b.changed_input_features.empty() ||
            a.accessibility != a.accessibility_after || b.accessibility != b.accessibility_after ||
            a.predicates_before != a.predicates_after || b.predicates_before != b.predicates_after ||
            workload_digest(identity_options, a, ah, true) != workload_digest(identity_options, b, bh, true) ||
            a.predicates_before != b.predicates_before || a.edge_roots != b.edge_roots ||
            ah.x16 != bh.x16 || ah.dy != bh.dy || ah.theta != bh.theta || ah.admitted != bh.admitted ||
            ah.primal != bh.primal || ah.derivative != bh.derivative)
            throw std::runtime_error("grouping probe horizons differ in materialized data or gates");
        auto original = workload_manifest().at(4);
        if (original.id != "dag7-64root-complete-consumer-coherent-dense-high-update-r16-change1")
            throw std::runtime_error("grouping probe source case changed");
        auto ow = materialize_workload(original);
        auto oh = make_consumer_fixture(ow, BodyKind::ordered_product, DType::fp16, GateProfile::supplied);
        if (ow.accessibility != a.accessibility || ow.predicates_before != a.predicates_before ||
            ow.edge_roots != a.edge_roots || ow.group_program != a.group_program ||
            ow.group_feature_offsets != a.group_feature_offsets || ow.group_features != a.group_features ||
            ow.group_output_offsets != a.group_output_offsets || ow.group_outputs != a.group_outputs ||
            oh.x16 != ah.x16 || oh.dy != ah.dy || oh.theta != ah.theta || oh.admitted != ah.admitted)
            throw std::runtime_error("grouping probe differs from coherent source inputs/maps");
        auto a_layout_copy = ah;
        auto a_layout = take_layout(a_layout_copy);
        verify_layout(bh, a_layout, true);
        verify_layout(oh, a_layout, true);
        const auto counts = grouping_probe_counts(ah);
        if (!counts.active || counts.active == counts.total || counts.tile_cohorts <= counts.tiles)
            throw std::runtime_error("grouping probe gate/cohort counts are degenerate");
        if (!case_id.empty() && (case_id == probes[0].id || case_id == probes[1].id)) ++cases;
        if (!cases) throw std::invalid_argument("layout check case is not in the manifest or grouping probe");
        std::printf("grouping probe host proof: active=%llu/%llu tile_cohorts=%llu neighborhood_tiles=%llu content_sha256=%s\n",
            static_cast<unsigned long long>(counts.active), static_cast<unsigned long long>(counts.total),
            static_cast<unsigned long long>(counts.tile_cohorts), static_cast<unsigned long long>(counts.tiles),
            workload_digest(identity_options, a, ah, true).c_str());
        std::printf("grouping probe workload digests: r1=%s r16=%s\n",
            workload_digest(identity_options, a, ah).c_str(),
            workload_digest(identity_options, b, bh).c_str());
        return true;
    } catch (const std::exception& ex) {
        if (error) *error = ex.what();
        return false;
    }
}

} // namespace cgs
