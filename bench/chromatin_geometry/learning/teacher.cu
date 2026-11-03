#include "learning/teacher.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cgs::teacher {
namespace {

constexpr unsigned kThreads = 128;

__device__ float sigmoid(float x) {
    // The two branches avoid exp overflow and define the exact probability
    // consumed by both training and the thresholded numerical teacher.
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__global__ void statistics_kernel(const float* logits, const std::uint8_t* rows,
                                  const float* targets, std::uint64_t count,
                                  RowStatistics* stats, std::uint32_t* status) {
    __shared__ float sums[kThreads];
    __shared__ unsigned long long counts[kThreads];
    const unsigned row = blockIdx.x, lane = threadIdx.x;
    const float z = logits[row];
    if (!isfinite(z)) atomicOr(status, unsigned(status_nonfinite));
    float sum = 0.0f;
    unsigned long long n = 0;
    for (std::uint64_t i = lane; i < count; i += kThreads) {
        const unsigned r = rows[i];
        const float target = targets[i];
        if (r >= 8 || !isfinite(target) || target < 0.0f || target > 1.0f) {
            atomicOr(status, unsigned(status_invalid_descriptor));
        } else if (r == row) {
            sum += target;
            ++n;
        }
    }
    sums[lane] = sum;
    counts[lane] = n;
    __syncthreads();
    for (unsigned stride = kThreads / 2; stride; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            counts[lane] += counts[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        RowStatistics result{};
        result.count = counts[0];
        if (result.count && isfinite(z)) {
            const float y = sums[0] / static_cast<float>(result.count);
            result.gradient = sigmoid(z) - y;
            // BCE with logits, stable even for very large finite magnitudes.
            result.loss = fmaxf(z, 0.0f) - z * y + log1pf(expf(-fabsf(z)));
            if (!isfinite(result.loss) || !isfinite(result.gradient))
                atomicOr(status, unsigned(status_nonfinite));
        }
        stats[row] = result;
    }
}

__global__ void update_kernel(float* logits, const RowStatistics* stats,
                              float rate, std::uint32_t* status) {
    __shared__ float candidate[8];
    const unsigned row = threadIdx.x;
    if (row < 8) {
        candidate[row] = logits[row] - rate * stats[row].gradient;
        if (!isfinite(candidate[row])) atomicOr(status, unsigned(status_nonfinite));
    }
    __syncthreads();
    if (row < 8 && *status == 0) logits[row] = candidate[row];
}

__global__ void harden_kernel(const float* logits, std::uint64_t table_count,
                              std::uint8_t* luts, float* change,
                              std::uint32_t* status) {
    for (std::uint64_t t = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         t < table_count; t += std::uint64_t(gridDim.x) * blockDim.x) {
        unsigned mask = 0;
        float error = 0.0f;
        bool valid = true;
        for (unsigned row = 0; row < 8; ++row) {
            const float z = logits[8 * t + row];
            if (!isfinite(z)) { valid = false; break; }
            const float p = sigmoid(z);
            const bool bit = p >= 0.5f;
            mask |= unsigned(bit) << row;
            error += fabsf(p - float(bit));
        }
        if (!valid) {
            atomicOr(status, unsigned(status_nonfinite));
        } else {
            luts[t] = static_cast<std::uint8_t>(mask);
            if (change) change[t] = error * 0.125f;
        }
    }
}

__global__ void predict_soft_kernel(const float* logits, std::uint64_t tables,
                                    const std::uint32_t* table_indices,
                                    const std::uint8_t* rows, std::uint64_t count,
                                    float* probabilities, std::uint8_t* decisions,
                                    float* changes, std::uint32_t* status) {
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::uint64_t(gridDim.x) * blockDim.x) {
        const auto table = table_indices[i];
        const auto row = rows[i];
        if (table >= tables || row >= 8) {
            atomicOr(status, unsigned(status_invalid_descriptor));
            continue;
        }
        const float z = logits[std::uint64_t(table) * 8 + row];
        if (!isfinite(z)) {
            atomicOr(status, unsigned(status_nonfinite));
            continue;
        }
        const float p = sigmoid(z);
        const bool decision = p >= 0.5f;
        if (probabilities) probabilities[i] = p;
        if (decisions) decisions[i] = static_cast<std::uint8_t>(decision);
        if (changes) changes[i] = fabsf(p - float(decision));
    }
}

__device__ std::uint32_t lut3(std::uint32_t a, std::uint32_t b,
                              std::uint32_t c, std::uint8_t table) {
    std::uint32_t result = 0;
    for (unsigned row = 0; row < 8; ++row) {
        if ((table >> row) & 1u)
            result |= ((row & 4u) ? a : ~a) & ((row & 2u) ? b : ~b) &
                      ((row & 1u) ? c : ~c);
    }
    return result;
}

__global__ void evaluate_level(const Node* nodes, std::uint32_t begin,
                               std::uint64_t count, const std::uint8_t* luts,
                               const std::uint32_t* predicates,
                               std::uint64_t batch, std::uint64_t words,
                               std::uint32_t* scratch, const std::uint32_t* status) {
    if (*status != 0) return;
    const std::uint64_t total = count * words;
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += std::uint64_t(gridDim.x) * blockDim.x) {
        const std::uint32_t node_index = begin + static_cast<std::uint32_t>(i / words);
        const std::uint64_t word = i % words;
        const Node n = nodes[node_index];
        std::uint32_t value = 0;
        switch (n.op) {
        case Op::predicate: value = predicates[std::uint64_t(n.x) * words + word]; break;
        case Op::lut3:
            value = lut3(scratch[std::uint64_t(n.x) * words + word],
                         scratch[std::uint64_t(n.y) * words + word],
                         scratch[std::uint64_t(n.z) * words + word], luts[n.table]);
            break;
        case Op::conjunction:
            value = scratch[std::uint64_t(n.x) * words + word] &
                    scratch[std::uint64_t(n.y) * words + word];
            break;
        case Op::disjunction:
            value = scratch[std::uint64_t(n.x) * words + word] |
                    scratch[std::uint64_t(n.y) * words + word];
            break;
        case Op::negate: value = ~scratch[std::uint64_t(n.x) * words + word]; break;
        }
        if (word + 1 == words && batch % 32) value &= (1u << (batch % 32)) - 1u;
        scratch[std::uint64_t(node_index) * words + word] = value;
    }
}

__global__ void gather_roots(const std::uint32_t* roots, std::uint32_t nroots,
                             std::uint64_t batch, std::uint64_t words,
                             const std::uint32_t* scratch, std::uint32_t* out,
                             std::uint8_t* bytes, const std::uint32_t* status) {
    if (*status != 0) return;
    const std::uint64_t total = std::uint64_t(nroots) * words;
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += std::uint64_t(gridDim.x) * blockDim.x) {
        const std::uint64_t root = i / words, word = i % words;
        const std::uint32_t v = scratch[std::uint64_t(roots[root]) * words + word];
        out[i] = v;
        if (bytes) {
            for (unsigned bit = 0; bit < 32 && word * 32 + bit < batch; ++bit)
                bytes[(word * 32 + bit) * nroots + root] = (v >> bit) & 1u;
        }
    }
}

unsigned blocks_for(std::uint64_t work) {
    return static_cast<unsigned>(std::min<std::uint64_t>((work - 1) / kThreads + 1, 65535));
}
bool product_fits(std::uint64_t a, std::uint64_t b) {
    return !a || b <= std::numeric_limits<std::size_t>::max() / a;
}

} // namespace

cudaError_t train_step(float* logits, const std::uint8_t* rows,
                       const float* targets, std::uint64_t count,
                       float rate, RowStatistics* scratch,
                       std::uint32_t* status, cudaStream_t stream) noexcept {
    if (!logits || !scratch || !status || (count && (!rows || !targets)) ||
        !std::isfinite(rate) || rate <= 0.0f ||
        !product_fits(count, sizeof(float))) return cudaErrorInvalidValue;
    statistics_kernel<<<8, kThreads, 0, stream>>>(logits, rows, targets, count, scratch, status);
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    update_kernel<<<1, 32, 0, stream>>>(logits, scratch, rate, status);
    return cudaPeekAtLastError();
}

cudaError_t harden(const float* logits, std::uint64_t count,
                   std::uint8_t* luts, float* change,
                   GateRevisions revisions, CircuitStamp* stamp,
                   std::uint32_t* status, cudaStream_t stream) noexcept {
    if (!stamp) return cudaErrorInvalidValue;
    *stamp = CircuitStamp{};
    if (!status || (count && (!logits || !luts)) ||
        !product_fits(count, 8 * sizeof(float))) return cudaErrorInvalidValue;
    if (count) {
        harden_kernel<<<blocks_for(count), kThreads, 0, stream>>>(logits, count, luts, change, status);
        const auto error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
    }
    stamp->revisions = revisions;
    stamp->table_count = count;
    stamp->luts = luts;
    stamp->valid = true;
    return cudaSuccess;
}

cudaError_t predict_soft(const float* logits, std::uint64_t tables,
                         const std::uint32_t* table_indices,
                         const std::uint8_t* rows, std::uint64_t count,
                         float* probabilities, std::uint8_t* thresholded,
                         float* changes, std::uint32_t* status,
                         cudaStream_t stream) noexcept {
    if (!status || (count && (!logits || !tables || !table_indices || !rows ||
                              (!probabilities && !thresholded && !changes))) ||
        !product_fits(tables, 8 * sizeof(float)) ||
        !product_fits(count, sizeof(std::uint32_t))) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    predict_soft_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
        logits, tables, table_indices, rows, count, probabilities, thresholded, changes, status);
    return cudaPeekAtLastError();
}

bool prepare_program(const Node* nodes, std::uint32_t n,
                     const std::uint32_t* levels, std::size_t nlevels,
                     const std::uint32_t* roots, std::uint32_t nroots,
                     std::uint32_t npred, std::uint32_t ntables,
                     GateRevisions revisions, Program* out) {
    if (!out) return false;
    *out = Program{};
    if (!levels || !nlevels || levels[0] != 0 || levels[nlevels - 1] != n ||
        (n && !nodes) || (nroots && !roots)) return false;
    for (std::size_t level = 1; level < nlevels; ++level) {
        const auto begin = levels[level - 1], end = levels[level];
        if (begin >= end || end > n) return false;
        for (std::uint32_t i = begin; i < end; ++i) {
            const Node& node = nodes[i];
            if (node.reserved[0] || node.reserved[1] || node.reserved[2]) return false;
            switch (node.op) {
            case Op::predicate: if (node.x >= npred) return false; break;
            case Op::lut3:
                if (node.x >= begin || node.y >= begin || node.z >= begin || node.table >= ntables)
                    return false;
                break;
            case Op::conjunction: case Op::disjunction:
                if (node.x >= begin || node.y >= begin) return false;
                break;
            case Op::negate: if (node.x >= begin) return false; break;
            default: return false;
            }
        }
    }
    for (std::uint32_t i = 0; i < nroots; ++i) if (roots[i] >= n) return false;
    out->levels.assign(levels, levels + nlevels);
    out->node_count = n;
    out->predicate_count = npred;
    out->table_count = ntables;
    out->root_count = nroots;
    out->revisions = revisions;
    out->valid = true;
    return true;
}

cudaError_t evaluate(const Program& p, const CircuitStamp& circuit,
                     const GateRevisions& current,
                     const Node* nodes, const std::uint32_t* roots,
                     const std::uint8_t* luts, const std::uint32_t* predicates,
                     std::uint64_t batch, std::uint32_t* scratch,
                     std::uint32_t* out, std::uint8_t* bytes, std::uint32_t* status,
                     cudaStream_t stream) noexcept {
    // An accessibility update invalidates evaluated planes but does not change
    // the LUT. Structural/parameter/teacher/mapping changes invalidate both.
    const auto& r = circuit.revisions;
    if (!p.valid || !same_revisions(p.revisions, current) || !circuit.valid ||
        circuit.table_count != p.table_count || circuit.luts != luts ||
        r.structure_epoch != current.structure_epoch ||
        r.mapping_revision != current.mapping_revision ||
        r.parameter_revision != current.parameter_revision ||
        r.teacher_revision != current.teacher_revision || !status) return cudaErrorInvalidValue;
    if (batch > std::numeric_limits<std::uint64_t>::max() - 31) return cudaErrorInvalidValue;
    const std::uint64_t words = (batch + 31) / 32;
    if (!words || !p.node_count) return cudaSuccess;
    if (!nodes || !scratch || (p.root_count && (!roots || !out)) ||
        (p.predicate_count && !predicates) || (p.table_count && !luts) ||
        !product_fits(p.node_count, words) || !product_fits(p.node_count * words, sizeof(std::uint32_t)) ||
        !product_fits(p.predicate_count, words) || !product_fits(p.predicate_count * words, sizeof(std::uint32_t)) ||
        !product_fits(p.root_count, words) || !product_fits(p.root_count * words, sizeof(std::uint32_t)) ||
        !product_fits(batch, p.root_count)) return cudaErrorInvalidValue;
    for (std::size_t i = 1; i < p.levels.size(); ++i) {
        const std::uint32_t begin = p.levels[i - 1], n = p.levels[i] - begin;
        evaluate_level<<<blocks_for(std::uint64_t(n) * words), kThreads, 0, stream>>>(
            nodes, begin, n, luts, predicates, batch, words, scratch, status);
        const auto error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
    }
    if (p.root_count) {
        gather_roots<<<blocks_for(std::uint64_t(p.root_count) * words), kThreads, 0, stream>>>(
            roots, p.root_count, batch, words, scratch, out, bytes, status);
        return cudaPeekAtLastError();
    }
    return cudaSuccess;
}

} // namespace cgs::teacher
