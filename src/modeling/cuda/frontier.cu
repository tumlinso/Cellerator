#include <Cellerator/modeling/cuda/frontier.hh>

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <limits>

namespace cellerator::modeling::cuda {
namespace {
constexpr unsigned threads = 128;
constexpr unsigned warp_width = 32;

// A warp cooperates on one changed-source row. No warp collective or implicit
// lockstep communication: all deduplication uses device-wide atomic claims.
__global__ void collect(FrontierPostings postings, const std::uint64_t* changed,
                        std::uint64_t changed_count, std::uint32_t* claims,
                        FrontierOutput output) {
    const auto lane = threadIdx.x % warp_width;
    const std::uint64_t warp = (std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / warp_width;
    const std::uint64_t stride = std::uint64_t(gridDim.x) * blockDim.x / warp_width;
    for (auto change = warp; change < changed_count; change += stride) {
        const auto source = changed[change];
        if (source >= postings.source_count) {
            if (!lane) atomicOr(&output.status->flags, frontier_invalid_source);
            continue;
        }
        const auto begin = postings.offsets[source], end = postings.offsets[source + 1];
        if (begin > end || end > postings.posting_count) {
            if (!lane) atomicOr(&output.status->flags, frontier_invalid_posting);
            continue;
        }
        // begin/end are bounded by an addressable array checked on the host;
        // adding lane/warp_width therefore cannot wrap a 64-bit index.
        for (auto entry = begin + lane; entry < end; entry += warp_width) {
            const auto interaction = postings.interaction_ordinals[entry];
            if (interaction >= postings.interaction_count) {
                atomicOr(&output.status->flags, frontier_invalid_posting);
                continue;
            }
            if (atomicCAS(claims + interaction, 0u, 1u) != 0u) continue;
            const auto position = atomicAdd(
                reinterpret_cast<unsigned long long*>(&output.status->required), 1ULL);
            if (position < output.capacity) output.ordinals[position] = interaction;
            else atomicOr(&output.status->flags, frontier_insufficient_capacity);
        }
    }
}

__global__ void finish(FrontierStatus* status) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        status->written = status->flags == frontier_success ? status->required : 0;
}
} // namespace

cudaError_t build_changed_frontier(FrontierPostings postings,
                                   const std::uint64_t* changed_sources,
                                   std::uint64_t changed_count,
                                   FrontierWorkspace workspace,
                                   FrontierOutput output,
                                   cudaStream_t stream) noexcept {
    constexpr auto max_size = std::numeric_limits<std::size_t>::max();
    if (!output.status || (changed_count && !changed_sources)
        || (postings.source_count && !postings.offsets)
        || (postings.posting_count && !postings.interaction_ordinals)
        || (postings.interaction_count && !workspace.claims)
        || (output.capacity && !output.ordinals)
        || workspace.capacity < postings.interaction_count
        || postings.source_count >= max_size / sizeof(std::uint64_t)
        || postings.posting_count > max_size / sizeof(std::uint64_t)
        || postings.interaction_count > max_size / sizeof(std::uint32_t)
        || changed_count > max_size / sizeof(std::uint64_t)
        || output.capacity > max_size / sizeof(std::uint64_t))
        return cudaErrorInvalidValue;
    auto error = cudaMemsetAsync(output.status, 0, sizeof(FrontierStatus), stream);
    if (error != cudaSuccess) return error;
    if (postings.interaction_count) {
        error = cudaMemsetAsync(workspace.claims, 0,
            static_cast<std::size_t>(postings.interaction_count) * sizeof(std::uint32_t), stream);
        if (error != cudaSuccess) return error;
    }
    if (changed_count) {
        constexpr unsigned warps_per_block = threads / warp_width;
        const auto needed = (changed_count + warps_per_block - 1) / warps_per_block;
        const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(needed, 65535));
        collect<<<blocks, threads, 0, stream>>>(postings, changed_sources, changed_count,
                                               workspace.claims, output);
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
    }
    finish<<<1, 1, 0, stream>>>(output.status);
    return cudaGetLastError();
}

} // namespace cellerator::modeling::cuda
