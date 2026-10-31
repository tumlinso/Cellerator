#include <Cellerator/modeling/cuda/frontier.hh>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace m = cellerator::modeling::cuda;

namespace cellerator::modeling::cuda {
cudaError_t build_changed_frontier_original(FrontierPostings,
    const std::uint64_t*, std::uint64_t, FrontierWorkspace, FrontierOutput,
    cudaStream_t) noexcept;
}

void check_cuda(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("line ") + std::to_string(line)
            + ": " + expression + " -> " + cudaGetErrorString(error));
}
#define CHECK(expression) check_cuda((expression), #expression, __LINE__)

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        if (count) CHECK(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Conventional alternative: set one dense flag per affected interaction,
// prefix-scan every interaction, then compact flags in identity order.
__global__ void mark_changed(const std::uint64_t* offsets,
                             const std::uint64_t* ordinals,
                             const std::uint64_t* changed,
                             std::uint64_t changed_count,
                             std::uint64_t source_count,
                             std::uint32_t* flags) {
    const unsigned lane = threadIdx.x & 31;
    const std::uint64_t warp =
        (std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const std::uint64_t warp_stride = std::uint64_t(gridDim.x) * blockDim.x / 32;
    for (auto i = warp; i < changed_count; i += warp_stride) {
        const auto source = changed[i];
        if (source >= source_count) continue;
        const auto begin = offsets[source];
        const auto end = offsets[source + 1];
        for (auto p = begin + lane; p < end; p += 32)
            atomicExch(flags + ordinals[p], 1u);
    }
}

__global__ void compact(const std::uint32_t* flags,
                        const std::uint32_t* prefix,
                        std::uint64_t count,
                        std::uint64_t* output) {
    const auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && flags[i]) output[prefix[i]] = i;
}

struct Fixture {
    std::uint64_t sources;
    std::uint64_t interactions;
    std::uint64_t changed;
    std::uint64_t degree;
};

struct Prepared {
    explicit Prepared(const Fixture& f)
        : d_offsets(f.sources + 1), d_ordinals(f.changed * f.degree),
          d_changed(f.changed), d_output(f.interactions),
          d_claims(f.interactions), d_status(1), d_flags(f.interactions),
          d_prefix(f.interactions) {
        offsets.reserve(f.sources + 1);
        offsets.push_back(0);
        for (std::uint64_t source = 0; source < f.sources; ++source) {
            for (std::uint64_t k = 0; k < f.degree; ++k) {
                // A controllable overlap pattern spanning low and high output
                // ordinals, rather than a single contiguous high-degree row.
                const auto id = (source * 17 + k * 13 + (source % 7) * k) % f.interactions;
                ordinals.push_back(id);
            }
            offsets.push_back(ordinals.size());
        }
        changes.resize(f.changed);
        for (std::uint64_t i = 0; i < f.changed; ++i)
            changes[i] = (i * 37) % f.sources;
        index = {d_offsets.data, d_ordinals.data, f.sources,
                 ordinals.size(), f.interactions};
        upload(d_offsets.data, offsets);
        upload(d_ordinals.data, ordinals);
        upload(d_changed.data, changes);

        std::set<std::uint64_t> expected;
        for (const auto source : changes)
            for (auto p = offsets[source]; p < offsets[source + 1]; ++p)
                expected.insert(ordinals[p]);
        expected_output.assign(expected.begin(), expected.end());

        scan_bytes = 0;
        CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, d_flags.data,
                                             d_prefix.data, f.interactions));
        CHECK(cudaMalloc(&scan_temp, scan_bytes));
    }
    ~Prepared() { if (scan_temp) cudaFree(scan_temp); }
    Prepared(const Prepared&) = delete;
    Prepared& operator=(const Prepared&) = delete;

    template<class T> void upload(T* destination, const std::vector<T>& source) {
        if (!source.empty())
            CHECK(cudaMemcpy(destination, source.data(), source.size() * sizeof(T),
                             cudaMemcpyHostToDevice));
    }

    DeviceBuffer<std::uint64_t> d_offsets, d_ordinals, d_changed, d_output;
    DeviceBuffer<std::uint32_t> d_claims;
    DeviceBuffer<m::FrontierStatus> d_status;
    DeviceBuffer<std::uint32_t> d_flags, d_prefix;
    std::uint64_t* scan_temp = nullptr;
    std::size_t scan_bytes = 0;
    std::vector<std::uint64_t> offsets, ordinals, changes, expected_output;
    m::FrontierPostings index{};
};

void candidate(Prepared& p, std::uint64_t changed, cudaStream_t stream) {
    CHECK(m::build_changed_frontier(p.index, p.d_changed.data, changed,
          {p.d_claims.data, p.index.interaction_count},
          {p.d_output.data, p.index.interaction_count, p.d_status.data}, stream));
}

void original(Prepared& p, std::uint64_t changed, cudaStream_t stream) {
    CHECK(m::build_changed_frontier_original(p.index, p.d_changed.data, changed,
          {p.d_claims.data, p.index.interaction_count},
          {p.d_output.data, p.index.interaction_count, p.d_status.data}, stream));
}

void scan_baseline(Prepared& p, std::uint64_t changed, cudaStream_t stream) {
    CHECK(cudaMemsetAsync(p.d_flags.data, 0,
          p.index.interaction_count * sizeof(std::uint32_t), stream));
    if (changed) {
        constexpr unsigned threads = 128;
        constexpr std::uint64_t max_blocks = 65535;
        const auto needed = (changed + threads / 32 - 1) / (threads / 32);
        mark_changed<<<static_cast<unsigned>(std::min(needed, max_blocks)), threads, 0, stream>>>(
            p.d_offsets.data, p.d_ordinals.data, p.d_changed.data, changed,
            p.index.source_count, p.d_flags.data);
        CHECK(cudaGetLastError());
    }
    CHECK(cub::DeviceScan::ExclusiveSum(p.scan_temp, p.scan_bytes, p.d_flags.data,
                                         p.d_prefix.data,
                                         p.index.interaction_count, stream));
    constexpr unsigned threads = 256;
    const auto blocks = (p.index.interaction_count + threads - 1) / threads;
    if (blocks) {
        compact<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            p.d_flags.data, p.d_prefix.data, p.index.interaction_count, p.d_output.data);
        CHECK(cudaGetLastError());
    }
}

template<class Fn> float measure_batch(cudaStream_t stream, Fn&& run, int repeats) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i) run();
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaEventSynchronize(stop));
    float elapsed = 0;
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return elapsed * 1000.0f / repeats;
}

std::uint64_t result_count(Prepared& p, cudaStream_t stream) {
    std::uint32_t tail[2]{};
    const auto n = p.index.interaction_count;
    CHECK(cudaMemcpyAsync(tail, p.d_prefix.data + n - 1, sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(tail + 1, p.d_flags.data + n - 1, sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    return std::uint64_t(tail[0]) + tail[1];
}

void check_output(Prepared& p, std::uint64_t count, cudaStream_t stream) {
    std::vector<std::uint64_t> actual(count);
    if (count)
        CHECK(cudaMemcpyAsync(actual.data(), p.d_output.data, count * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    std::sort(actual.begin(), actual.end());
    if (actual != p.expected_output)
        throw std::runtime_error("frontier candidate differs from independent set oracle");
}

int main(int argc, char** argv) {
    int repeats = 80;
    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--check-only") {
            check_only = true;
        } else {
            repeats = std::max(1, std::atoi(argv[i]));
        }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (check_only)
        std::cout << "mode,sources,interactions,selected,degree,union,original,"
                     "warp_reserved,scan_compact\n";
    else
        std::cout << "sources,interactions,selected,degree,union,original_us,"
                     "warp_reserved_us,scan_compact_us\n";

    for (const Fixture fixture : {
        Fixture{4096, 65536, 1, 1},
        Fixture{4096, 65536, 32, 4},
        Fixture{4096, 65536, 256, 8},
        Fixture{4096, 65536, 1024, 2},
        Fixture{4096, 65536, 256, 64},
        Fixture{4096, 65536, 1024, 64},
    }) {
        Prepared p(fixture);
        if (check_only) {
            original(p, fixture.changed, stream);
            m::FrontierStatus original_status{};
            CHECK(cudaMemcpyAsync(&original_status, p.d_status.data, sizeof(original_status),
                                  cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
            if (original_status.flags != m::frontier_success
                || original_status.written != p.expected_output.size()
                || original_status.required != p.expected_output.size())
                throw std::runtime_error("original frontier status differs from oracle");
            check_output(p, original_status.written, stream);

            candidate(p, fixture.changed, stream);
            m::FrontierStatus candidate_status{};
            CHECK(cudaMemcpyAsync(&candidate_status, p.d_status.data, sizeof(candidate_status),
                                  cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
            if (candidate_status.flags != m::frontier_success
                || candidate_status.written != p.expected_output.size()
                || candidate_status.required != p.expected_output.size())
                throw std::runtime_error("warp-reserved frontier status differs from oracle");
            check_output(p, candidate_status.written, stream);

            scan_baseline(p, fixture.changed, stream);
            const auto scanned_count = result_count(p, stream);
            if (scanned_count != p.expected_output.size())
                throw std::runtime_error("scan baseline cardinality differs from oracle");
            check_output(p, scanned_count, stream);

            std::cout << "check-only," << fixture.sources << ',' << fixture.interactions
                      << ',' << fixture.changed << ',' << fixture.degree << ','
                      << p.expected_output.size() << ",PASS,PASS,PASS\n";
            continue;
        }
        for (int i = 0; i < 5; ++i) {
            original(p, fixture.changed, stream);
            candidate(p, fixture.changed, stream);
            scan_baseline(p, fixture.changed, stream);
        }
        const float original_us = measure_batch(stream,
            [&] { original(p, fixture.changed, stream); }, repeats);
        m::FrontierStatus original_status{};
        CHECK(cudaMemcpyAsync(&original_status, p.d_status.data, sizeof(original_status),
                              cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        if (original_status.flags != m::frontier_success
            || original_status.written != p.expected_output.size()
            || original_status.required != p.expected_output.size())
            throw std::runtime_error("original frontier status differs from oracle");
        check_output(p, original_status.written, stream);

        const float warp_us = measure_batch(stream,
            [&] { candidate(p, fixture.changed, stream); }, repeats);
        m::FrontierStatus frontier{};
        CHECK(cudaMemcpyAsync(&frontier, p.d_status.data, sizeof(frontier),
                              cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        if (frontier.flags != m::frontier_success
            || frontier.written != p.expected_output.size()
            || frontier.required != p.expected_output.size())
            throw std::runtime_error("atomic frontier status differs from oracle");
        check_output(p, frontier.written, stream);

        const float scan_us = measure_batch(stream,
            [&] { scan_baseline(p, fixture.changed, stream); }, repeats);
        const auto scanned_count = result_count(p, stream);
        if (scanned_count != p.expected_output.size())
            throw std::runtime_error("scan baseline cardinality differs from oracle");
        check_output(p, scanned_count, stream);

        std::cout << fixture.sources << ',' << fixture.interactions << ','
                  << fixture.changed << ',' << fixture.degree << ','
                  << p.expected_output.size() << ',' << original_us << ','
                  << warp_us << ',' << scan_us << '\n';
    }
    CHECK(cudaStreamDestroy(stream));
}
