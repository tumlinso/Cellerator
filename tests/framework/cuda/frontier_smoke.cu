#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/cuda/frontier.hh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace m = cellerator::modeling::cuda;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(std::size_t count) {
        if (count) check(cudaMalloc(&data, count * sizeof(T)));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& values, cudaStream_t stream) {
        if (!values.empty())
            check(cudaMemcpyAsync(data, values.data(), values.size() * sizeof(T),
                                  cudaMemcpyHostToDevice, stream));
    }
};

struct Case {
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> postings;
    std::vector<std::uint64_t> changes;
    std::uint64_t interaction_count = 0;
    std::uint64_t capacity = 0;
};

std::set<std::uint64_t> oracle(const Case& test) {
    std::set<std::uint64_t> expected;
    for (const auto source : test.changes) {
        assert(source < test.offsets.size() - 1);
        assert(test.offsets[source] <= test.offsets[source + 1]);
        assert(test.offsets[source + 1] <= test.postings.size());
        for (auto i = test.offsets[source]; i < test.offsets[source + 1]; ++i) {
            assert(test.postings[i] < test.interaction_count);
            expected.insert(test.postings[i]);
        }
    }
    return expected;
}

m::FrontierStatus run(cudaStream_t stream, const Case& test) {
    Buffer<std::uint64_t> offsets(test.offsets.size());
    Buffer<std::uint64_t> postings(test.postings.size());
    Buffer<std::uint64_t> changed(test.changes.size());
    Buffer<std::uint64_t> output(test.capacity);
    Buffer<std::uint32_t> claims(test.interaction_count);
    Buffer<m::FrontierStatus> status(1);
    offsets.upload(test.offsets, stream);
    postings.upload(test.postings, stream);
    changed.upload(test.changes, stream);
    m::FrontierPostings index{
        offsets.data, postings.data,
        test.offsets.empty() ? 0 : test.offsets.size() - 1,
        test.postings.size(), test.interaction_count};
    check(m::build_changed_frontier(index, changed.data, test.changes.size(),
          {claims.data, test.interaction_count},
          {output.data, test.capacity, status.data}, stream));
    m::FrontierStatus result{};
    check(cudaMemcpyAsync(&result, status.data, sizeof(result), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    if (result.flags == m::frontier_success) {
        assert(result.written == result.required);
        assert(result.required <= test.capacity);
        std::vector<std::uint64_t> actual(result.written);
        if (!actual.empty())
            check(cudaMemcpy(actual.data(), output.data,
                             actual.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
        const std::set<std::uint64_t> actual_set(actual.begin(), actual.end());
        assert(actual_set.size() == actual.size());
        assert(actual_set == oracle(test));
    } else {
        assert(result.written == 0);
    }
    return result;
}

void check_generated_shape(cudaStream_t stream, std::uint64_t changed_count,
                           std::uint64_t degree, std::uint64_t interactions) {
    Case test;
    test.interaction_count = interactions;
    test.capacity = interactions;
    test.offsets.reserve(changed_count + 1);
    test.offsets.push_back(0);
    for (std::uint64_t row = 0; row < changed_count; ++row) {
        for (std::uint64_t edge = 0; edge < degree; ++edge) {
            // Rows overlap heavily and repeat within a row, exercising both
            // same-warp and cross-warp duplicate claims.
            test.postings.push_back((row * 7 + edge * 3) % interactions);
        }
        test.offsets.push_back(test.postings.size());
        test.changes.push_back(row);
    }
    if (changed_count > 1) {
        test.changes.push_back(0);
        test.changes.push_back(changed_count - 1);
    }
    const auto result = run(stream, test);
    assert(result.flags == m::frontier_success);
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const Case overlap{{0, 4, 7, 7, 9},
                       {2, 0, 0, 2, 1, 2, 4, 4, 2},
                       {0, 1, 0, 2, 3}, 5, 5};
    const auto success = run(stream, overlap);
    assert(success.flags == m::frontier_success && success.required == 4);

    auto exact_capacity = overlap;
    exact_capacity.capacity = 4;
    const auto exact = run(stream, exact_capacity);
    assert(exact.flags == m::frontier_success && exact.required == 4
           && exact.written == 4);

    auto undersized = overlap;
    undersized.capacity = 3;
    const auto overflow = run(stream, undersized);
    assert(overflow.flags == m::frontier_insufficient_capacity
           && overflow.required == 4 && overflow.written == 0);

    auto zero_capacity = overlap;
    zero_capacity.capacity = 0;
    const auto no_output = run(stream, zero_capacity);
    assert(no_output.flags == m::frontier_insufficient_capacity
           && no_output.required == 4 && no_output.written == 0);

    const auto empty = run(stream, Case{{0}, {}, {}, 0, 0});
    assert(empty.flags == m::frontier_success && !empty.required && !empty.written);
    const auto empty_rows = run(stream, Case{{0, 0, 0, 0}, {}, {0, 1, 2}, 5, 5});
    assert(empty_rows.flags == m::frontier_success && !empty_rows.required);

    auto invalid_source = overlap;
    invalid_source.changes = {0, 99};
    const auto bad_source = run(stream, invalid_source);
    assert(bad_source.flags == m::frontier_invalid_source && !bad_source.written);

    auto decreasing = overlap;
    decreasing.offsets[2] = 5;
    decreasing.offsets[3] = 4;
    decreasing.changes = {2};
    assert(run(stream, decreasing).flags == m::frontier_invalid_posting);

    auto beyond_end = overlap;
    beyond_end.offsets[2] = beyond_end.postings.size() + 1;
    beyond_end.offsets[3] = beyond_end.postings.size() + 1;
    beyond_end.changes = {1};
    assert(run(stream, beyond_end).flags == m::frontier_invalid_posting);

    auto invalid_interaction = overlap;
    invalid_interaction.postings[1] = invalid_interaction.interaction_count;
    invalid_interaction.changes = {0};
    assert(run(stream, invalid_interaction).flags == m::frontier_invalid_posting);

    // Warp and block tails, plus uneven row lengths and overlapping postings.
    for (const auto count : {std::uint64_t{1}, std::uint64_t{31},
                             std::uint64_t{32}, std::uint64_t{33},
                             std::uint64_t{127}, std::uint64_t{128},
                             std::uint64_t{129}, std::uint64_t{1025}})
        check_generated_shape(stream, count, 5, 37);
    check_generated_shape(stream, 33, 97, 257);

    assert(m::build_changed_frontier(
        {nullptr, overlap.postings.data(), 1, overlap.postings.size(), 5},
        nullptr, 1, {nullptr, 5}, {nullptr, 0, nullptr}, stream)
        == cudaErrorInvalidValue);
    assert(m::build_changed_frontier(
        {nullptr, nullptr, 1, 0, 0}, nullptr, 0, {nullptr, 0},
        {nullptr, 0, nullptr}, stream) == cudaErrorInvalidValue);

    check(cudaStreamDestroy(stream));
    std::cout << "CUDA changed-source frontier oracle matrix passed\n";
}
