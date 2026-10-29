#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/cuda/frontier.hh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

namespace m = cellerator::modeling::cuda;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(std::size_t count) { if (count) check(cudaMalloc(&data, count*sizeof(T))); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& values, cudaStream_t stream) {
        if (!values.empty()) check(cudaMemcpyAsync(data, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice, stream));
    }
};

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<std::uint64_t> offsets{0,3,5,6,6}, postings{2,0,0,1,2,4}, changes{0,1,0};
    Buffer<std::uint64_t> d_offsets(offsets.size()), d_postings(postings.size()), d_changes(3), output(6);
    Buffer<std::uint32_t> claims(5);
    Buffer<m::FrontierStatus> status(1);
    d_offsets.upload(offsets, stream);
    d_postings.upload(postings, stream);
    m::FrontierPostings index{d_offsets.data,d_postings.data,4,6,5};
    const std::vector<std::uint64_t> sentinel(6, 999);
    auto run = [&](const std::vector<std::uint64_t>& changed, std::uint64_t capacity) {
        d_changes.upload(changed, stream);
        output.upload(sentinel, stream);
        check(m::build_changed_frontier(index, d_changes.data, changed.size(),
              {claims.data,5}, {output.data,capacity,status.data}, stream));
        m::FrontierStatus result;
        check(cudaMemcpyAsync(&result, status.data, sizeof(result), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return result;
    };
    const auto result = run(changes,5);
    std::vector<std::uint64_t> actual(6);
    check(cudaMemcpy(actual.data(), output.data, actual.size()*sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    assert(result.flags == 0 && result.written == 3 && actual[5] == 999);
    std::set<std::uint64_t> oracle;
    for (auto source : changes)
        for (auto i = offsets[source]; i < offsets[source+1]; ++i) oracle.insert(postings[i]);
    std::sort(actual.begin(), actual.begin()+result.written);
    assert(std::vector<std::uint64_t>(actual.begin(), actual.begin()+result.written)
           == std::vector<std::uint64_t>(oracle.begin(), oracle.end()));
    const auto overflow = run(changes,1);
    assert(overflow.flags == m::frontier_insufficient_capacity && overflow.required == 3 && overflow.written == 0);
    check(cudaMemcpy(actual.data(), output.data, actual.size()*sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    assert(actual[1] == 999 && actual[5] == 999);
    const auto zero_capacity = run(changes,0);
    assert(zero_capacity.required == 3 && zero_capacity.written == 0);
    assert(run({},5).written == 0);
    assert(run({3},5).flags == 0); // empty posting row
    assert(run({99},5).flags == m::frontier_invalid_source);
    offsets[3] = 9; d_offsets.upload(offsets,stream);
    const auto bad_offset = run({2},5);
    assert(bad_offset.flags == m::frontier_invalid_posting && bad_offset.written == 0);
    offsets[3] = 6; d_offsets.upload(offsets,stream);
    postings[4] = 9; d_postings.upload(postings,stream);
    assert(run({1},5).flags == m::frontier_invalid_posting);
    assert(m::build_changed_frontier(index,d_changes.data,1,{claims.data,4},
        {output.data,5,status.data},stream) == cudaErrorInvalidValue);
    check(cudaStreamDestroy(stream));
    std::cout << "CUDA changed-source frontier union, deduplication, capacity, and content checks passed\n";
}
