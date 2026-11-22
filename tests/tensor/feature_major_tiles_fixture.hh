#pragma once

#include <Cellerator/layout/projection/feature_major_tiles.hh>
#include <Cellerator/ops/matrix/feature_major_tiles.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cellerator::test::feature_tiles {

namespace layout = cellerator::layout::projection;
namespace matrix = cellerator::ops::matrix;
using layout::FeatureMajorTileGrouping;
using layout::FeatureMajorTileRange;
using layout::FeatureMajorTilesIndexView;
using layout::FeatureMajorTilesRequirements;
using layout::FeatureMajorTilesStatus;
using tensor::FeatureMajorTileStructureView;
using tensor::FeatureMajorTilesView;
constexpr uint32_t width = 16;
constexpr float learning_rate = 0.002f;

inline void require(bool pass, const char* message) {
    if (!pass) throw std::runtime_error(message);
}

inline void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

inline void check(FeatureMajorTilesStatus status, const char* operation) {
    if (status != FeatureMajorTilesStatus::ok)
        throw std::runtime_error(std::string(operation) + ": " +
            std::to_string(static_cast<int>(status)));
}

class Stream {
public:
    Stream() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { cudaStreamDestroy(stream); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    operator cudaStream_t() const { return stream; }
    void sync() const { check(cudaStreamSynchronize(stream), "synchronize stream"); }

private:
    cudaStream_t stream = nullptr;
};

template<class T>
class DeviceArray {
public:
    explicit DeviceArray(size_t count = 0) : count_(count) {
        if (count != 0) check(cudaMalloc(&data, count * sizeof(T)), "allocate device array");
    }
    ~DeviceArray() { if (data) cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    T* get() { return data; }
    const T* get() const { return data; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

    void upload(const std::vector<T>& input, cudaStream_t stream) const {
        require(input.size() == count_, "incorrect upload size");
        if (count_) check(cudaMemcpyAsync(data, input.data(), bytes(), cudaMemcpyHostToDevice,
                                         stream), "upload device array");
    }
    std::vector<T> read(const Stream& stream) const {
        std::vector<T> output(count_);
        if (count_) check(cudaMemcpyAsync(output.data(), data, bytes(), cudaMemcpyDeviceToHost,
                                         stream), "download device array");
        stream.sync();
        return output;
    }

private:
    size_t count_;
    T* data = nullptr;
};

struct Support {
    uint32_t rows = 0, features = 0;
    std::vector<uint32_t> offsets, columns;

    layout::FeatureMajorTilesCsr view() const {
        return {rows, features, columns.size(), tensor::base::CompressedAxis::row,
                offsets.data(), columns.data()};
    }
};

struct HostTiles {
    FeatureMajorTilesRequirements required{};
    std::vector<uint32_t> indices;
    std::vector<FeatureMajorTileRange> chunks;

    HostTiles(const Support& support, FeatureMajorTileGrouping grouping) {
        size_t count = 0;
        check(layout::feature_major_tiles_workspace_count(support.view(), grouping, &count),
              "query workspace");
        std::vector<uint32_t> workspace(count);
        check(layout::query_feature_major_tiles(support.view(), grouping, workspace.data(),
              workspace.size(), &required), "query layout");
        indices.resize(required.index_count);
        chunks.resize(required.chunk_count);
        check(layout::prepare_feature_major_tiles(support.view(), required, workspace.data(),
              workspace.size(), {indices.data(), indices.size(), chunks.data(), chunks.size()}),
              "prepare layout");
    }
};

struct DeviceTiles {
    DeviceArray<uint32_t> indices;
    DeviceArray<FeatureMajorTileRange> chunks;
    FeatureMajorTileStructureView structure{};
    FeatureMajorTilesIndexView index{};

    DeviceTiles(const HostTiles& host, const Stream& stream)
        : indices(host.indices.size()), chunks(host.chunks.size()) {
        indices.upload(host.indices, stream);
        chunks.upload(host.chunks, stream);
        check(layout::bind_feature_major_tiles(host.required, indices.get(), indices.count(),
              chunks.get(), chunks.count(), &structure, &index), "bind layout");
    }

    size_t bytes() const { return indices.bytes() + chunks.bytes(); }
};

struct Parameters {
    std::vector<float> x, w, target;
};

inline float parameter(size_t index, uint32_t multiplier, uint32_t mod, float scale) {
    return (float((index * multiplier) % mod) - float(mod / 2)) * scale;
}

inline Parameters parameters(const Support& support) {
    Parameters p{std::vector<float>(support.columns.size()),
                 std::vector<float>(size_t(support.features) * width),
                 std::vector<float>(size_t(support.rows) * width)};
    for (size_t i = 0; i < p.x.size(); ++i) p.x[i] = parameter(i, 13, 41, 0.0031f);
    if (!p.x.empty()) p.x[0] = 0.0f;
    for (size_t i = 0; i < p.w.size(); ++i) p.w[i] = parameter(i, 7, 29, 0.01f);
    for (size_t i = 0; i < p.target.size(); ++i) p.target[i] = parameter(i, 19, 53, 0.002f);
    return p;
}

struct Reference {
    std::vector<double> y, g, dw, dx;
};

inline Reference reference(const Support& support, const Parameters& p) {
    Reference out{std::vector<double>(p.target.size()), std::vector<double>(p.target.size()),
                  std::vector<double>(p.w.size()), std::vector<double>(p.x.size())};
    for (uint32_t row = 0; row < support.rows; ++row)
        for (uint32_t e = support.offsets[row]; e < support.offsets[row + 1]; ++e) {
            const double half = __half2float(__float2half_rn(p.x[e]));
            for (uint32_t c = 0; c < width; ++c)
                out.y[size_t(row) * width + c] += half * p.w[size_t(support.columns[e]) * width + c];
        }
    for (size_t i = 0; i < out.g.size(); ++i)
        out.g[i] = (out.y[i] - p.target[i]) / width;
    for (uint32_t row = 0; row < support.rows; ++row)
        for (uint32_t e = support.offsets[row]; e < support.offsets[row + 1]; ++e) {
            const double half = __half2float(__float2half_rn(p.x[e]));
            for (uint32_t c = 0; c < width; ++c) {
                const size_t at = size_t(support.columns[e]) * width + c;
                const double g = out.g[size_t(row) * width + c];
                out.dw[at] += half * g;
                out.dx[e] += g * p.w[at];
            }
        }
    return out;
}

template<class Expected>
void compare(const std::vector<float>& actual, const std::vector<Expected>& expected,
             const char* label, double absolute = 2e-6, double relative = 8e-5) {
    require(actual.size() == expected.size(), "reference extent mismatch");
    for (size_t i = 0; i < actual.size(); ++i)
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - double(expected[i])) > absolute + relative * std::abs(expected[i]))
            throw std::runtime_error(std::string(label) + " differs at " + std::to_string(i) +
                " expected " + std::to_string(expected[i]) + " got " + std::to_string(actual[i]));
}

inline void reference_sgd(Parameters& p, const Reference& ref) {
    for (size_t i = 0; i < p.x.size(); ++i) p.x[i] -= learning_rate * float(ref.dx[i]);
    for (size_t i = 0; i < p.w.size(); ++i) p.w[i] -= learning_rate * float(ref.dw[i]);
}

static __global__ void squared_loss_gradient(const float* y, const float* target,
                                              float* g, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) g[i] = (y[i] - target[i]) / width;
}

static __global__ void dense_sgd(float* w, const float* dw, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) w[i] -= learning_rate * dw[i];
}

inline void loss(const DeviceArray<float>& y, const DeviceArray<float>& target,
                 DeviceArray<float>& g, cudaStream_t stream) {
    if (y.count()) {
        squared_loss_gradient<<<(y.count() + 255) / 256, 256, 0, stream>>>(
            y.get(), target.get(), g.get(), y.count());
        check(cudaGetLastError(), "enqueue loss gradient");
    }
}

inline void update_w(DeviceArray<float>& w, const DeviceArray<float>& dw, cudaStream_t stream) {
    if (w.count()) {
        dense_sgd<<<(w.count() + 255) / 256, 256, 0, stream>>>(
            w.get(), dw.get(), w.count());
        check(cudaGetLastError(), "enqueue weight update");
    }
}

} // namespace cellerator::test::feature_tiles
