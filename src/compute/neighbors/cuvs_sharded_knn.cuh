#pragma once

#include "../../../extern/CellShard/src/CellShard.hh"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

namespace cellerator {
namespace compute {
namespace neighbors {

template<typename T>
class host_buffer {
public:
    host_buffer() = default;

    explicit host_buffer(std::size_t count) {
        resize(count);
    }

    host_buffer(const host_buffer &other) {
        assign_copy(other.data(), other.size());
    }

    host_buffer(host_buffer &&other) noexcept
        : data_(std::move(other.data_)),
          size_(other.size_),
          capacity_(other.capacity_) {
        other.size_ = 0u;
        other.capacity_ = 0u;
    }

    host_buffer &operator=(const host_buffer &other) {
        if (this != &other) assign_copy(other.data(), other.size());
        return *this;
    }

    host_buffer &operator=(host_buffer &&other) noexcept {
        if (this == &other) return *this;
        data_ = std::move(other.data_);
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.size_ = 0u;
        other.capacity_ = 0u;
        return *this;
    }

    void clear() {
        size_ = 0u;
    }

    void reserve(std::size_t capacity) {
        if (capacity <= capacity_) return;
        std::unique_ptr<T[]> next(new T[capacity]);
        if constexpr (std::is_trivially_copyable_v<T>) {
            if (size_ != 0u) std::memcpy(next.get(), data_.get(), size_ * sizeof(T));
        } else {
            for (std::size_t i = 0; i < size_; ++i) next[i] = std::move(data_[i]);
        }
        data_ = std::move(next);
        capacity_ = capacity;
    }

    void resize(std::size_t count) {
        if (count > capacity_) reserve(count);
        size_ = count;
    }

    void assign(std::size_t count, const T &value) {
        resize(count);
        for (std::size_t i = 0; i < count; ++i) data_[i] = value;
    }

    void assign_copy(const T *src, std::size_t count) {
        resize(count);
        if (count == 0u || src == nullptr) return;
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(data_.get(), src, count * sizeof(T));
        } else {
            for (std::size_t i = 0; i < count; ++i) data_[i] = src[i];
        }
    }

    void swap(host_buffer &other) noexcept {
        data_.swap(other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    T *data() {
        return data_.get();
    }

    const T *data() const {
        return data_.get();
    }

    std::size_t size() const {
        return size_;
    }

    bool empty() const {
        return size_ == 0u;
    }

    T &operator[](std::size_t idx) {
        return data_[idx];
    }

    const T &operator[](std::size_t idx) const {
        return data_[idx];
    }

private:
    std::unique_ptr<T[]> data_;
    std::size_t size_ = 0u;
    std::size_t capacity_ = 0u;
};

enum class metric_kind {
    l2_unexpanded,
    cosine_expanded,
    inner_product
};

struct knn_result_host {
    unsigned long rows;
    int k;
    host_buffer<std::int64_t> neighbors;
    host_buffer<float> distances;
};

struct sparse_exact_params {
    int k;
    metric_kind metric;
    float metric_arg;
    int exclude_self;
    int gpu_limit;
    int batch_size_index;
    int batch_size_query;
    int drop_host_parts_after_index_pack;
    int drop_host_parts_after_query_use;
};

struct dense_ann_params {
    int k;
    metric_kind metric;
    float metric_arg;
    int exclude_self;
    int gpu_limit;
    unsigned int n_lists;
    unsigned int n_probes;
    int use_cagra;
    unsigned int intermediate_graph_degree;
    unsigned int graph_degree;
    std::int64_t rows_per_batch;
};

struct proprietary_dense_params {
    int k;
    metric_kind metric;
    float metric_arg;
    int exclude_self;
    int gpu_limit;
    int query_block_rows;
    int index_block_rows;
};

void init(sparse_exact_params *params);
void init(dense_ann_params *params);
void init(proprietary_dense_params *params);
void init(knn_result_host *result);
void clear(knn_result_host *result);

int sparse_exact_self_knn(const ::cellshard::sharded< ::cellshard::sparse::compressed > *view,
                          const ::cellshard::shard_storage *storage,
                          const sparse_exact_params *params,
                          knn_result_host *result);

int dense_ann_self_knn(const ::cellshard::sharded< ::cellshard::dense > *view,
                       const dense_ann_params *params,
                       knn_result_host *result);

int proprietary_dense_self_knn(const ::cellshard::sharded< ::cellshard::dense > *view,
                               const proprietary_dense_params *params,
                               knn_result_host *result);

} // namespace neighbors
} // namespace compute
} // namespace cellerator
