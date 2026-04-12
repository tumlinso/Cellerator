#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cellerator::compute::neighbors::forward_neighbors {

template<typename T>
class host_array {
public:
    host_array() = default;

    explicit host_array(std::size_t count) {
        resize(count);
    }

    host_array(std::initializer_list<T> values) {
        assign_copy(values.begin(), values.size());
    }

    host_array(const host_array &other) {
        assign_copy(other.data(), other.size());
    }

    host_array(host_array &&other) noexcept
        : data_(std::move(other.data_)),
          size_(other.size_),
          capacity_(other.capacity_) {
        other.size_ = 0u;
        other.capacity_ = 0u;
    }

    host_array &operator=(const host_array &other) {
        if (this != &other) assign_copy(other.data(), other.size());
        return *this;
    }

    host_array &operator=(host_array &&other) noexcept {
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

    void assign_copy(const T *src, std::size_t count) {
        resize(count);
        if (count == 0u || src == nullptr) return;
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(data_.get(), src, count * sizeof(T));
        } else {
            for (std::size_t i = 0; i < count; ++i) data_[i] = src[i];
        }
    }

    void assign_fill(std::size_t count, const T &value) {
        resize(count);
        for (std::size_t i = 0; i < count; ++i) data_[i] = value;
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

    std::size_t capacity() const {
        return capacity_;
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

    T &back() {
        return data_[size_ - 1u];
    }

    const T &back() const {
        return data_[size_ - 1u];
    }

    T *begin() {
        return data_.get();
    }

    const T *begin() const {
        return data_.get();
    }

    T *end() {
        return data_.get() + size_;
    }

    const T *end() const {
        return data_.get() + size_;
    }

private:
    std::unique_ptr<T[]> data_;
    std::size_t size_ = 0u;
    std::size_t capacity_ = 0u;
};

template<typename T>
struct const_array_view {
    const T *data = nullptr;
    std::size_t size = 0u;

    const T &operator[](std::size_t idx) const {
        return data[idx];
    }

    bool empty() const {
        return size == 0u;
    }
};

template<typename T>
inline const_array_view<T> view_of(const host_array<T> &values) {
    return const_array_view<T>{ values.data(), values.size() };
}

enum class ForwardNeighborBackend {
    exact_windowed,
    ann_windowed
};

enum class ForwardNeighborEmbryoPolicy {
    any_embryo,
    same_embryo_first,
    same_embryo_only
};

struct ForwardTimeWindow {
    float min_delta = 0.0f;
    float max_delta = std::numeric_limits<float>::infinity();
};

struct ForwardNeighborQueryBatch {
    host_array<std::int64_t> cell_indices;
    host_array<float> developmental_time;
    host_array<float> latent_unit;
    host_array<std::int64_t> embryo_ids;
    std::int64_t latent_dim = 0;
};

struct ForwardNeighborSearchConfig {
    ForwardNeighborBackend backend = ForwardNeighborBackend::exact_windowed;
    ForwardNeighborEmbryoPolicy embryo_policy = ForwardNeighborEmbryoPolicy::any_embryo;
    std::int64_t top_k = 15;
    std::int64_t candidate_k = 15;
    std::int64_t query_block_rows = 256;
    std::int64_t index_block_rows = 16384;
    std::int64_t ann_probe_list_count = 8;
    float strict_future_epsilon = 0.0f;
    ForwardTimeWindow time_window{};
};

struct ForwardNeighborSearchResult {
    std::int64_t query_count = 0;
    std::int64_t top_k = 0;
    host_array<std::int64_t> query_cell_indices;
    host_array<float> query_time;
    host_array<std::int64_t> query_embryo_ids;
    host_array<std::int64_t> neighbor_cell_indices;
    host_array<float> neighbor_time;
    host_array<std::int64_t> neighbor_embryo_ids;
    host_array<float> neighbor_similarity;
    host_array<float> neighbor_sqdist;
    host_array<float> neighbor_distance;
};

namespace detail {

inline std::size_t checked_size_(std::int64_t value, const char *label) {
    if (value < 0) throw std::invalid_argument(std::string(label) + " must be >= 0");
    return static_cast<std::size_t>(value);
}

inline std::size_t result_offset_(std::int64_t row, std::int64_t slot, std::int64_t top_k) {
    return checked_size_(row, "row") * checked_size_(top_k, "top_k") + checked_size_(slot, "slot");
}

inline float negative_infinity_() {
    return -std::numeric_limits<float>::infinity();
}

inline float positive_infinity_() {
    return std::numeric_limits<float>::infinity();
}

inline float quiet_nan_() {
    return std::numeric_limits<float>::quiet_NaN();
}

inline host_array<std::int64_t> make_missing_i64_array_(std::size_t rows) {
    host_array<std::int64_t> out;
    out.assign_fill(rows, static_cast<std::int64_t>(-1));
    return out;
}

inline void validate_forward_neighbor_search_config_(const ForwardNeighborSearchConfig &config) {
    if (config.top_k <= 0) throw std::invalid_argument("ForwardNeighborSearchConfig.top_k must be > 0");
    if (config.candidate_k < config.top_k) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.candidate_k must be >= top_k");
    }
    if (config.query_block_rows <= 0) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.query_block_rows must be > 0");
    }
    if (config.index_block_rows <= 0) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.index_block_rows must be > 0");
    }
    if (config.ann_probe_list_count <= 0) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.ann_probe_list_count must be > 0");
    }
    if (config.strict_future_epsilon < 0.0f) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.strict_future_epsilon must be >= 0");
    }
    if (config.time_window.min_delta < 0.0f) {
        throw std::invalid_argument("ForwardTimeWindow.min_delta must be >= 0");
    }
    if (std::isfinite(config.time_window.max_delta) && config.time_window.max_delta < config.time_window.min_delta) {
        throw std::invalid_argument("ForwardTimeWindow.max_delta must be >= min_delta");
    }
}

inline void validate_forward_neighbor_query_batch_(const ForwardNeighborQueryBatch &query) {
    if (query.latent_dim <= 0) {
        throw std::invalid_argument("ForwardNeighborQueryBatch.latent_dim must be > 0");
    }
    if (query.cell_indices.size() != query.developmental_time.size()) {
        throw std::invalid_argument("ForwardNeighborQueryBatch cell_indices and developmental_time must align");
    }
    if (query.cell_indices.size() * checked_size_(query.latent_dim, "latent_dim") != query.latent_unit.size()) {
        throw std::invalid_argument("ForwardNeighborQueryBatch latent_unit must equal rows * latent_dim");
    }
    if (!query.embryo_ids.empty() && query.embryo_ids.size() != query.cell_indices.size()) {
        throw std::invalid_argument("ForwardNeighborQueryBatch embryo_ids must be empty or align with rows");
    }
}

inline ForwardNeighborSearchResult empty_forward_neighbor_result_(std::int64_t top_k) {
    ForwardNeighborSearchResult result;
    result.query_count = 0;
    result.top_k = top_k;
    return result;
}

} // namespace detail

} // namespace cellerator::compute::neighbors::forward_neighbors
