#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace cellerator {

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

    void push_back(const T &value) {
        if (size_ == capacity_) reserve(capacity_ != 0u ? capacity_ * 2u : 1u);
        data_[size_++] = value;
    }

    void push_back(T &&value) {
        if (size_ == capacity_) reserve(capacity_ != 0u ? capacity_ * 2u : 1u);
        data_[size_++] = std::move(value);
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

struct RngFetchOptions {
    bool with_replacement = true;
    std::uint64_t seed = std::random_device{}();
};

class RngFetch {
public:
    RngFetch() = default;

    explicit RngFetch(unsigned long population_size, RngFetchOptions options = RngFetchOptions())
        : population_size_(population_size),
          options_(options),
          rng_(options.seed) {
        if (population_size_ == 0) {
            throw std::invalid_argument("RngFetch requires a non-zero population size");
        }
    }

    std::size_t population_size() const {
        return static_cast<std::size_t>(population_size_);
    }

    // CPU sampler helper. With replacement is O(count); without replacement
    // adds hash-table traffic and should stay out of tight inner loops.
    host_buffer<unsigned long> next(std::size_t count) {
        host_buffer<unsigned long> indices;

        if (count == 0) throw std::invalid_argument("RngFetch::next requires count > 0");
        if (!options_.with_replacement && count > static_cast<std::size_t>(population_size_)) {
            throw std::invalid_argument("RngFetch::next count exceeds population when sampling without replacement");
        }

        indices.reserve(count);
        std::uniform_int_distribution<unsigned long> dist(0, population_size_ - 1);
        if (options_.with_replacement) {
            for (std::size_t i = 0; i < count; ++i) indices.push_back(dist(rng_));
            return indices;
        }

        // Without replacement pays extra host work for set membership checks.
        std::unordered_set<unsigned long> seen;
        seen.reserve(count * 2u);
        while (indices.size() < count) {
            const unsigned long idx = dist(rng_);
            if (seen.insert(idx).second) indices.push_back(idx);
        }
        return indices;
    }

private:
    unsigned long population_size_;
    RngFetchOptions options_;
    std::mt19937_64 rng_;
};

} // namespace cellerator
