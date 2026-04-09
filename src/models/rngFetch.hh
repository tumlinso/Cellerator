#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator {

struct RngFetchOptions {
    bool with_replacement = true;
    std::uint64_t seed = std::random_device{}();
};

class RngFetch {
public:
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

    std::vector<unsigned long> next(std::size_t count) {
        std::vector<unsigned long> indices;

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
