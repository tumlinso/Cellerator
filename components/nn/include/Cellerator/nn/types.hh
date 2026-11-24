#pragma once

#include <utility>

namespace cellerator::nn {

// Per-invocation behavior controls, kept independent from autograd state.
struct Run {
    bool training = false;
};

// A plain callable identity component. It accepts any value so native model
// compositions are not restricted to tensor-only intermediate results.
struct Identity {
    template<class T>
    constexpr T&& operator()(T&& value, Run = {}) const noexcept {
        return std::forward<T>(value);
    }
};

} // namespace cellerator::nn
