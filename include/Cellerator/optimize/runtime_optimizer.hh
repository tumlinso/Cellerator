#pragma once

#include <cstdint>
#include <cstring>

namespace cellerator::optimize {

enum class optimizer_mode : std::uint32_t {
    disabled = 0u,
    light = 1u
};

enum class optimizer_provider : std::uint32_t {
    none = 0u,
    preprocess = 1u
};

enum class optimizer_stop_reason : std::uint32_t {
    not_run = 0u,
    disabled = 1u,
    budget_skipped = 2u,
    completed = 3u,
    close_enough = 4u,
    failed = 5u
};

struct optimizer_options {
    optimizer_mode mode = optimizer_mode::disabled;
    std::uint32_t max_trials = 2u;
    std::uint32_t warmup_iters = 0u;
    std::uint32_t timed_iters = 1u;
    std::uint32_t sample_rows = 512u;
    std::uint64_t max_sample_nnz = 1ull << 20;
    float close_enough_fraction = 0.10f;
    float max_tune_fraction = 0.25f;
};

struct optimizer_result {
    optimizer_provider provider = optimizer_provider::none;
    std::uint32_t selected_plan = 0u;
    float baseline_ms = 0.0f;
    float best_ms = 0.0f;
    std::uint32_t trials_run = 0u;
    optimizer_stop_reason stop_reason = optimizer_stop_reason::not_run;
    char message[192] = {};
};

struct optimizer_candidate {
    std::uint32_t plan = 0u;
    const char *name = nullptr;
};

inline optimizer_options light_options() {
    optimizer_options out{};
    out.mode = optimizer_mode::light;
    return out;
}

inline void set_message(optimizer_result *result, const char *message) {
    if (result == nullptr) return;
    result->message[0] = '\0';
    if (message == nullptr) return;
    std::strncpy(result->message, message, sizeof(result->message) - 1u);
    result->message[sizeof(result->message) - 1u] = '\0';
}

inline void mark_disabled(optimizer_result *result, const char *message = "optimizer disabled") {
    if (result == nullptr) return;
    *result = optimizer_result{};
    result->stop_reason = optimizer_stop_reason::disabled;
    set_message(result, message);
}

template<typename MeasureFn>
inline std::uint32_t choose_light_plan(const optimizer_options &options,
                                       optimizer_provider provider,
                                       const optimizer_candidate *candidates,
                                       std::uint32_t candidate_count,
                                       MeasureFn measure_ms,
                                       optimizer_result *result) {
    if (result != nullptr) *result = optimizer_result{};
    if (candidates == nullptr || candidate_count == 0u || options.mode == optimizer_mode::disabled) {
        mark_disabled(result);
        return candidates != nullptr && candidate_count != 0u ? candidates[0].plan : 0u;
    }

    const std::uint32_t trials = options.max_trials != 0u && options.max_trials < candidate_count
        ? options.max_trials
        : candidate_count;
    float baseline = measure_ms(candidates[0].plan);
    float best = baseline;
    std::uint32_t selected = 0u;
    for (std::uint32_t i = 1u; i < trials; ++i) {
        const float elapsed = measure_ms(candidates[i].plan);
        if (elapsed < best) {
            best = elapsed;
            selected = i;
        }
    }

    optimizer_stop_reason reason = optimizer_stop_reason::completed;
    if (selected != 0u && best >= baseline * (1.0f - options.close_enough_fraction)) {
        selected = 0u;
        best = baseline;
        reason = optimizer_stop_reason::close_enough;
    } else if (selected == 0u) {
        reason = optimizer_stop_reason::close_enough;
    }

    if (result != nullptr) {
        result->provider = provider;
        result->selected_plan = candidates[selected].plan;
        result->baseline_ms = baseline;
        result->best_ms = best;
        result->trials_run = trials;
        result->stop_reason = reason;
        set_message(result, candidates[selected].name != nullptr ? candidates[selected].name : "selected light optimizer plan");
    }
    return candidates[selected].plan;
}

} // namespace cellerator::optimize
