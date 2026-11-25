#pragma once

#include <stddef.h>
#include <stdint.h>

namespace cellerator::execution::tuning {

struct FetchSchedule {
    unsigned lookahead = 1;
    bool reorder_rows = false;
    size_t minimum_early_bytes = SIZE_MAX;
};

struct FetchWindow {
    uint64_t revision = 0, elapsed_ns = 0;
    uint64_t batches = 0, cells = 0, entries = 0;
};

struct FetchTuneOptions {
    bool adaptive = false, may_reorder = false;
    unsigned maximum_lookahead = 2;
    size_t split_threshold = SIZE_MAX;
    unsigned minimum_batches = 16;
    double minimum_speedup = 0.05, workload_tolerance = 0.10;
};

enum class FetchTuneDecision : uint8_t { unchanged, trial, accepted, restored, incomparable };

// Compare completed windows of the same operation and source. Each window
// contains only work issued with its schedule revision. The coordinator
// applies schedule changes to future requests and uploads.
struct FetchTuner {
    bool prepare(FetchSchedule schedule, FetchTuneOptions options) noexcept {
        if (!schedule.lookahead || schedule.lookahead > options.maximum_lookahead ||
            options.maximum_lookahead > 2 || !options.minimum_batches ||
            !(options.minimum_speedup >= 0 && options.minimum_speedup < 1) ||
            !(options.workload_tolerance >= 0 && options.workload_tolerance < 1) ||
            (schedule.reorder_rows && !options.may_reorder)) return false;
        if (revision_ == UINT64_MAX) return false;
        current_ = stable_ = schedule;
        options_ = options; next_ = 0; testing_ = false;
        ++revision_;
        return true;
    }

    FetchSchedule schedule() const noexcept { return current_; }
    uint64_t revision() const noexcept { return revision_; }

    FetchTuneDecision observe(FetchWindow w) noexcept {
        if (!options_.adaptive || !revision_ || w.revision != revision_ || revision_ == UINT64_MAX ||
            w.batches < options_.minimum_batches || !w.elapsed_ns || !w.cells) return FetchTuneDecision::unchanged;
        if (testing_) {
            testing_ = false;
            const bool same_work = w.batches == baseline_.batches &&
                comparable(w.cells, baseline_.cells, options_.workload_tolerance) &&
                comparable(w.entries, baseline_.entries, options_.workload_tolerance);
            if (same_work && double(w.elapsed_ns) < double(baseline_.elapsed_ns)*(1-options_.minimum_speedup)) {
                stable_ = current_;
                return FetchTuneDecision::accepted;
            }
            current_ = stable_;
            ++revision_;
            return same_work ? FetchTuneDecision::restored : FetchTuneDecision::incomparable;
        }

        while (next_ < 3) {
            const unsigned knob = next_++;
            if (knob == 0 && current_.lookahead < options_.maximum_lookahead)
                current_.lookahead = options_.maximum_lookahead;
            else if (knob == 1 && options_.may_reorder && !current_.reorder_rows)
                current_.reorder_rows = true;
            else if (knob == 2 && options_.split_threshold != current_.minimum_early_bytes)
                current_.minimum_early_bytes = options_.split_threshold;
            else continue;
            baseline_ = w;
            testing_ = true;
            ++revision_;
            return FetchTuneDecision::trial;
        }
        return FetchTuneDecision::unchanged;
    }

private:
    static bool comparable(uint64_t a, uint64_t b, double tolerance) noexcept {
        const uint64_t d = a > b ? a-b : b-a;
        return double(d) <= double(b)*tolerance;
    }

    FetchSchedule current_{}, stable_{};
    FetchTuneOptions options_{};
    FetchWindow baseline_{};
    uint64_t revision_ = 0;
    unsigned next_ = 0;
    bool testing_ = false;
};

} // namespace cellerator::execution::tuning
