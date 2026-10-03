#pragma once
#include <Cellerator/state/structured_state.hh>
#include <functional>
#include <vector>

namespace cellerator::math::adaptive {
// Complete caller-declared context, including program/query/dependency semantics
// and every whole-world hypothesis. Equality is direct, never hash-only.
struct context {
    state::generations generations{};
    std::vector<state::coordinate> coordinates;
    std::vector<std::uint64_t> program_query_dependencies;
    std::vector<double> world, parameters;
};
bool same_context(const context&, const context&) noexcept;
// Supplied math owner evaluates at sent state, or applies a delta from old_sent.
// No polynomial implementation, solver, detached trainable cache or world mixing.
using evaluator = std::function<std::vector<double>(std::span<const double>)>;
using delta_provider = std::function<std::vector<double>(std::span<const double> old_sent,
    std::span<const double> delta, std::span<const double> old_output)>;
struct delta_result {
    std::vector<double> output, sent;
    double discrepancy = 0; // max absolute current-versus-sent output, measured
    bool reset = false;
    std::size_t transmitted = 0;
};
class delta_ledger {
    context context_;
    std::vector<double> sent_, output_;
    bool ready_ = false;
public:
    // Providers must be deterministic, side-effect-free for the declared context.
    // Changing provider definition requires a different program context identity.
    // Failure leaves the ledger untouched; discrepancy includes provider roundoff.
    delta_result update(const context&, std::span<const double>, double threshold,
                        const evaluator&, const delta_provider& = {});
};
struct linear_snapshot {
    state::generations generations{};
    std::vector<state::coordinate> coordinates;
    std::vector<double> values, law, readout; // row-major law n*n, readout outputs*n
    std::size_t outputs = 0;
    std::vector<double> first_moment, second_moment;
    std::vector<std::uint64_t> optimizer_steps;
};
enum class rewrite_kind { exact_linear, approximate_linear };
struct supplied_rewrite {
    linear_snapshot candidate;
    std::vector<double> forward, backward; // new*n_old, old*n_new
    // -1 resets; nonnegative old index preserves moments only for unchanged
    // coordinate, state value and law/readout column. Basis changes reset.
    std::vector<std::int64_t> optimizer_from;
    rewrite_kind kind = rewrite_kind::exact_linear;
    double tolerance = 0;
};
struct rewrite_report {
    rewrite_kind kind;
    double inverse_residual, dynamics_residual, readout_residual;
    double current_readout_discrepancy;
};
// Small host publication owner. Caller serializes access, including lease release.
// Leases pin native snapshot metadata; they are not an automatic differentiation
// engine. External tapes hold a lease until backward/release has finished.
class publication {
    linear_snapshot snapshot_;
    std::size_t leases_ = 0;
public:
    explicit publication(linear_snapshot);
    publication(const publication&) = delete;
    publication& operator=(const publication&) = delete;
    class tape_lease {
        publication* owner_;
        friend class publication;
        explicit tape_lease(publication* owner) : owner_(owner) {}
    public:
        tape_lease(const tape_lease&) = delete;
        tape_lease& operator=(const tape_lease&) = delete;
        tape_lease(tape_lease&& other) noexcept : owner_(other.owner_) { other.owner_=nullptr; }
        ~tape_lease() { release(); }
        void release() noexcept { if(owner_) { --owner_->leases_; owner_=nullptr; } }
    };
    // Owner must outlive every lease. No structure/value/parameter mutation while pinned.
    tape_lease acquire_tape() { ++leases_; return tape_lease(this); }
    const linear_snapshot& snapshot() const noexcept { return snapshot_; }
    rewrite_report publish(const supplied_rewrite&);
};
} // namespace cellerator::math::adaptive
