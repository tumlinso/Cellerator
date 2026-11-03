#pragma once

#include <Cellerator/geometry/dependency/reads.hh>
#include <Cellerator/ops/mechanism/omission.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace cellerator::execution {

// A time-dependent condition must be invalidated when time_generation changes,
// even when none of its explicitly listed numerical sources was written.
struct ConditionDefinition {
    geometry::dependency::ConditionReads reads{};
    bool depends_on_time = false;
};

struct SourceRevision {
    geometry::dependency::SourceRef source{};
    std::uint64_t generation = 0;
};

namespace detail {
struct PreparedConditionIndexStorage;
struct ConditionStateStorage;
}

// Immutable, owner-retained reverse index. Preparation deep-copies the
// catalogs, including explicit-coordinate selections and condition reads.
// Postings map a source to its condition/selection pairs; there is no
// materialized candidate-pair graph and no biological exclusion inference.
class PreparedConditionIndex final {
public:
    PreparedConditionIndex(
        structure::structure_epoch epoch,
        std::span<const geometry::dependency::SourceDescriptor> sources,
        std::span<const geometry::dependency::SelectionDescriptor> selections,
        std::span<const ConditionDefinition> conditions);

    [[nodiscard]] structure::structure_epoch structure_epoch() const noexcept;
    [[nodiscard]] std::span<const geometry::dependency::SourceDescriptor> sources() const noexcept;
    [[nodiscard]] std::size_t condition_count() const noexcept;

    // Changed coordinates have their own selection identity. They must match
    // the source's domain, logical order and extent. Different selection-space
    // kinds are conservatively treated as overlapping.
    [[nodiscard]] std::vector<geometry::dependency::condition_id> affected_conditions(
        geometry::dependency::SourceRef source,
        const geometry::dependency::SelectionDescriptor& changed) const;
    [[nodiscard]] std::vector<geometry::dependency::condition_id> affected_conditions(
        geometry::dependency::SourceRef source) const;

private:
    std::shared_ptr<const detail::PreparedConditionIndexStorage> storage_;
    friend class ConditionState;
};

// A ticket pins the exact declared source revisions and, where applicable,
// temporal generation for one condition evaluation. It carries no source
// values and therefore never substitutes for an owner lease or saved contents.
class ConditionEvaluation final {
public:
    [[nodiscard]] geometry::dependency::condition_id condition() const noexcept;
    [[nodiscard]] structure::structure_epoch structure_epoch() const noexcept;
    [[nodiscard]] std::uint64_t binding_revision() const noexcept;
    [[nodiscard]] std::uint64_t temporal_generation() const noexcept;
    [[nodiscard]] std::span<const SourceRevision> sources() const noexcept;

private:
    geometry::dependency::condition_id condition_{};
    structure::structure_epoch epoch_{};
    std::uint64_t binding_revision_ = 0;
    std::uint64_t temporal_generation_ = 0;
    std::uint64_t ticket_sequence_ = 0;
    std::size_t condition_ordinal_ = 0;
    std::vector<SourceRevision> sources_;
    std::shared_ptr<const std::uint8_t> owner_;
    friend class ConditionState;
};

struct CachedCondition {
    ops::mechanism::ConditionOutcome outcome = ops::mechanism::ConditionOutcome::unresolved;
    ConditionEvaluation evaluation;
};

// Host-serialized mutable cache for one execution context. Owners must publish
// source generation changes and synchronize readiness before evaluation;
// mutation outside this protocol cannot be detected. The source owner must
// order publication against writes and keep the actual values stable through
// evaluation; asynchronous uses also require the owner's readiness/lease.
// Saved forward contents remain controlled by their actual owners.
class ConditionState final {
public:
    ConditionState(std::shared_ptr<const PreparedConditionIndex> index,
                   std::uint64_t binding_revision,
                   std::span<const SourceRevision> initial_sources,
                   std::uint64_t temporal_generation = 0);
    ~ConditionState();
    ConditionState(const ConditionState&) = delete;
    ConditionState& operator=(const ConditionState&) = delete;
    ConditionState(ConditionState&&) noexcept;
    ConditionState& operator=(ConditionState&&) noexcept;

    [[nodiscard]] structure::structure_epoch structure_epoch() const noexcept;
    [[nodiscard]] std::uint64_t binding_revision() const noexcept;
    [[nodiscard]] std::uint64_t temporal_generation() const noexcept;
    [[nodiscard]] std::uint64_t source_generation(geometry::dependency::SourceRef source) const;
    [[nodiscard]] bool dirty(geometry::dependency::condition_id condition) const;
    [[nodiscard]] bool active(geometry::dependency::condition_id condition) const;
    void set_active(geometry::dependency::condition_id condition, bool active);
    [[nodiscard]] std::vector<geometry::dependency::condition_id> dirty_conditions(
        bool active_only = true) const;

    // begin snapshots current revisions. publish rejects a foreign, superseded
    // stale, superseded or already-published ticket before changing the cached
    // result; excluded, eligible and unresolved are distinct outcomes.
    [[nodiscard]] ConditionEvaluation begin(geometry::dependency::condition_id condition);
    void publish(const ConditionEvaluation& evaluation,
                 ops::mechanism::ConditionOutcome outcome);
    [[nodiscard]] std::optional<ops::mechanism::ConditionOutcome> cached(
        geometry::dependency::condition_id condition) const;

    // An execution owner may retain this checked snapshot to associate a
    // cached outcome with the current input revisions. Recheck it before use;
    // source ownership/readiness still controls concurrent byte mutation.
    [[nodiscard]] std::optional<CachedCondition> acquire_cached(
        geometry::dependency::condition_id condition) const;
    [[nodiscard]] bool is_current(const ConditionEvaluation& evaluation) const noexcept;

    // A generation change is strictly increasing and invalidates affected
    // conditions even when inactive. Non-overlapping cached results remain
    // reusable. Existing in-flight tickets are conservatively rejected when
    // any declared input source generation has changed.
    [[nodiscard]] std::vector<geometry::dependency::condition_id> source_changed(
        geometry::dependency::SourceRef source, std::uint64_t generation,
        const geometry::dependency::SelectionDescriptor& changed);
    [[nodiscard]] std::vector<geometry::dependency::condition_id> source_changed(
        geometry::dependency::SourceRef source, std::uint64_t generation);
    [[nodiscard]] std::vector<geometry::dependency::condition_id> time_changed(
        std::uint64_t temporal_generation);

private:
    std::unique_ptr<detail::ConditionStateStorage> storage_;
};

} // namespace cellerator::execution
