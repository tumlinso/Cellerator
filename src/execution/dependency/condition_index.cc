#include <Cellerator/execution/dependency/condition_index.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace cellerator::execution {
namespace semantic = geometry::dependency;
namespace {

template<class Identity>
bool identity_less(const Identity& left, const Identity& right) noexcept {
    return left.low < right.low || (left.low == right.low && left.high < right.high);
}

template<class Identity, class Accessor>
std::size_t find_ordinal(const std::vector<std::size_t>& sorted,
                         const Identity& identity, Accessor accessor) noexcept {
    const auto found = std::lower_bound(sorted.begin(), sorted.end(), identity,
        [&](std::size_t ordinal, const Identity& key) {
            return identity_less(accessor(ordinal), key);
        });
    if (found == sorted.end() || !structure::same_identity(accessor(*found), identity))
        return sorted.size();
    return *found;
}

void require_valid(semantic::Status status, const char* operation) {
    if (status.code == semantic::StatusCode::allocation_failure)
        throw std::bad_alloc();
    if (!status)
        throw std::invalid_argument(std::string(operation) + " (status "
            + std::to_string(static_cast<unsigned>(status.code)) + ", index "
            + std::to_string(status.index) + ")");
}

std::uint64_t selection_size(const semantic::SelectionDescriptor& selection) noexcept {
    switch (selection.kind) {
    case semantic::SelectionKind::all: return selection.extent;
    case semantic::SelectionKind::range: return selection.count;
    case semantic::SelectionKind::explicit_coordinates: return selection.coordinates.size();
    }
    return 0;
}

// Both selections have already been validated against the same source axis.
// Distinct selection-space meanings cannot establish disjointness.
bool overlaps(const semantic::SelectionDescriptor& left,
              const semantic::SelectionDescriptor& right) noexcept {
    if (selection_size(left) == 0 || selection_size(right) == 0) return false;
    if (left.reference.space != right.reference.space) return true;
    if (left.kind == semantic::SelectionKind::all
        || right.kind == semantic::SelectionKind::all) return true;
    if (left.kind == semantic::SelectionKind::range
        && right.kind == semantic::SelectionKind::range)
        return left.begin < right.begin + right.count
            && right.begin < left.begin + left.count;
    const auto range_intersects_set = [](const auto& range, const auto& set) {
        const auto first = std::lower_bound(set.coordinates.begin(),
                                            set.coordinates.end(), range.begin);
        return first != set.coordinates.end() && *first - range.begin < range.count;
    };
    if (left.kind == semantic::SelectionKind::range)
        return range_intersects_set(left, right);
    if (right.kind == semantic::SelectionKind::range)
        return range_intersects_set(right, left);
    auto a = left.coordinates.begin();
    auto b = right.coordinates.begin();
    while (a != left.coordinates.end() && b != right.coordinates.end()) {
        if (*a < *b) ++a;
        else if (*b < *a) ++b;
        else return true;
    }
    return false;
}

} // namespace

namespace detail {

struct ConditionPlan {
    semantic::condition_id identity{};
    bool depends_on_time = false;
    std::size_t source_begin = 0;
    std::size_t source_count = 0;
};

struct ConditionPosting {
    std::size_t condition_ordinal = 0;
    std::size_t selection_ordinal = 0;
};

struct PreparedConditionIndexStorage {
    structure::structure_epoch epoch{};
    std::vector<semantic::SourceDescriptor> sources;
    std::vector<semantic::SelectionDescriptor> selections;
    std::vector<std::uint64_t> selection_coordinates;
    std::vector<ConditionPlan> conditions;
    std::vector<std::size_t> condition_sources;
    std::vector<std::size_t> source_order;
    std::vector<std::size_t> selection_order;
    std::vector<std::size_t> condition_order;
    std::vector<std::size_t> temporal_conditions;
    std::vector<std::size_t> posting_offsets;
    std::vector<ConditionPosting> postings;
};

struct LiveCondition {
    bool dirty = true;
    bool active = true;
    ops::mechanism::ConditionOutcome outcome = ops::mechanism::ConditionOutcome::unresolved;
    std::uint64_t ticket_sequence = 0;
    std::uint64_t published_sequence = 0;
};

struct ConditionStateStorage {
    std::shared_ptr<const PreparedConditionIndexStorage> index;
    std::shared_ptr<const std::uint8_t> identity = std::make_shared<const std::uint8_t>(0);
    std::uint64_t binding_revision = 0;
    std::uint64_t temporal_generation = 0;
    std::vector<std::uint64_t> source_generations;
    std::vector<LiveCondition> conditions;
};

} // namespace detail

namespace {
using IndexStorage = detail::PreparedConditionIndexStorage;
using LiveStorage = detail::ConditionStateStorage;

std::size_t source_ordinal(const IndexStorage& index, semantic::SourceRef source) {
    if (!structure::valid_identity(source.identity)
        || !structure::valid_identity(source.domain)
        || !structure::valid_identity(source.representation))
        throw std::invalid_argument("invalid condition source identity");
    const auto ordinal = find_ordinal(index.source_order, source.identity,
        [&](std::size_t i) { return index.sources[i].reference.identity; });
    if (ordinal == index.sources.size())
        throw std::invalid_argument("unknown condition source");
    if (!semantic::same_source(source, index.sources[ordinal].reference))
        throw std::invalid_argument("condition source domain or representation mismatch");
    return ordinal;
}

std::size_t condition_ordinal(const IndexStorage& index,
                              semantic::condition_id condition) {
    if (!structure::valid_identity(condition))
        throw std::invalid_argument("invalid condition identity");
    const auto ordinal = find_ordinal(index.condition_order, condition,
        [&](std::size_t i) { return index.conditions[i].identity; });
    if (ordinal == index.conditions.size())
        throw std::invalid_argument("unknown condition");
    return ordinal;
}

std::size_t selection_ordinal(const IndexStorage& index,
                              semantic::selection_id selection) {
    const auto ordinal = find_ordinal(index.selection_order, selection,
        [&](std::size_t i) { return index.selections[i].reference.identity; });
    if (ordinal == index.selections.size())
        throw std::invalid_argument("unknown condition selection");
    return ordinal;
}

void validate_changed_selection(const semantic::SourceDescriptor& source,
                                const semantic::SelectionDescriptor& changed) {
    const semantic::ReadDependency read{source.reference, changed.reference};
    require_valid(semantic::validate_reads({&source, 1}, {&changed, 1}, {&read, 1}),
                  "invalid changed condition source selection");
}

std::vector<std::size_t> affected_ordinals(
    const IndexStorage& index, std::size_t source,
    const semantic::SelectionDescriptor* changed) {
    if (changed) validate_changed_selection(index.sources[source], *changed);
    std::vector<std::size_t> result;
    for (auto p = index.posting_offsets[source]; p < index.posting_offsets[source + 1]; ++p) {
        const auto posting = index.postings[p];
        if ((!changed || overlaps(index.selections[posting.selection_ordinal], *changed))
            && (result.empty() || result.back() != posting.condition_ordinal))
            result.push_back(posting.condition_ordinal);
    }
    return result;
}

std::vector<semantic::condition_id> condition_ids(
    const IndexStorage& index, std::span<const std::size_t> ordinals) {
    std::vector<semantic::condition_id> result;
    result.reserve(ordinals.size());
    for (const auto ordinal : ordinals) result.push_back(index.conditions[ordinal].identity);
    return result;
}

void ensure_live(const std::unique_ptr<LiveStorage>& storage) {
    if (!storage) throw std::logic_error("moved-from condition state");
}

std::span<const std::size_t> condition_sources(const IndexStorage& index,
                                               std::size_t ordinal) noexcept {
    const auto& plan = index.conditions[ordinal];
    if (plan.source_count == 0) return {};
    return {index.condition_sources.data() + plan.source_begin, plan.source_count};
}

} // namespace

PreparedConditionIndex::PreparedConditionIndex(
    structure::structure_epoch epoch,
    std::span<const semantic::SourceDescriptor> sources,
    std::span<const semantic::SelectionDescriptor> selections,
    std::span<const ConditionDefinition> conditions) {
    if (!structure::valid_epoch(epoch))
        throw std::invalid_argument("invalid condition structure epoch");

    std::vector<semantic::ConditionReads> reads;
    reads.reserve(conditions.size());
    for (const auto& condition : conditions) reads.push_back(condition.reads);
    require_valid(semantic::validate_conditions(sources, selections, reads),
                  "invalid condition index declarations");

    auto storage = std::make_shared<IndexStorage>();
    storage->epoch = epoch;
    storage->sources.assign(sources.begin(), sources.end());
    storage->selections.resize(selections.size());
    std::size_t coordinate_count = 0;
    for (const auto& selection : selections) {
        if (selection.coordinates.size() > std::numeric_limits<std::size_t>::max() - coordinate_count)
            throw std::overflow_error("condition selection coordinate count overflows");
        coordinate_count += selection.coordinates.size();
    }
    storage->selection_coordinates.reserve(coordinate_count);
    for (std::size_t i = 0; i < selections.size(); ++i) {
        storage->selections[i] = selections[i];
        const auto begin = storage->selection_coordinates.size();
        if (selections[i].coordinates.empty()) {
            storage->selections[i].coordinates = {};
        } else {
            storage->selection_coordinates.insert(storage->selection_coordinates.end(),
                selections[i].coordinates.begin(), selections[i].coordinates.end());
            storage->selections[i].coordinates = {
                storage->selection_coordinates.data() + begin, selections[i].coordinates.size()};
        }
    }

    storage->source_order.resize(sources.size());
    std::iota(storage->source_order.begin(), storage->source_order.end(), 0);
    std::sort(storage->source_order.begin(), storage->source_order.end(), [&](auto a, auto b) {
        return identity_less(storage->sources[a].reference.identity, storage->sources[b].reference.identity);
    });
    storage->selection_order.resize(selections.size());
    std::iota(storage->selection_order.begin(), storage->selection_order.end(), 0);
    std::sort(storage->selection_order.begin(), storage->selection_order.end(), [&](auto a, auto b) {
        return identity_less(storage->selections[a].reference.identity,
                             storage->selections[b].reference.identity);
    });
    storage->conditions.resize(conditions.size());
    storage->posting_offsets.assign(sources.size() + 1, 0);
    std::vector<std::size_t> condition_source_scratch;
    for (std::size_t c = 0; c < conditions.size(); ++c) {
        auto& target = storage->conditions[c];
        target.identity = conditions[c].reads.identity;
        target.depends_on_time = conditions[c].depends_on_time;
        if (target.depends_on_time) storage->temporal_conditions.push_back(c);
        condition_source_scratch.clear();
        for (const auto& input : conditions[c].reads.inputs) {
            const auto source = source_ordinal(*storage, input.source);
            ++storage->posting_offsets[source + 1];
            condition_source_scratch.push_back(source);
        }
        std::sort(condition_source_scratch.begin(), condition_source_scratch.end());
        condition_source_scratch.erase(
            std::unique(condition_source_scratch.begin(), condition_source_scratch.end()),
            condition_source_scratch.end());
        target.source_begin = storage->condition_sources.size();
        target.source_count = condition_source_scratch.size();
        storage->condition_sources.insert(storage->condition_sources.end(),
            condition_source_scratch.begin(), condition_source_scratch.end());
    }
    std::partial_sum(storage->posting_offsets.begin(), storage->posting_offsets.end(),
                     storage->posting_offsets.begin());
    storage->postings.resize(storage->posting_offsets.back());
    auto next_posting = storage->posting_offsets;
    for (std::size_t c = 0; c < conditions.size(); ++c) {
        for (const auto& input : conditions[c].reads.inputs) {
            const auto source = source_ordinal(*storage, input.source);
            const auto selection = selection_ordinal(*storage, input.selection.identity);
            storage->postings[next_posting[source]++] = {c, selection};
        }
    }
    storage->condition_order.resize(conditions.size());
    std::iota(storage->condition_order.begin(), storage->condition_order.end(), 0);
    std::sort(storage->condition_order.begin(), storage->condition_order.end(), [&](auto a, auto b) {
        return identity_less(storage->conditions[a].identity, storage->conditions[b].identity);
    });
    storage_ = std::move(storage);
}

structure::structure_epoch PreparedConditionIndex::structure_epoch() const noexcept {
    return storage_ ? storage_->epoch : structure::structure_epoch{};
}

std::span<const semantic::SourceDescriptor> PreparedConditionIndex::sources() const noexcept {
    return storage_ ? std::span<const semantic::SourceDescriptor>(storage_->sources)
                    : std::span<const semantic::SourceDescriptor>{};
}

std::size_t PreparedConditionIndex::condition_count() const noexcept {
    return storage_ ? storage_->conditions.size() : 0;
}

std::vector<semantic::condition_id> PreparedConditionIndex::affected_conditions(
    semantic::SourceRef source, const semantic::SelectionDescriptor& changed) const {
    if (!storage_) throw std::logic_error("moved-from condition index");
    const auto ordinal = source_ordinal(*storage_, source);
    const auto affected = affected_ordinals(*storage_, ordinal, &changed);
    return condition_ids(*storage_, affected);
}

std::vector<semantic::condition_id> PreparedConditionIndex::affected_conditions(
    semantic::SourceRef source) const {
    if (!storage_) throw std::logic_error("moved-from condition index");
    const auto ordinal = source_ordinal(*storage_, source);
    const auto affected = affected_ordinals(*storage_, ordinal, nullptr);
    return condition_ids(*storage_, affected);
}

semantic::condition_id ConditionEvaluation::condition() const noexcept { return condition_; }
structure::structure_epoch ConditionEvaluation::structure_epoch() const noexcept { return epoch_; }
std::uint64_t ConditionEvaluation::binding_revision() const noexcept { return binding_revision_; }
std::uint64_t ConditionEvaluation::temporal_generation() const noexcept { return temporal_generation_; }
std::span<const SourceRevision> ConditionEvaluation::sources() const noexcept { return sources_; }

ConditionState::ConditionState(std::shared_ptr<const PreparedConditionIndex> index,
                               std::uint64_t binding_revision,
                               std::span<const SourceRevision> initial_sources,
                               std::uint64_t temporal_generation) {
    if (!index || binding_revision == 0)
        throw std::invalid_argument("condition state requires an index and binding revision");
    if (!index->storage_)
        throw std::invalid_argument("condition state index was moved from");
    auto retained_plan = index->storage_;
    const auto& plan = *retained_plan;
    if (initial_sources.size() != plan.sources.size())
        throw std::invalid_argument("condition source generations do not cover the source catalog");
    auto state = std::make_unique<LiveStorage>();
    state->index = std::move(retained_plan);
    state->binding_revision = binding_revision;
    state->temporal_generation = temporal_generation;
    state->source_generations.resize(plan.sources.size());
    state->conditions.resize(plan.conditions.size());
    std::vector<std::uint8_t> seen(plan.sources.size(), 0);
    for (const auto& initial : initial_sources) {
        const auto ordinal = source_ordinal(plan, initial.source);
        if (seen[ordinal]) throw std::invalid_argument("duplicate condition source generation");
        seen[ordinal] = 1;
        state->source_generations[ordinal] = initial.generation;
    }
    storage_ = std::move(state);
}

ConditionState::~ConditionState() = default;
ConditionState::ConditionState(ConditionState&&) noexcept = default;
ConditionState& ConditionState::operator=(ConditionState&&) noexcept = default;

structure::structure_epoch ConditionState::structure_epoch() const noexcept {
    return storage_ ? storage_->index->epoch : structure::structure_epoch{};
}
std::uint64_t ConditionState::binding_revision() const noexcept {
    return storage_ ? storage_->binding_revision : 0;
}
std::uint64_t ConditionState::temporal_generation() const noexcept {
    return storage_ ? storage_->temporal_generation : 0;
}
std::uint64_t ConditionState::source_generation(semantic::SourceRef source) const {
    ensure_live(storage_);
    return storage_->source_generations[source_ordinal(*storage_->index, source)];
}
bool ConditionState::dirty(semantic::condition_id condition) const {
    ensure_live(storage_);
    return storage_->conditions[condition_ordinal(*storage_->index, condition)].dirty;
}
bool ConditionState::active(semantic::condition_id condition) const {
    ensure_live(storage_);
    return storage_->conditions[condition_ordinal(*storage_->index, condition)].active;
}
void ConditionState::set_active(semantic::condition_id condition, bool active_value) {
    ensure_live(storage_);
    storage_->conditions[condition_ordinal(*storage_->index, condition)].active = active_value;
}
std::vector<semantic::condition_id> ConditionState::dirty_conditions(bool active_only) const {
    ensure_live(storage_);
    const auto& index = *storage_->index;
    std::vector<semantic::condition_id> result;
    for (std::size_t i = 0; i < storage_->conditions.size(); ++i)
        if (storage_->conditions[i].dirty && (!active_only || storage_->conditions[i].active))
            result.push_back(index.conditions[i].identity);
    return result;
}

ConditionEvaluation ConditionState::begin(semantic::condition_id condition) {
    ensure_live(storage_);
    const auto& index = *storage_->index;
    const auto ordinal = condition_ordinal(index, condition);
    auto& live = storage_->conditions[ordinal];
    if (live.ticket_sequence == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("condition evaluation ticket overflow");
    ConditionEvaluation result;
    result.condition_ = condition;
    result.epoch_ = index.epoch;
    result.binding_revision_ = storage_->binding_revision;
    result.temporal_generation_ = storage_->temporal_generation;
    result.condition_ordinal_ = ordinal;
    result.owner_ = storage_->identity;
    const auto inputs = condition_sources(index, ordinal);
    result.sources_.reserve(inputs.size());
    for (const auto source : inputs)
        result.sources_.push_back({index.sources[source].reference, storage_->source_generations[source]});
    result.ticket_sequence_ = ++live.ticket_sequence;
    return result;
}

void ConditionState::publish(const ConditionEvaluation& evaluation,
                             ops::mechanism::ConditionOutcome outcome) {
    ensure_live(storage_);
    if (outcome != ops::mechanism::ConditionOutcome::excluded
        && outcome != ops::mechanism::ConditionOutcome::eligible
        && outcome != ops::mechanism::ConditionOutcome::unresolved)
        throw std::invalid_argument("invalid condition outcome");
    const auto& index = *storage_->index;
    if (evaluation.owner_ != storage_->identity
        || !structure::same_epoch(evaluation.epoch_, index.epoch)
        || evaluation.binding_revision_ != storage_->binding_revision
        || evaluation.condition_ordinal_ >= index.conditions.size()
        || !structure::same_identity(evaluation.condition_,
                                     index.conditions[evaluation.condition_ordinal_].identity))
        throw std::logic_error("foreign condition evaluation ticket");
    const auto ordinal = evaluation.condition_ordinal_;
    auto& live = storage_->conditions[ordinal];
    if (evaluation.ticket_sequence_ != live.ticket_sequence
        || evaluation.ticket_sequence_ == live.published_sequence
        || (index.conditions[ordinal].depends_on_time
            && evaluation.temporal_generation_ != storage_->temporal_generation)
        || evaluation.sources_.size() != index.conditions[ordinal].source_count)
        throw std::logic_error("stale or superseded condition evaluation ticket");
    const auto inputs = condition_sources(index, ordinal);
    for (std::size_t i = 0; i < evaluation.sources_.size(); ++i) {
        const auto source = inputs[i];
        if (!semantic::same_source(evaluation.sources_[i].source, index.sources[source].reference)
            || evaluation.sources_[i].generation != storage_->source_generations[source])
            throw std::logic_error("stale condition source generation");
    }
    live.outcome = outcome;
    live.dirty = false;
    live.published_sequence = evaluation.ticket_sequence_;
}

std::optional<ops::mechanism::ConditionOutcome> ConditionState::cached(
    semantic::condition_id condition) const {
    ensure_live(storage_);
    const auto& live = storage_->conditions[condition_ordinal(*storage_->index, condition)];
    if (live.dirty) return std::nullopt;
    return live.outcome;
}

std::optional<CachedCondition> ConditionState::acquire_cached(
    semantic::condition_id condition) const {
    ensure_live(storage_);
    const auto& index = *storage_->index;
    const auto ordinal = condition_ordinal(index, condition);
    const auto& live = storage_->conditions[ordinal];
    if (live.dirty) return std::nullopt;
    CachedCondition result;
    result.outcome = live.outcome;
    auto& evaluation = result.evaluation;
    evaluation.condition_ = condition;
    evaluation.epoch_ = index.epoch;
    evaluation.binding_revision_ = storage_->binding_revision;
    evaluation.temporal_generation_ = storage_->temporal_generation;
    evaluation.condition_ordinal_ = ordinal;
    evaluation.owner_ = storage_->identity;
    evaluation.ticket_sequence_ = live.published_sequence;
    const auto inputs = condition_sources(index, ordinal);
    evaluation.sources_.reserve(inputs.size());
    for (const auto source : inputs)
        evaluation.sources_.push_back({index.sources[source].reference,
                                       storage_->source_generations[source]});
    return result;
}

bool ConditionState::is_current(const ConditionEvaluation& evaluation) const noexcept {
    if (!storage_ || evaluation.owner_ != storage_->identity
        || evaluation.binding_revision_ != storage_->binding_revision)
        return false;
    const auto& index = *storage_->index;
    if (!structure::same_epoch(evaluation.epoch_, index.epoch)
        || evaluation.condition_ordinal_ >= index.conditions.size()) return false;
    const auto ordinal = evaluation.condition_ordinal_;
    const auto& plan = index.conditions[ordinal];
    const auto& live = storage_->conditions[ordinal];
    if (live.dirty || evaluation.ticket_sequence_ != live.published_sequence
        || !structure::same_identity(evaluation.condition_, plan.identity)
        || (plan.depends_on_time
            && evaluation.temporal_generation_ != storage_->temporal_generation)
        || evaluation.sources_.size() != plan.source_count) return false;
    const auto inputs = condition_sources(index, ordinal);
    for (std::size_t i = 0; i < evaluation.sources_.size(); ++i) {
        const auto source = inputs[i];
        if (!semantic::same_source(evaluation.sources_[i].source, index.sources[source].reference)
            || evaluation.sources_[i].generation != storage_->source_generations[source])
            return false;
    }
    return true;
}

std::vector<semantic::condition_id> ConditionState::source_changed(
    semantic::SourceRef source, std::uint64_t generation,
    const semantic::SelectionDescriptor& changed) {
    ensure_live(storage_);
    const auto& index = *storage_->index;
    const auto ordinal = source_ordinal(index, source);
    if (generation <= storage_->source_generations[ordinal])
        throw std::logic_error("condition source generation must increase");
    const auto affected = affected_ordinals(index, ordinal, &changed);
    auto result = condition_ids(index, affected);
    storage_->source_generations[ordinal] = generation;
    for (const auto condition : affected) storage_->conditions[condition].dirty = true;
    return result;
}

std::vector<semantic::condition_id> ConditionState::source_changed(
    semantic::SourceRef source, std::uint64_t generation) {
    ensure_live(storage_);
    const auto& index = *storage_->index;
    const auto ordinal = source_ordinal(index, source);
    if (generation <= storage_->source_generations[ordinal])
        throw std::logic_error("condition source generation must increase");
    const auto affected = affected_ordinals(index, ordinal, nullptr);
    auto result = condition_ids(index, affected);
    storage_->source_generations[ordinal] = generation;
    for (const auto condition : affected) storage_->conditions[condition].dirty = true;
    return result;
}

std::vector<semantic::condition_id> ConditionState::time_changed(
    std::uint64_t temporal_generation) {
    ensure_live(storage_);
    if (temporal_generation <= storage_->temporal_generation)
        throw std::logic_error("condition temporal generation must increase");
    const auto& index = *storage_->index;
    auto result = condition_ids(index, index.temporal_conditions);
    storage_->temporal_generation = temporal_generation;
    for (const auto condition : index.temporal_conditions)
        storage_->conditions[condition].dirty = true;
    return result;
}

} // namespace cellerator::execution
