#pragma once

#include <Cellerator/execution/dependency/read_index.hh>
#include <Cellerator/execution/dependency/workset.hh>

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cellerator::execution {

struct EvaluationEpochs {
    std::uint64_t parameters = 0;
    std::uint64_t context = 0;
    bool operator==(const EvaluationEpochs& other) const noexcept {
        return parameters == other.parameters && context == other.context;
    }
    bool operator!=(const EvaluationEpochs& other) const noexcept { return !(*this == other); }
};

template<class State>
struct SourceRead {
    containers::PostingTerm source;
    std::uint64_t generation;
    State value;
};

// Owns immutable reads in original participant order, including repeated
// incidences. Old snapshots survive later source updates and cache replacement.
template<class State>
class SourceReadSnapshot final {
public:
    explicit SourceReadSnapshot(std::vector<std::shared_ptr<const SourceRead<State>>> reads)
        : reads_(std::move(reads)) {}
    std::size_t size() const noexcept { return reads_.size(); }
    const SourceRead<State>& at(std::size_t incidence) const { return *reads_.at(incidence); }
private:
    std::vector<std::shared_ptr<const SourceRead<State>>> reads_;
};

template<class State, class Output>
struct InteractionEvaluation {
    std::uint64_t interaction_id;
    EvaluationEpochs epochs;
    SourceReadSnapshot<State> reads;
    Output output;
};

struct IncrementalStepResult {
    std::vector<std::uint64_t> evaluated_ids;
    std::size_t pending_invalid = 0;
    std::uint64_t step_generation = 0;
};

// Single-owner host engine. Exact cache reuse is conditional on complete read
// dependencies and a pure evaluator whose external inputs are versioned by
// EvaluationEpochs. No hidden tolerance, event suppression, or numerical cast.
// State and Output may be numeric or model-specific immutable value types.
template<class State, class Output>
class IncrementalInteractions final {
public:
    using Read = SourceRead<State>;
    using Record = InteractionEvaluation<State, Output>;
    using Evaluator = std::function<Output(std::uint64_t, const SourceReadSnapshot<State>&,
                                           EvaluationEpochs)>;
    struct CacheEntry {
        std::shared_ptr<const Record> record;
        bool valid;
    };

    IncrementalInteractions(
        std::shared_ptr<const geometry::dependency::ExplicitPattern> dependencies,
        std::vector<Read> initial_sources, Evaluator evaluator, EvaluationEpochs epochs = {})
        : pattern_(require_pattern(std::move(dependencies))), incidence_(pattern_),
          evaluator_(std::move(evaluator)), epochs_(epochs) {
        if (!evaluator_) throw std::invalid_argument("incremental evaluator cannot be empty");
        if (initial_sources.size() != incidence_.sources().size())
            throw std::invalid_argument("initial sources must cover each declared dependency exactly once");
        sources_.resize(initial_sources.size());
        for (auto& source : initial_sources) {
            const auto slot = incidence_.source_ordinal(source.source);
            if (sources_[slot]) throw std::invalid_argument("duplicate initial source");
            sources_[slot] = std::make_shared<const Read>(std::move(source));
        }
        for (const auto& participant : pattern_->participants())
            participant_slots_.push_back(incidence_.source_ordinal(
                {participant.role, 0, participant.entity_id}));
        cache_.resize(pattern_->size());
        dirty_.resize(pattern_->size());
        std::iota(dirty_.begin(), dirty_.end(), std::size_t{0});
    }

    const std::shared_ptr<const geometry::dependency::ExplicitPattern>& pattern() const noexcept {
        return pattern_;
    }
    const PreparedReadIndex& incidence() const noexcept { return incidence_; }
    EvaluationEpochs epochs() const noexcept { return epochs_; }
    std::uint64_t step_generation() const noexcept { return generation_; }
    std::size_t pending_invalid() const noexcept { return dirty_.size(); }
    std::shared_ptr<const Read> source(containers::PostingTerm key) const {
        return sources_[incidence_.source_ordinal(key)];
    }
    CacheEntry cached(std::uint64_t id) const {
        const auto ordinal = pattern_->ordinal(id);
        return {cache_[ordinal], !std::binary_search(dirty_.begin(), dirty_.end(), ordinal)};
    }
    const Output& result(std::uint64_t id) const {
        const auto entry = cached(id);
        if (!entry.valid) throw std::logic_error("interaction output is absent or invalidated");
        return entry.record->output;
    }
    // Events restrict eligibility independently of value invalidation. Event
    // identity is a role/entity posting key; no temporal biology is inferred.
    ActiveWorklist event_eligibility(
        const std::vector<containers::PostingTerm>& event_sources,
        std::uint64_t event_generation = 0) const {
        return ActiveWorklist(pattern_, pattern_->query(event_sources), event_generation);
    }

    // New source generations must strictly increase. Changing either epoch
    // invalidates all interactions. Empty eligibility evaluates nothing; null
    // eligibility evaluates every dirty interaction. All state/cache publication
    // occurs after all callbacks succeed. Callback external side effects cannot
    // be rolled back and therefore violate this class's purity contract.
    IncrementalStepResult step(std::vector<Read> updates, EvaluationEpochs next_epochs,
                               const ActiveWorklist* eligibility = nullptr) {
        if (eligibility && eligibility->pattern().get() != pattern_.get())
            throw std::invalid_argument("eligibility must use the same immutable dependency snapshot");
        if (generation_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("incremental step generation exhausted");
        using StagedSource = std::pair<std::size_t, std::shared_ptr<const Read>>;
        std::vector<StagedSource> staged_sources;
        std::vector<containers::PostingTerm> changed_terms;
        for (auto& update : updates) {
            const auto slot = incidence_.source_ordinal(update.source);
            if (update.generation <= sources_[slot]->generation)
                throw std::invalid_argument("source generation must strictly increase");
            changed_terms.push_back(update.source);
            staged_sources.emplace_back(slot, std::make_shared<const Read>(std::move(update)));
        }
        std::sort(staged_sources.begin(), staged_sources.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (std::size_t i = 1; i < staged_sources.size(); ++i)
            if (staged_sources[i - 1].first == staged_sources[i].first)
                throw std::invalid_argument("duplicate source update");

        std::vector<std::size_t> pending;
        if (next_epochs != epochs_) {
            pending.resize(pattern_->size());
            std::iota(pending.begin(), pending.end(), std::size_t{0});
        } else {
            const auto changed = pattern_->query(changed_terms);
            std::set_union(dirty_.begin(), dirty_.end(), changed.begin(), changed.end(),
                           std::back_inserter(pending));
        }
        std::vector<std::size_t> selected;
        if (eligibility)
            std::set_intersection(pending.begin(), pending.end(), eligibility->ordinals().begin(),
                                  eligibility->ordinals().end(), std::back_inserter(selected));
        else selected = pending;
        std::vector<std::size_t> remaining;
        std::set_difference(pending.begin(), pending.end(), selected.begin(), selected.end(),
                            std::back_inserter(remaining));
        std::vector<std::shared_ptr<const Record>> staged_records;
        IncrementalStepResult report;
        for (const auto ordinal : selected) {
            const auto& edge = pattern_->edges()[ordinal];
            std::vector<std::shared_ptr<const Read>> reads;
            reads.reserve(edge.participant_count);
            for (std::size_t i = 0; i < edge.participant_count; ++i) {
                const auto slot = participant_slots_[edge.participant_begin + i];
                const auto p = std::lower_bound(staged_sources.begin(), staged_sources.end(), slot,
                    [](const auto& a, std::size_t b) { return a.first < b; });
                reads.push_back(p != staged_sources.end() && p->first == slot ? p->second : sources_[slot]);
            }
            SourceReadSnapshot<State> snapshot(std::move(reads));
            auto output = evaluator_(edge.logical_edge_id, snapshot, next_epochs);
            staged_records.push_back(std::make_shared<const Record>(Record{
                edge.logical_edge_id, next_epochs, std::move(snapshot), std::move(output)}));
            report.evaluated_ids.push_back(edge.logical_edge_id);
        }
        // The remaining operations cannot throw for these standard containers.
        for (auto& item : staged_sources) sources_[item.first] = std::move(item.second);
        for (std::size_t i = 0; i < selected.size(); ++i) cache_[selected[i]] = std::move(staged_records[i]);
        dirty_.swap(remaining);
        epochs_ = next_epochs;
        report.step_generation = ++generation_;
        report.pending_invalid = dirty_.size();
        return report;
    }
private:
    static std::shared_ptr<const geometry::dependency::ExplicitPattern> require_pattern(
        std::shared_ptr<const geometry::dependency::ExplicitPattern> value) {
        if (!value) throw std::invalid_argument("dependency pattern cannot be null");
        return value;
    }
    std::shared_ptr<const geometry::dependency::ExplicitPattern> pattern_;
    PreparedReadIndex incidence_;
    Evaluator evaluator_;
    EvaluationEpochs epochs_;
    std::uint64_t generation_ = 0;
    std::vector<std::shared_ptr<const Read>> sources_;
    std::vector<std::size_t> participant_slots_, dirty_;
    std::vector<std::shared_ptr<const Record>> cache_;
};

} // namespace cellerator::execution
