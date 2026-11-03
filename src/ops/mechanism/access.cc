#include <Cellerator/ops/mechanism/access.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace cellerator::ops::mechanism {
namespace {

bool supported_phase(Phase phase) noexcept {
    switch (phase) {
    case Phase::condition:
    case Phase::primal:
    case Phase::input_vjp:
    case Phase::condition_vjp:
    case Phase::parameter_vjp:
    case Phase::jvp:
        return true;
    }
    return false;
}

bool numeric_axis_space(dependency::SelectionSpace space) noexcept {
    return space == dependency::SelectionSpace::coordinates
        || space == dependency::SelectionSpace::occurrences
        || space == dependency::SelectionSpace::derivative_coordinates;
}

template <typename Identity>
bool identity_less(const Identity& lhs, const Identity& rhs) noexcept {
    return lhs.low < rhs.low
        || (lhs.low == rhs.low && lhs.high < rhs.high);
}

template <typename Identity, typename IdentityAt>
std::size_t find_index(const std::vector<std::size_t>& sorted_indices,
                       const Identity& identity,
                       IdentityAt identity_at) noexcept {
    const auto found = std::lower_bound(
        sorted_indices.begin(), sorted_indices.end(), identity,
        [&](std::size_t index, const Identity& key) {
            return identity_less(identity_at(index), key);
        });
    if (found == sorted_indices.end()
        || !structure::same_identity(identity_at(*found), identity))
        return sorted_indices.size();
    return *found;
}

class AccessIndex {
public:
    AccessIndex(std::span<const dependency::SourceDescriptor> sources,
                std::span<const dependency::SelectionDescriptor> selections)
        : sources_(sources), selections_(selections) {
        source_order_.resize(sources_.size());
        std::iota(source_order_.begin(), source_order_.end(), 0);
        std::sort(source_order_.begin(), source_order_.end(),
                  [&](std::size_t lhs, std::size_t rhs) {
                      const auto& left = sources_[lhs].reference.identity;
                      const auto& right = sources_[rhs].reference.identity;
                      if (identity_less(left, right)) return true;
                      if (identity_less(right, left)) return false;
                      return lhs < rhs;
                  });
        selection_order_.resize(selections_.size());
        std::iota(selection_order_.begin(), selection_order_.end(), 0);
        std::sort(selection_order_.begin(), selection_order_.end(),
                  [&](std::size_t lhs, std::size_t rhs) {
                      const auto& left = selections_[lhs].reference.identity;
                      const auto& right = selections_[rhs].reference.identity;
                      if (identity_less(left, right)) return true;
                      if (identity_less(right, left)) return false;
                      return lhs < rhs;
                  });
    }

    const dependency::SourceDescriptor* source(
        dependency::source_id identity) const noexcept {
        const auto index = find_index(source_order_, identity,
                                      [&](std::size_t i) {
                                          return sources_[i].reference.identity;
                                      });
        return index == source_order_.size() ? nullptr : &sources_[index];
    }

    const dependency::SelectionDescriptor* selection(
        dependency::selection_id identity) const noexcept {
        const auto index = find_index(selection_order_, identity,
                                      [&](std::size_t i) {
                                          return selections_[i].reference.identity;
                                      });
        return index == selection_order_.size() ? nullptr : &selections_[index];
    }

private:
    std::span<const dependency::SourceDescriptor> sources_;
    std::span<const dependency::SelectionDescriptor> selections_;
    std::vector<std::size_t> source_order_;
    std::vector<std::size_t> selection_order_;
};

bool read_less(const dependency::ReadDependency& lhs,
               const dependency::ReadDependency& rhs) noexcept {
    if (identity_less(lhs.source.identity, rhs.source.identity)) return true;
    if (identity_less(rhs.source.identity, lhs.source.identity)) return false;
    if (identity_less(lhs.selection.identity, rhs.selection.identity)) return true;
    if (identity_less(rhs.selection.identity, lhs.selection.identity)) return false;
    return false;
}

bool same_read_key(const dependency::ReadDependency& lhs,
                   const dependency::ReadDependency& rhs) noexcept {
    return dependency::same_source(lhs.source, rhs.source)
        && dependency::same_selection(lhs.selection, rhs.selection);
}

bool condition_phase_covers(
    const dependency::ConditionReads& condition,
    const PhaseDemand& phase,
    std::span<const std::size_t> read_order,
    const AccessIndex& index) noexcept {
    for (const auto& required : condition.inputs) {
        // Exact source/selection matches are the common case and remain
        // logarithmic even when many groups share a condition phase.
        const auto exact = std::lower_bound(
            read_order.begin(), read_order.end(), required,
            [&](std::size_t read_index, const dependency::ReadDependency& key) {
                return read_less(phase.reads[read_index], key);
            });
        if (exact != read_order.end()
            && same_read_key(phase.reads[*exact], required))
            continue;

        const auto* required_selection = index.selection(
            required.selection.identity);
        if (required_selection == nullptr
            || index.source(required.source.identity) == nullptr)
            return false; // Normally reported earlier by validate_conditions.

        // A wider phase read may satisfy a narrower declared dependency. The
        // source run is indexed, then selection subset checks fail closed.
        const auto first_source = std::lower_bound(
            read_order.begin(), read_order.end(), required.source.identity,
            [&](std::size_t read_index, dependency::source_id key) {
                return identity_less(phase.reads[read_index].source.identity,
                                     key);
            });
        bool covered = false;
        for (auto candidate = first_source; candidate != read_order.end();
             ++candidate) {
            const auto& phase_read = phase.reads[*candidate];
            if (!structure::same_identity(phase_read.source.identity,
                                          required.source.identity))
                break;
            if (!dependency::same_source(phase_read.source, required.source))
                continue;
            const auto* phase_selection = index.selection(
                phase_read.selection.identity);
            if (phase_selection != nullptr
                && dependency::selection_covers(*phase_selection,
                                                *required_selection)) {
                covered = true;
                break;
            }
        }
        if (!covered) return false;
    }
    return true;
}

dependency::Status validate_output(
    const OutputAccess& output, const AccessIndex& index) noexcept {
    if (!structure::valid_identity(output.destination.identity)
        || !structure::valid_identity(output.destination.domain)
        || !structure::valid_identity(output.destination.representation)
        || !structure::valid_identity(output.selection.identity)
        || !structure::valid_identity(output.selection.domain))
        return {dependency::StatusCode::invalid_identity, 0};
    if (!numeric_axis_space(output.selection.space))
        return {dependency::StatusCode::invalid_read, 0};
    if (!execution::valid_output_effect_contract(output.effect))
        return {dependency::StatusCode::invalid_output_effect, 0};

    const auto* source = index.source(output.destination.identity);
    if (source == nullptr)
        return {dependency::StatusCode::unknown_source, 0};
    if (!structure::same_identity(output.destination.domain,
                                  source->reference.domain))
        return {dependency::StatusCode::domain_mismatch, 0};
    if (!structure::same_identity(output.destination.representation,
                                  source->reference.representation))
        return {dependency::StatusCode::invalid_read, 0};

    const auto* selection = index.selection(output.selection.identity);
    if (selection == nullptr)
        return {dependency::StatusCode::unknown_selection, 0};
    if (!structure::same_identity(output.selection.domain,
                                  selection->reference.domain)
        || !structure::same_identity(output.selection.domain,
                                     source->axis.domain))
        return {dependency::StatusCode::domain_mismatch, 0};
    if (output.selection.space != selection->reference.space)
        return {dependency::StatusCode::invalid_read, 0};
    if (!structure::same_identity(selection->order,
                                  source->axis.logical_order))
        return {dependency::StatusCode::order_mismatch, 0};
    if (selection->extent != source->axis.extent)
        return {dependency::StatusCode::extent_mismatch, 0};
    return {};
}

} // namespace

dependency::Status validate_access(
    const AccessContract& access,
    std::span<const dependency::SourceDescriptor> sources,
    std::span<const dependency::SelectionDescriptor> selections) noexcept {
    try {
        if (!structure::valid_identity(access.identity)
            || !structure::valid_identity(access.operation))
            return {dependency::StatusCode::invalid_identity, 0};
        if (access.revision == 0)
            return {dependency::StatusCode::invalid_epoch, 0};
        if (access.phases.empty())
            return {dependency::StatusCode::unsupported_scope, 0};

        auto status = dependency::validate_reads(sources, selections, {});
        if (!status) return status;
        status = dependency::validate_conditions(sources, selections,
                                                 access.conditions);
        if (!status) return status;

        AccessIndex index{sources, selections};
        const PhaseDemand* condition_phase = nullptr;
        for (std::size_t i = 0; i < access.phases.size(); ++i) {
            const auto& phase = access.phases[i];
            if (!supported_phase(phase.phase))
                return {dependency::StatusCode::unsupported_scope,
                        static_cast<std::uint64_t>(i)};
            for (std::size_t j = 0; j < i; ++j) {
                if (phase.phase == access.phases[j].phase)
                    return {dependency::StatusCode::duplicate_identity,
                            static_cast<std::uint64_t>(i)};
            }
            if (phase.phase == Phase::condition) condition_phase = &phase;

            status = dependency::validate_reads(sources, selections,
                                                phase.reads);
            if (!status)
                return {status.code, static_cast<std::uint64_t>(i)};
            for (std::size_t j = 0; j < phase.writes.size(); ++j) {
                status = validate_output(phase.writes[j], index);
                if (!status)
                    return {status.code, static_cast<std::uint64_t>(i)};
            }
        }

        bool has_condition_inputs = false;
        for (const auto& condition : access.conditions)
            has_condition_inputs = has_condition_inputs
                || !condition.inputs.empty();
        if (has_condition_inputs && condition_phase == nullptr)
            return {dependency::StatusCode::invalid_condition, 0};
        if (condition_phase != nullptr) {
            std::vector<std::size_t> read_order(condition_phase->reads.size());
            std::iota(read_order.begin(), read_order.end(), 0);
            std::sort(read_order.begin(), read_order.end(),
                      [&](std::size_t lhs, std::size_t rhs) {
                          if (read_less(condition_phase->reads[lhs],
                                        condition_phase->reads[rhs]))
                              return true;
                          if (read_less(condition_phase->reads[rhs],
                                        condition_phase->reads[lhs]))
                              return false;
                          return lhs < rhs;
                      });
            for (std::size_t i = 0; i < access.conditions.size(); ++i) {
                if (!condition_phase_covers(access.conditions[i],
                                            *condition_phase, read_order,
                                            index))
                    return {dependency::StatusCode::invalid_read,
                            static_cast<std::uint64_t>(i)};
            }
        }
        return {};
    } catch (...) {
        return {dependency::StatusCode::allocation_failure, 0};
    }
}

} // namespace cellerator::ops::mechanism
