#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace cellerator::execution::conditional {

// Logical module identity and the exact parameter revision selected for this
// invocation. Physical placement and residency cannot change either field.
struct ExpertKey {
    std::uint64_t expert_id = 0;
    std::uint64_t revision = 0;
};
inline bool operator==(const ExpertKey& a, const ExpertKey& b) noexcept {
    return a.expert_id == b.expert_id && a.revision == b.revision;
}
inline bool operator!=(const ExpertKey& a, const ExpertKey& b) noexcept { return !(a == b); }
inline bool operator<(const ExpertKey& a, const ExpertKey& b) noexcept {
    return std::tie(a.expert_id, a.revision) < std::tie(b.expert_id, b.revision);
}

// Origin identity is independent of a physical input/output row. rank names an
// originating process; instance may identify a cell, trajectory or other owner.
struct Origin {
    std::uint32_t rank = 0;
    std::uint64_t instance = 0;
    std::uint64_t row = 0;
};
inline bool operator==(const Origin& a, const Origin& b) noexcept {
    return a.rank == b.rank && a.instance == b.instance && a.row == b.row;
}
inline bool operator!=(const Origin& a, const Origin& b) noexcept { return !(a == b); }
inline bool operator<(const Origin& a, const Origin& b) noexcept {
    return std::tie(a.rank, a.instance, a.row) < std::tie(b.rank, b.instance, b.row);
}

struct Selection {
    ExpertKey expert;
    Origin origin;
    double weight = 1.0;
};

// Immutable sparse preparation, independent of a routing policy or module
// implementation. Every selected occurrence survives, including repeated,
// zero-weight and signed-weight routes. Stable grouping batches exact logical
// module/revision keys and never substitutes an expert for locality.
class DispatchPlan final {
public:
    explicit DispatchPlan(std::vector<Selection> selections);
    DispatchPlan(const DispatchPlan&) = default;
    DispatchPlan& operator=(const DispatchPlan&) = delete;
    const std::vector<Selection>& selections() const noexcept { return selections_; }
    const std::vector<ExpertKey>& experts() const noexcept { return experts_; }
    const std::vector<std::size_t>& expert_offsets() const noexcept { return expert_offsets_; }
    const std::vector<std::size_t>& dispatch_to_selection() const noexcept { return dispatch_to_selection_; }
    const std::vector<std::size_t>& selection_to_dispatch() const noexcept { return selection_to_dispatch_; }
    const std::vector<Origin>& origins() const noexcept { return origins_; }
    const std::vector<Origin>& dispatch_origins() const noexcept { return dispatch_origins_; }
    const std::vector<std::size_t>& dispatch_origin_slots() const noexcept { return dispatch_origin_slots_; }
    const std::vector<double>& dispatch_weights() const noexcept { return dispatch_weights_; }
    const std::vector<std::size_t>& selection_to_origin() const noexcept { return selection_to_origin_; }
    const std::vector<std::size_t>& origin_offsets() const noexcept { return origin_offsets_; }
    const std::vector<std::size_t>& origin_dispatch_indices() const noexcept { return origin_dispatch_indices_; }
    std::size_t size() const noexcept { return selections_.size(); }
    std::size_t origin_count() const noexcept { return origins_.size(); }

private:
    std::vector<Selection> selections_;
    std::vector<ExpertKey> experts_;
    std::vector<Origin> origins_, dispatch_origins_;
    std::vector<std::size_t> expert_offsets_, dispatch_to_selection_, selection_to_dispatch_;
    std::vector<std::size_t> selection_to_origin_, origin_offsets_, origin_dispatch_indices_;
    std::vector<std::size_t> dispatch_origin_slots_;
    std::vector<double> dispatch_weights_;
};

inline std::size_t checked_size(std::size_t rows, std::size_t features) {
    if (features == 0) throw std::invalid_argument("conditional feature count must be positive");
    if (rows > std::numeric_limits<std::size_t>::max() / features)
        throw std::length_error("conditional buffer shape overflow");
    return rows * features;
}

template<class Storage>
struct InputBatch {
    const Storage* data;
    const Origin* origins;
    std::size_t rows;
    std::size_t features;
    // Original occurrence indices permit keyed stochastic behavior or other
    // provenance-sensitive modules without depending on physical batch order.
    const std::size_t* selection_indices = nullptr;
    const Storage& at(std::size_t row, std::size_t feature) const {
        if (row >= rows || feature >= features) throw std::out_of_range("input batch index outside shape");
        return data[row * features + feature];
    }
};

template<class Storage>
struct OutputBatch {
    Storage* data;
    std::size_t rows;
    std::size_t features;
    Storage& at(std::size_t row, std::size_t feature) const {
        if (row >= rows || feature >= features) throw std::out_of_range("output batch index outside shape");
        return data[row * features + feature];
    }
};

// Read each logical origin exactly once through an opaque provider, then copy
// that common row to each selected dispatch occurrence. The provider writes
// `features` values and must not retain the one-row scratch pointer. Its source
// may be sparse/ragged state or an external layout; dense storage is optional.
template<class Storage, class InputProvider>
std::vector<Storage> gather_inputs(const DispatchPlan& plan, std::size_t features, InputProvider&& provider) {
    std::vector<Storage> dispatched(checked_size(plan.size(), features));
    if (plan.size() == 0) return dispatched;
    std::vector<Storage> row(features);
    for (std::size_t origin = 0; origin < plan.origin_count(); ++origin) {
        provider(plan.origins()[origin], row.data(), features);
        for (std::size_t i = plan.origin_offsets()[origin]; i < plan.origin_offsets()[origin + 1]; ++i) {
            const auto dispatch = plan.origin_dispatch_indices()[i];
            std::copy(row.begin(), row.end(), dispatched.begin() + dispatch * features);
        }
    }
    return dispatched;
}

// Combine in the original selected-occurrence order within each origin. This
// makes floating-point addition order independent of expert-major batching.
// Precision of storage, multiplication/accumulation and output is explicit.
template<class Storage, class Accumulator = Storage, class Output = Accumulator>
std::vector<Output> combine(const DispatchPlan& plan, const std::vector<Storage>& dispatch_values,
                            std::size_t features) {
    if (dispatch_values.size() != checked_size(plan.size(), features))
        throw std::invalid_argument("dispatch output shape mismatch");
    std::vector<Output> output(checked_size(plan.origin_count(), features));
    for (std::size_t origin = 0; origin < plan.origin_count(); ++origin)
        for (std::size_t feature = 0; feature < features; ++feature) {
            Accumulator value{0};
            for (std::size_t i = plan.origin_offsets()[origin]; i < plan.origin_offsets()[origin + 1]; ++i) {
                const auto dispatch = plan.origin_dispatch_indices()[i];
                const auto selection = plan.dispatch_to_selection()[dispatch];
                value += static_cast<Accumulator>(plan.selections()[selection].weight)
                       * static_cast<Accumulator>(dispatch_values[dispatch * features + feature]);
            }
            output[origin * features + feature] = static_cast<Output>(value);
        }
    return output;
}

// Transpose of the weighted combine only: map origin output adjoints back to
// every dispatch occurrence with its unchanged route weight. Expert and weight
// derivatives remain with the module/autograd owner.
template<class Storage, class Accumulator = Storage, class Output = Accumulator>
std::vector<Output> transpose_combine(const DispatchPlan& plan,
                                      const std::vector<Storage>& origin_gradients,
                                      std::size_t features) {
    if (origin_gradients.size() != checked_size(plan.origin_count(), features))
        throw std::invalid_argument("origin gradient shape mismatch");
    std::vector<Output> dispatched(checked_size(plan.size(), features));
    for (std::size_t dispatch = 0; dispatch < plan.size(); ++dispatch) {
        const auto selection = plan.dispatch_to_selection()[dispatch];
        const auto origin = plan.selection_to_origin()[selection];
        for (std::size_t feature = 0; feature < features; ++feature)
            dispatched[dispatch * features + feature] = static_cast<Output>(
                static_cast<Accumulator>(plan.selections()[selection].weight)
              * static_cast<Accumulator>(origin_gradients[origin * features + feature]));
    }
    return dispatched;
}

// Reference route/group/execute/combine path. Modules are generic callbacks on
// contiguous rows for one exact ExpertKey. They may be learned, numerical or
// custom; they must write every output and not retain borrowed views. Placement,
// communication, caching and scheduling are deliberately separate consumers.
// Equivalence to independent route calls additionally requires row-separable
// modules (including stochastic/normalization semantics). Grouping cannot make
// batch-dependent modules mathematically invariant by itself.
template<class InputStorage, class OutputStorage = InputStorage, class Accumulator = OutputStorage,
         class CombinedOutput = Accumulator, class InputProvider, class ExpertCallback>
std::vector<CombinedOutput> execute(const DispatchPlan& plan, std::size_t input_features,
                                   std::size_t output_features, InputProvider&& provider,
                                   ExpertCallback&& callback) {
    std::vector<OutputStorage> outputs(checked_size(plan.size(), output_features));
    auto inputs = gather_inputs<InputStorage>(plan, input_features, std::forward<InputProvider>(provider));
    for (std::size_t expert = 0; expert < plan.experts().size(); ++expert) {
        const auto begin = plan.expert_offsets()[expert];
        const auto rows = plan.expert_offsets()[expert + 1] - begin;
        const InputBatch<InputStorage> input{inputs.data() + begin * input_features,
                                            plan.dispatch_origins().data() + begin, rows, input_features,
                                            plan.dispatch_to_selection().data() + begin};
        OutputBatch<OutputStorage> output{outputs.data() + begin * output_features, rows, output_features};
        callback(plan.experts()[expert], input, output);
    }
    return combine<OutputStorage, Accumulator, CombinedOutput>(plan, outputs, output_features);
}

} // namespace cellerator::execution::conditional
