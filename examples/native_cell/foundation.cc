#include <Cellerator/cell.hh>
#include <Cellerator/memory/view.hh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// A model borrows each row's values through its Cell instance and retains a
// pooled context once through shared model ownership.
struct RowValuesView {
    cellerator::memory::const_array_view<double> values;
};

struct BorrowedRow {
    const double* values;
    std::size_t count;
    std::size_t context_slot;

    RowValuesView read(cellerator::ViewTag<RowValuesView>) const & {
        return {{values, count,
                 {cellerator::memory::domain::host, -1, -1, 0}}};
    }
};

// This representation names a row inside pooled contiguous values instead of
// borrowing a row-specific pointer. It exposes the same typed values contract.
struct PooledRowSlot {
    const double* pooled_values;
    std::size_t pool_offset;
    std::size_t count;
    std::size_t context_slot;

    RowValuesView read(cellerator::ViewTag<RowValuesView>) const & {
        const double* row_values = pooled_values;
        if (pool_offset != 0) row_values += pool_offset;
        return {{row_values, count,
                 {cellerator::memory::domain::host, -1, -1, 0}}};
    }
};

struct PooledContextModel {
    explicit PooledContextModel(std::shared_ptr<const std::vector<double>> context)
        : context(std::move(context)) {}

    double operator()(const BorrowedRow& row) const {
        return evaluate_row(row);
    }

    double operator()(const PooledRowSlot& row) const {
        return evaluate_row(row);
    }

    // A model batch operation reads several continuing Cell instances
    // directly. It does not call Cell::forward for each item or wrap the batch
    // itself in another Cell.
    template<class Cells>
    std::vector<double> evaluate_batch(const Cells& cells) const {
        std::vector<double> outputs;
        outputs.reserve(cells.size());
        for (const auto& cell : cells) outputs.push_back(evaluate_row(cell.instance()));
        return outputs;
    }

    std::shared_ptr<const std::vector<double>> context;

private:
    template<class RowRepresentation>
    double evaluate_row(const RowRepresentation& row) const {
        const auto values = row.read(cellerator::ViewTag<RowValuesView>{}).values;
        double total = context->at(row.context_slot);
        for (std::size_t i = 0; i < values.count; ++i) total += values.data[i];
        return total;
    }
};

void shared_model_with_borrowed_rows() {
    std::array<double, 2> first_values{1.0, 2.0};
    std::array<double, 3> second_values{4.0, 5.0, 6.0};
    const std::array<double, 5> pooled_values{1.0, 2.0, 4.0, 5.0, 6.0};
    auto context = std::make_shared<const std::vector<double>>(
        std::vector<double>{10.0, 20.0});
    auto model = std::make_shared<PooledContextModel>(context);

    auto first = cellerator::make_cell(
        model, BorrowedRow{first_values.data(), first_values.size(), 0});
    auto second = cellerator::make_cell(
        model, BorrowedRow{second_values.data(), second_values.size(), 1});
    auto pooled_first = cellerator::make_cell(
        model, PooledRowSlot{pooled_values.data(), 0, 2, 0});
    auto pooled_second = cellerator::make_cell(
        model, PooledRowSlot{pooled_values.data(), 2, 3, 1});

    static_assert(!std::is_same<decltype(first), decltype(pooled_first)>::value,
                  "the two Cells should retain distinct instance types");
    require(&first.model() == &second.model() &&
                &first.model() == &pooled_first.model() &&
                &first.model() == &pooled_second.model(),
            "different instance types should share one model object");
    require(first.forward() == 13.0, "first borrowed row result is incorrect");
    require(second.forward() == 35.0, "second borrowed row result is incorrect");
    require(pooled_first.forward() == first.forward() &&
                pooled_second.forward() == second.forward(),
            "different row representations should produce equivalent results");
    std::cout << "shared callable: borrowed/pooled=" << first.forward() << '/'
              << pooled_first.forward() << ',' << second.forward() << '/'
              << pooled_second.forward() << " pooled-context=" << context->size() << '\n';
}

// Ordinary value members copy independently. A shared_ptr member shares only
// the resource that the instance type explicitly chooses to share.
struct OwnedRow {
    std::vector<double> values;
};

struct OwnedSumModel {
    double operator()(const OwnedRow& row) const {
        double total = 0.0;
        for (double value : row.values) total += value;
        return total;
    }
};

struct MixedInstance {
    std::shared_ptr<std::vector<int>> shared_values;
    std::vector<int> private_values;
};

struct MixedModel {
    int operator()(const MixedInstance& instance) const {
        int total = 0;
        for (int value : *instance.shared_values) total += value;
        for (int value : instance.private_values) total += value;
        return total;
    }
};

struct MoveOnlyState {
    explicit MoveOnlyState(int value) : value(std::make_unique<int>(value)) {}
    MoveOnlyState(MoveOnlyState&&) = default;
    MoveOnlyState& operator=(MoveOnlyState&&) = default;
    MoveOnlyState(const MoveOnlyState&) = delete;
    MoveOnlyState& operator=(const MoveOnlyState&) = delete;

    std::unique_ptr<int> value;
};

struct MoveOnlyModel {
    int operator()(MoveOnlyState& state) const { return *state.value; }
};

void copy_and_move_policy() {
    OwnedSumModel owned_model;
    auto owned = cellerator::make_cell(owned_model, OwnedRow{{1.0, 2.0}});
    auto independent_copy = owned;
    independent_copy.instance().values[0] = 40.0;
    require(&owned.model() == &independent_copy.model(),
            "borrowed model handles should continue to name the same model");
    require(owned.forward() == 3.0 && independent_copy.forward() == 42.0,
            "value-owned instance data should copy independently");

    MixedModel mixed_model;
    auto pooled = std::make_shared<std::vector<int>>(std::initializer_list<int>{2, 3});
    auto mixed = cellerator::make_cell(
        mixed_model, MixedInstance{pooled, {5}});
    auto mixed_copy = mixed;
    mixed_copy.instance().shared_values->push_back(7);
    mixed_copy.instance().private_values[0] = 11;
    require(mixed.forward() == 17 && mixed_copy.forward() == 23,
            "instance copy should share only its explicit pooled member");

    MoveOnlyModel move_model;
    using MoveCell = decltype(cellerator::make_cell(move_model, MoveOnlyState{1}));
    static_assert(!std::is_copy_constructible<MoveCell>::value,
                  "a move-only instance should make its Cell non-copyable");
    static_assert(std::is_move_constructible<MoveCell>::value,
                  "a move-only instance should remain movable");
    auto original = cellerator::make_cell(move_model, MoveOnlyState{29});
    auto moved = std::move(original);
    require(moved.forward() == 29, "moved Cell lost its move-only state");
    std::cout << "copy policy: independent=" << owned.forward() << ','
              << independent_copy.forward() << " shared/private="
              << mixed.forward() << ',' << mixed_copy.forward()
              << " move-only=" << moved.forward() << '\n';
}

struct JointState {
    double* shared_part;
    double private_part;
};

struct JointModel {
    double operator()(JointState& selected, const JointState& partner) const {
        if (selected.shared_part != partner.shared_part)
            throw std::invalid_argument("joint cells do not share the declared part");
        *selected.shared_part += 1.0;
        selected.private_part += partner.private_part;
        return *selected.shared_part + selected.private_part + partner.private_part;
    }
};

void joint_operation() {
    double shared = 3.0;
    JointModel model;
    auto first = cellerator::make_cell(model, JointState{&shared, 2.0});
    auto second = cellerator::make_cell(model, JointState{&shared, 5.0});
    const double result = first.forward(second.instance());
    require(result == 16.0 && shared == 4.0,
            "joint operation should update its shared part and selected instance");
    require(second.instance().private_part == 5.0,
            "joint operation should leave the partner's private part alone");
    std::cout << "joint cells: shared=" << shared
              << " selected-private=" << first.instance().private_part
              << " partner-private=" << second.instance().private_part << '\n';
}

struct SplitPartsView {
    cellerator::memory::const_array_view<double> left;
    cellerator::memory::const_array_view<double> right;
};

class SplitPartsOwner {
public:
    SplitPartsOwner() : left_{1.0, 2.0}, right_{7.0, 8.0} { publish(); }

    SplitPartsOwner(const SplitPartsOwner&) = delete;
    SplitPartsOwner& operator=(const SplitPartsOwner&) = delete;

    SplitPartsView view() const { return {left_view_, right_view_}; }

    bool relocate_left_on_host() {
        // Allocate while the old buffer is still live, copy on the host, then
        // publish the new descriptor. The right part remains separately owned.
        const double* old_left = left_.data();
        std::vector<double> replacement(left_.size());
        std::copy(left_.begin(), left_.end(), replacement.begin());
        const bool address_changed = replacement.data() != old_left;
        left_.swap(replacement);
        publish();
        return address_changed;
    }

private:
    void publish() {
        const cellerator::memory::placement host{
            cellerator::memory::domain::host, -1, -1, 0};
        left_view_ = {left_.data(), left_.size(), host};
        right_view_ = {right_.data(), right_.size(), host};
    }

    std::vector<double> left_;
    std::vector<double> right_;
    cellerator::memory::const_array_view<double> left_view_{};
    cellerator::memory::const_array_view<double> right_view_{};
};

struct SplitPartsInstance {
    SplitPartsOwner* owner;
    SplitPartsView read(cellerator::ViewTag<SplitPartsView>) const & {
        return owner->view();
    }
};

struct SplitPartsModel {
    double operator()(const SplitPartsInstance& instance) const {
        const auto parts = instance.read(cellerator::ViewTag<SplitPartsView>{});
        return parts.left.data[0] + parts.right.data[0];
    }
};

void owner_relocation_and_separate_parts() {
    SplitPartsOwner owner;
    SplitPartsModel model;
    auto first = cellerator::make_cell(model, SplitPartsInstance{&owner});
    auto second = first;
    const auto before = first.view<SplitPartsView>();
    require(before.left.data != before.right.data,
            "separate logical parts should remain separately allocated");
    const auto* right_before = before.right.data;
    const bool relocated = owner.relocate_left_on_host();
    const auto after = second.view<SplitPartsView>();
    require(relocated && after.left.data[0] == 1.0,
            "owner relocation should publish the copied host part");
    require(after.right.data == right_before,
            "relocating one part must not move the other part");
    require(first.forward() == 8.0 && second.forward() == 8.0,
            "cells sharing the owner should observe its refreshed descriptors");
    std::cout << "owner relocation: copied-left-buffer=true right-buffer-shared=true\n";
}

void pooled_batch_direct() {
    const std::array<double, 7> pooled_values{1.0, 2.0, 4.0, 5.0, 6.0, 3.0, 4.0};
    auto context = std::make_shared<const std::vector<double>>(
        std::vector<double>{10.0, 20.0});
    auto model = std::make_shared<PooledContextModel>(context);

    auto first = cellerator::make_cell(
        model, PooledRowSlot{pooled_values.data(), 0, 2, 0});
    using PooledCell = decltype(first);
    std::vector<PooledCell> cells;
    cells.push_back(std::move(first));
    cells.push_back(cellerator::make_cell(
        model, PooledRowSlot{pooled_values.data(), 2, 3, 1}));
    cells.push_back(cellerator::make_cell(
        model, PooledRowSlot{pooled_values.data(), 5, 2, 0}));

    for (const auto& cell : cells)
        require(&cell.model() == &cells.front().model(),
                "pooled batch Cells should share their model context");
    const auto outputs = model->evaluate_batch(cells);
    require(outputs == std::vector<double>{13.0, 35.0, 17.0},
            "direct pooled Cell batch results are incorrect");
    std::cout << "pooled Cell batch: instances=" << cells.size()
              << " direct-model-results=" << outputs[0] << ',' << outputs[1]
              << ',' << outputs[2] << '\n';
}

struct CallerEstimate {
    std::string host_target;
    bool feasible;
    double estimated_cost;
};

struct SelectedOccurrence {
    // The caller has already selected this cell occurrence and expert. Host
    // estimates are context supplied for this selected execution only.
    std::string cell_identity;
    std::size_t occurrence;
    int expert_identity;
    double activation;
    std::vector<CallerEstimate> estimates;
};

struct HostResult {
    std::string cell_identity;
    std::size_t occurrence;
    int expert_identity;
    std::string host_target;
    double value;
};

struct CellActivations {
    std::string identity;
    std::vector<SelectedOccurrence> selected;
};

struct SelectedExpertModel {
    // Shared expert parameters. Each Cell carries its own selected activation
    // context; an activation is a work item, not a separate Cell instance.
    double multiplier = 2.0;
    double offset = 7.0;

    std::vector<SelectedOccurrence> operator()(const CellActivations& instance) const {
        for (const auto& occurrence : instance.selected)
            require(occurrence.cell_identity == instance.identity,
                    "selected work belongs to another continuing Cell");
        return instance.selected;
    }

    HostResult execute(const SelectedOccurrence& occurrence,
                       const std::string& host_target) const {
        double value;
        switch (occurrence.expert_identity) {
        case 2: value = occurrence.activation * multiplier; break;
        case 7: value = occurrence.activation + offset; break;
        default: throw std::invalid_argument("selected expert is not implemented here");
        }
        return {occurrence.cell_identity, occurrence.occurrence,
                occurrence.expert_identity, host_target, value};
    }
};

std::vector<HostResult> choose_and_execute_selected_host_occurrences(
    const SelectedExpertModel& model,
    const std::vector<SelectedOccurrence>& selected) {
    // Model-produced work already names its experts. The caller supplies
    // feasible destinations and estimates; this local policy never substitutes
    // an expert to make placement cheaper. These records are not a core schema.
    std::map<std::string, std::vector<SelectedOccurrence>> host_buffers;
    std::vector<std::string> selected_targets;
    selected_targets.reserve(selected.size());
    for (const auto& occurrence : selected) {
        const CallerEstimate* best = nullptr;
        for (const auto& estimate : occurrence.estimates) {
            if (!estimate.feasible) continue;
            if (!best || estimate.estimated_cost < best->estimated_cost)
                best = &estimate;
        }
        if (!best) throw std::runtime_error("caller supplied no feasible host target");

        // Explicitly copy the activation record into a distinct host buffer.
        host_buffers[best->host_target].push_back(occurrence);
        selected_targets.push_back(best->host_target);
        std::cout << "host-buffer copy: cell=" << occurrence.cell_identity
                  << " occurrence=" << occurrence.occurrence
                  << " expert=" << occurrence.expert_identity
                  << " target=" << best->host_target << '\n';
    }

    std::vector<HostResult> results;
    std::map<std::string, std::size_t> consumed_per_target;
    results.reserve(selected.size());
    for (const auto& target : selected_targets) {
        const auto& target_buffer = host_buffers.at(target);
        const auto slot = consumed_per_target[target]++;
        // Execute the copied work, restoring the original occurrence order.
        results.push_back(model.execute(target_buffer.at(slot), target));
    }
    return results;
}

void caller_cost_host_choice() {
    SelectedExpertModel model;
    auto first = cellerator::make_cell(model, CellActivations{"cell-A", {
        {"cell-A", 3, 2, 1.5, {{"host-0", true, 8.0}, {"host-1", true, 3.0}}},
        {"cell-A", 8, 7, 2.0, {{"host-0", false, 1.0}, {"host-1", true, 5.0}}},
    }});
    auto second = cellerator::make_cell(model, CellActivations{"cell-B", {
        {"cell-B", 1, 2, 4.0, {{"host-0", true, 2.0}, {"host-1", true, 6.0}}},
    }});
    require(&first.model() == &second.model(), "expert parameters should be shared");
    auto selected = first.forward();
    const auto second_selected = second.forward();
    selected.insert(selected.end(), second_selected.begin(), second_selected.end());
    const auto results = choose_and_execute_selected_host_occurrences(model, selected);
    require(results.size() == 3, "host choice lost an activation occurrence");
    require(results[0].cell_identity == "cell-A" && results[0].occurrence == 3 &&
                results[0].expert_identity == 2 && results[0].host_target == "host-1" &&
                results[0].value == 3.0,
            "first selected occurrence lost its expert or caller-cost choice");
    require(results[1].cell_identity == "cell-A" && results[1].occurrence == 8 &&
                results[1].expert_identity == 7 && results[1].host_target == "host-1" &&
                results[1].value == 9.0,
            "infeasible target should not receive the selected expert");
    require(results[2].cell_identity == "cell-B" && results[2].occurrence == 1 &&
                results[2].expert_identity == 2 && results[2].host_target == "host-0" &&
                results[2].value == 8.0,
            "second cell occurrence identity was not preserved");
    std::cout << "host execution: continuing-cells=2 activation-occurrences="
              << results.size() << " occurrence-order-and-expert-preserved=true\n";
}

struct CacheInput {
    double value;
};

struct RevisionedModel {
    explicit RevisionedModel(double initial_scale) : scale(initial_scale) {}

    void set_scale(double next) {
        scale = next;
        ++revision;
    }

    double operator()(const CacheInput& input) const {
        if (cached_revision != revision) {
            cached_scale = scale;
            cached_revision = revision;
            ++cache_rebuilds;
        }
        return input.value * cached_scale;
    }

    double scale;
    std::uint64_t revision = 1;
    mutable std::uint64_t cached_revision = 0;
    mutable double cached_scale = 0.0;
    mutable std::size_t cache_rebuilds = 0;
};

void model_owned_revision_cache() {
    RevisionedModel model(2.0);
    auto first = cellerator::make_cell(model, CacheInput{3.0});
    auto second = first;
    second.instance().value = 4.0;
    require(first.forward() == 6.0 && second.forward() == 8.0,
            "shared model cache returned incorrect values");
    require(model.cache_rebuilds == 1,
            "shared cells should reuse the model-owned cache revision");
    first.model().set_scale(5.0);
    require(first.forward() == 15.0 && model.cache_rebuilds == 2,
            "model revision should invalidate its own cached derived value");
    std::cout << "model cache: revision=" << model.revision
              << " rebuilds=" << model.cache_rebuilds << '\n';
}

struct FutureInput {
    double value;
};

struct DeferredModel {
    explicit DeferredModel(std::shared_ptr<const std::vector<double>> resources)
        : resources(std::move(resources)) {}

    std::future<double> operator()(const FutureInput& input) const {
        auto retained = resources;
        const double value = input.value;
        return std::async(std::launch::deferred,
                          [retained = std::move(retained), value] {
                              return retained->at(0) + value;
                          });
    }

    std::shared_ptr<const std::vector<double>> resources;
};

void future_retains_resources() {
    std::weak_ptr<const std::vector<double>> resource_lifetime;
    std::future<double> pending;
    {
        auto resources = std::make_shared<const std::vector<double>>(
            std::vector<double>{12.0});
        resource_lifetime = resources;
        auto model = std::make_shared<DeferredModel>(resources);
        auto cell = cellerator::make_cell(model, FutureInput{5.0});
        pending = cell.forward();
    }
    require(!resource_lifetime.expired(),
            "future result should retain resources after its Cell and model expire");
    require(pending.get() == 17.0, "future returned an incorrect result");
    std::cout << "future result: resource-retained-until-get=true value=17\n";
}

} // namespace

int main() {
    try {
        shared_model_with_borrowed_rows();
        copy_and_move_policy();
        joint_operation();
        owner_relocation_and_separate_parts();
        pooled_batch_direct();
        caller_cost_host_choice();
        model_owned_revision_cache();
        future_retains_resources();
    } catch (const std::exception& error) {
        std::cerr << "native Cell foundation example failed: " << error.what() << '\n';
        return 1;
    }
}
