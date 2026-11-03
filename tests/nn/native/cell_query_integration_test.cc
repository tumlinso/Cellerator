#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cell.hh>
#include <Cellerator/nn/query.hh>

#include <cassert>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace nn = cellerator::nn;

struct InputView { int value; };

struct QueryInstance {
    int information;
    std::string identity;

    InputView read(cellerator::ViewTag<InputView>) const & {
        return {information};
    }
    cellerator::CellIdentityView<std::string>
    read(cellerator::ViewTag<cellerator::CellIdentityView<std::string>>) const & {
        return {identity};
    }
};

struct BaseModel {
    int* calls;
    int operator()(const InputView& input, nn::Run run) const {
        ++*calls;
        return input.value + (run.training ? 100 : 0);
    }
};

struct Binding {
    std::vector<int> ids;
    const std::vector<int>& canonical_ids() const noexcept { return ids; }
};

struct Targets {
    std::vector<int> requested;
    std::vector<int> sources;
    std::vector<std::int64_t> columns;
    const std::vector<int>& canonical_ids() const noexcept { return requested; }
    const std::vector<int>& source_ids() const noexcept { return sources; }
    const std::vector<std::int64_t>& output_columns() const noexcept { return columns; }
    const std::vector<std::int64_t>& model_indices() const noexcept { return columns; }
};

struct Coordinates { int row; };

struct Decoder {
    int query(const int& source, const Targets& targets,
              const Coordinates& coordinates, nn::Run run) const {
        return source + targets.columns.front() + coordinates.row +
            (run.training ? 1000 : 0);
    }
};

int main() {
    int calls = 0;
    BaseModel model{&calls};
    auto cell = cellerator::make_cell(
        model, QueryInstance{5, "query-cell"});

    Binding binding{{10, 20}};
    Decoder decoder;
    auto selector = [](const int& result, const InputView& input) {
        return result * 2 + input.value;
    };
    auto readout = nn::readout(decoder, selector, binding);
    Targets first_targets{{20}, {10, 20}, {1}};
    Targets second_targets{{10}, {10, 20}, {0}};
    Coordinates first_coordinates{4};
    Coordinates second_coordinates{7};
    auto first_request = nn::query(readout, first_targets, first_coordinates);
    auto second_request = nn::query(readout, second_targets, second_coordinates);

    // The binding exposes a model-specific typed input. Batched NN readouts
    // stay ordinary operations and share one base evaluation.
    const auto input = cell.view<InputView>();
    auto both = nn::predict(cell.model(), input, first_request, second_request);
    static_assert(std::tuple_size_v<decltype(both)> == 2);
    assert(calls == 1);
    assert(std::get<0>(both).values == 20);
    assert(std::get<1>(both).values == 22);

    // Reacquiring the typed view observes the live model-specific instance.
    cell.instance().information = 8;
    const auto changed_input = cell.view<InputView>();
    auto selected = nn::predict_query(
        cell.model(), changed_input, first_request);
    assert(calls == 2 && selected.values == 29);

    auto training = nn::predict(
        cell.model(), changed_input, nn::Run{true}, first_request);
    assert(calls == 3 && training.values == 1229);
    assert(cell.view<cellerator::CellIdentityView<std::string>>().value == "query-cell");
}
