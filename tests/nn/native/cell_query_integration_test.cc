#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/cell.hh>
#include <Cellerator/nn/query.hh>

#include <cassert>
#include <string>
#include <tuple>
#include <vector>

namespace m = cellerator::modeling;
namespace nn = cellerator::nn;

struct BaseModel {
    int* calls;
    int operator()(const int& input, nn::Run run) const {
        ++*calls;
        return input + (run.training ? 100 : 0);
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
        return source + targets.columns.front() + coordinates.row + (run.training ? 1000 : 0);
    }
};

int main() {
    int information = 5;
    auto cell = m::make_cell(std::ref(information), std::string("query-cell"));
    int calls = 0;
    BaseModel model{&calls};
    Binding binding{{10, 20}};
    Decoder decoder;
    auto selector = [](const int& result, const int& input) { return result * 2 + input; };
    auto readout = nn::readout(decoder, selector, binding);
    Targets first_targets{{20}, {10, 20}, {1}};
    Targets second_targets{{10}, {10, 20}, {0}};
    Coordinates first_coordinates{4};
    Coordinates second_coordinates{7};
    auto first_request = nn::query(readout, first_targets, first_coordinates);
    auto second_request = nn::query(readout, second_targets, second_coordinates);

    auto bound = cell.bind(model);
    auto both = bound.predict(first_request, second_request);
    static_assert(std::tuple_size_v<decltype(both)> == 2);
    // One base evaluation is shared across requests; source=(5*2+5)=15.
    assert(calls == 1);
    assert(std::get<0>(both).values == 20);
    assert(std::get<1>(both).values == 22);

    // The query spelling uses the same ADL evaluation hook, and the borrowed
    // information source is read again for each invocation.
    information = 8;
    auto selected = bound.query(first_request);
    assert(calls == 2 && selected.values == 29);

    auto training = bound.predict(nn::Run{true}, first_request);
    assert(calls == 3 && training.values == 1229);
}
