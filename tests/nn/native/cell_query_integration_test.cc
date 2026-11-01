#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cell.hh>
#include <Cellerator/nn/query.hh>

#include <cassert>
#include <string>
#include <tuple>
#include <vector>

namespace nn = cellerator::nn;

struct InputView { const int& value; };

class QueryCell final
    : public cellerator::CellAccess<InputView>,
      public cellerator::CellAccess<cellerator::CellIdentityView<std::string>> {
public:
    QueryCell(int& information, std::string identity)
        : information_(information), identity_(std::move(identity)) {}

protected:
    InputView read(cellerator::ViewTag<InputView>) const & override {
        return {information_};
    }
    cellerator::CellIdentityView<std::string> read(
        cellerator::ViewTag<cellerator::CellIdentityView<std::string>>) const & override {
        return {identity_};
    }

private:
    int& information_;
    std::string identity_;
};

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
    QueryCell cell(information, "query-cell");
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

    // Cell supplies a typed input view; the operation owns evaluation and
    // shares one base forward across both readout requests.
    const auto& input = cell.view<InputView>().value;
    auto both = nn::predict(model, input, first_request, second_request);
    static_assert(std::tuple_size_v<decltype(both)> == 2);
    // One base evaluation is shared across requests; source=(5*2+5)=15.
    assert(calls == 1);
    assert(std::get<0>(both).values == 20);
    assert(std::get<1>(both).values == 22);

    // The second operation call re-reads the cell's live view.
    information = 8;
    const auto& changed_input = cell.view<InputView>().value;
    auto selected = nn::predict_query(model, changed_input, first_request);
    assert(calls == 2 && selected.values == 29);

    auto training = nn::predict(model, changed_input, nn::Run{true}, first_request);
    assert(calls == 3 && training.values == 1229);
    assert(cell.view<cellerator::CellIdentityView<std::string>>().value == "query-cell");
}
