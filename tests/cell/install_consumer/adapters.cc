#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cell.hh>
#include <Cellerator/cells/event.hh>
#include <Cellerator/cells/expression.hh>
#include <Cellerator/memory/view.hh>

#include <cassert>
#include <span>
#include <string>

struct AdapterModel {};

int main() {
    namespace ce = cellerator;
    namespace cells = ce::cells;
    AdapterModel model;

    int tokens[] = {3, 3, 7};
    const ce::memory::const_array_view<int> buffer{tokens, 3, {}};
    auto events = ce::make_cell(
        model, cells::EventPrefix<int, std::string>(buffer, 0, 2, "cell-A"));
    auto prefix = events.view<cells::EventView<int>>();
    assert(prefix.tokens.data == tokens && prefix.tokens.count == 2);
    assert(events.view<ce::CellIdentityView<std::string>>().value == "cell-A");

    float values[] = {1.0f, 2.0f, -99.0f, 3.0f, 5.0f, -99.0f};
    int genes[] = {101, 205};
    ce::memory::matrix_view<float> matrix{values, 2, 2, 3, {}};
    auto expression = ce::make_cell(model,
        cells::MutableDenseExpressionRow<float, int, int>(
            matrix, std::span<const int>{genes}, 1, 42));
    auto row = expression.view<cells::ExpressionView<float, int>>();
    assert(row.values.data == values + 3 && row.values.count == 2);
    assert(row.feature_ids[1] == 205);
    auto edit = expression.edit<cells::MutableExpressionView<float, int>>();
    edit.values.data[1] = 8.0f;
    assert(values[4] == 8.0f);
    assert(expression.view<ce::CellIdentityView<int>>().value == 42);

    ce::memory::const_matrix_view<float> read_only_matrix{
        values, 2, 2, 3, {}};
    auto read_only = ce::make_cell(model,
        cells::DenseExpressionRow<float, int, int>(
            read_only_matrix, std::span<const int>{genes}, 0, 41));
    auto first_row = read_only.view<cells::ExpressionView<float, int>>();
    assert(first_row.values.data == values && first_row.values.count == 2);
    assert(read_only.view<ce::CellIdentityView<int>>().value == 41);
}
