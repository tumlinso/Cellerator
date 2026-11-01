#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cells/event.hh>
#include <Cellerator/cells/expression.hh>

#include <cassert>
#include <span>
#include <string>

int main() {
    namespace ce = cellerator;
    namespace cells = ce::cells;
    int tokens[] = {3, 3, 7};
    const ce::memory::const_array_view<int> buffer{tokens, 3, {}};
    cells::EventCell<int, std::string> events(buffer, 0, 2, "cell-A");
    const ce::Cell& event_cell = events;
    auto prefix = event_cell.view<cells::EventView<int>>();
    assert(prefix.tokens.data == tokens && prefix.tokens.count == 2);
    assert(event_cell.view<ce::CellIdentityView<std::string>>().value == "cell-A");

    float values[] = {1.0f, 2.0f, -99.0f, 3.0f, 5.0f, -99.0f};
    int genes[] = {101, 205};
    ce::memory::matrix_view<float> matrix{values, 2, 2, 3, {}};
    cells::MutableDenseExpressionCell<float, int, int> expression(
        matrix, std::span<const int>{genes}, 1, 42);
    ce::Cell& cell = expression;
    auto row = cell.view<cells::ExpressionView<float, int>>();
    assert(row.values.data == values + 3 && row.values.count == 2);
    assert(row.feature_ids[1] == 205);
    auto edit = cell.edit<cells::MutableExpressionView<float, int>>();
    edit.values.data[1] = 8.0f;
    assert(values[4] == 8.0f);
}
