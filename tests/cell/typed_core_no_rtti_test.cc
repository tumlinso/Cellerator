#include <Cellerator/cell.hh>
#include <cassert>

struct ScalarModel { int operator()(int x) const { return x * 2; } };
int main() {
    ScalarModel model;
    auto cell = cellerator::make_cell(model, 21);
    assert(cell.forward() == 42);
}
