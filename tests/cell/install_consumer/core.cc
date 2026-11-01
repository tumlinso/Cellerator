#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cell.hh>
#include <Cellerator/modeling/cell.hh>

#include <cassert>
#include <type_traits>

struct Example final : cellerator::CellAccess<int> {
protected:
    int read(cellerator::ViewTag<int>) const & override { return 17; }
};

int main() {
    static_assert(std::is_abstract_v<cellerator::Cell>);
    Example example;
    const cellerator::Cell& cell = example;
    assert(cell.supports<int>() && !cell.supports<double>());
    assert(cell.view<int>() == 17);
    try {
        (void)cell.view<double>();
        assert(false);
    } catch (const std::bad_cast&) {}
}
