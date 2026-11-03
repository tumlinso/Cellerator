#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cells/access.hh>
#include <cassert>
#include <string>
#include <type_traits>
#include <utility>

struct Read { const int* value; };
struct Write { int* value; };
struct Other {};
struct Model {};
struct Instance {
    int value;
    Read read(cellerator::ViewTag<Read>) const & { return {&value}; }
    Write write(cellerator::ViewTag<Write>) & { return {&value}; }
};
template<class T, class V, class = void> struct HasRead : std::false_type {};
template<class T, class V> struct HasRead<T,V,std::void_t<decltype(std::declval<const T&>().read(cellerator::ViewTag<V>{}))>> : std::true_type {};
template<class T, class V, class = void> struct HasWrite : std::false_type {};
template<class T, class V> struct HasWrite<T,V,std::void_t<decltype(std::declval<T&>().write(cellerator::ViewTag<V>{}))>> : std::true_type {};
template<class T, class = void> struct CanBridgeRvalue : std::false_type {};
template<class T> struct CanBridgeRvalue<T,std::void_t<decltype(cellerator::make_cell_views<cellerator::CellAccess<Read>>(std::declval<T&&>()))>> : std::true_type {};
int main() {
    using namespace cellerator;
    static_assert(!HasRead<CellAccess<Read>,Read>::value && !HasWrite<CellEdit<Write>,Write>::value);
    Model model;
    auto cell = make_cell(model, Instance{3});
    static_assert(!CanBridgeRvalue<decltype(cell)>::value);
    static_assert(!CanBridgeRvalue<const decltype(cell)>::value);
    auto bridge = make_cell_views<CellAccess<Read>, CellEdit<Write>>(cell);
    CellViews& views = bridge;
    assert(views.supports<Read>() && !views.supports<Other>());
    assert(views.supports_edit<Write>() && !views.supports_edit<Read>());
    assert(*views.view<Read>().value == 3);
    *views.edit<Write>().value = 9;
    assert(cell.instance().value == 9);
    try { (void)views.view<Other>(); assert(false); } catch (const std::bad_cast&) {}
    try { (void)views.edit<Other>(); assert(false); } catch (const std::bad_cast&) {}
    auto const_bridge = make_cell_views<CellAccess<Read>>(std::as_const(cell));
    assert(*const_bridge.view<Read>().value == 9 && !const_bridge.supports_edit<Write>());
    auto empty_bridge = make_cell_views<>(cell);
    assert(!empty_bridge.supports<Read>());
}
