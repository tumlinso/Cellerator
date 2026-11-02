#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cell.hh>

#include <cassert>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {
struct View { const int* value; };
struct Edit { int* value; };
struct Missing {};
struct Instance {
    int value;
    std::string id;
    View read(cellerator::ViewTag<View>) const & { return {&value}; }
    cellerator::CellIdentityView<std::string> read(
        cellerator::ViewTag<cellerator::CellIdentityView<std::string>>) const & { return {id}; }
    Edit write(cellerator::ViewTag<Edit>) & { return {&value}; }
};
struct WrongView {
    View& read(cellerator::ViewTag<View>) const &;
};
struct MutableModel {
    int& operator()(Instance& i) { return i.value; }
    std::unique_ptr<int> operator()(Instance& i, int delta) { return std::make_unique<int>(i.value + delta); }
    void operator()(Instance& i, bool) { ++i.value; }
    int operator()(Instance&, const char*) { throw std::runtime_error("model failure"); }
    std::future<int> operator()(Instance& i, double) { return std::async(std::launch::deferred, [n=i.value] { return n; }); }
};
struct ConstModel {
    int operator()(const Instance& i) const { return i.value; }
};
struct AddressModel {
    AddressModel* operator&() = delete;
    int operator()(const Instance& i) const { return i.value; }
};
struct NonCallable {};
struct MoveOnly {
    std::unique_ptr<int> value = std::make_unique<int>(1);
};
struct CopyOnly {
    CopyOnly() = default;
    CopyOnly(const CopyOnly&) = default;
    CopyOnly& operator=(const CopyOnly&) = default;
    CopyOnly(CopyOnly&&) = delete;
    CopyOnly& operator=(CopyOnly&&) = delete;
};

template<class T, class = void> struct HasConstForward : std::false_type {};
template<class T> struct HasConstForward<T, std::void_t<decltype(std::declval<const T&>().forward())>> : std::true_type {};
template<class T, class = void> struct HasRvalueForward : std::false_type {};
template<class T> struct HasRvalueForward<T, std::void_t<decltype(std::declval<T&&>().forward())>> : std::true_type {};
template<class T, class = void> struct HasRvalueModel : std::false_type {};
template<class T> struct HasRvalueModel<T, std::void_t<decltype(std::declval<T&&>().model())>> : std::true_type {};
template<class T, class = void> struct HasRvalueInstance : std::false_type {};
template<class T> struct HasRvalueInstance<T, std::void_t<decltype(std::declval<T&&>().instance())>> : std::true_type {};
template<class T, class V, class = void> struct HasView : std::false_type {};
template<class T, class V> struct HasView<T, V, std::void_t<decltype(std::declval<T>().template view<V>())>> : std::true_type {};
template<class T, class V, class = void> struct HasEdit : std::false_type {};
template<class T, class V> struct HasEdit<T, V, std::void_t<decltype(std::declval<T>().template edit<V>())>> : std::true_type {};
template<class T, class = void> struct CanBorrowTemporary : std::false_type {};
template<class T> struct CanBorrowTemporary<T, std::void_t<decltype(cellerator::make_cell(std::declval<T&&>(), 0))>> : std::true_type {};
} // namespace

int main() {
    using namespace cellerator;
    using C = Cell<MutableModel, Instance>;
    static_assert(C::supports<View>() && C::supports<CellIdentityView<std::string>>());
    static_assert(C::supports_edit<Edit>() && !C::supports<Missing>() && !C::supports_edit<View>());
    static_assert(!Cell<NonCallable, WrongView>::supports<View>());
    static_assert(HasView<C&, View>::value && HasView<const C&, View>::value);
    static_assert(!HasView<C&&, View>::value && !HasView<C&, Missing>::value);
    static_assert(HasEdit<C&, Edit>::value && !HasEdit<const C&, Edit>::value && !HasEdit<C&&, Edit>::value);
    static_assert(!HasRvalueModel<C>::value && !HasRvalueInstance<C>::value && !HasRvalueForward<C>::value);
    static_assert(!HasConstForward<C>::value && HasConstForward<Cell<ConstModel, Instance>>::value);
    static_assert(!CanBorrowTemporary<MutableModel>::value);
    static_assert(!CanBorrowTemporary<const MutableModel>::value);
    static_assert(std::is_copy_constructible_v<C> && std::is_move_constructible_v<C>);
    static_assert(!std::is_copy_constructible_v<Cell<MutableModel, MoveOnly>> && std::is_move_constructible_v<Cell<MutableModel, MoveOnly>>);
    static_assert(std::is_copy_constructible_v<Cell<MutableModel, CopyOnly>> == std::is_copy_constructible_v<CopyOnly>);
    static_assert(std::is_move_constructible_v<Cell<MutableModel, CopyOnly>> == std::is_move_constructible_v<CopyOnly>);
    static_assert(std::is_copy_assignable_v<Cell<MutableModel, CopyOnly>> == std::is_copy_assignable_v<CopyOnly>);
    static_assert(std::is_move_assignable_v<Cell<MutableModel, CopyOnly>> == std::is_move_assignable_v<CopyOnly>);
    static_assert(!std::is_copy_assignable_v<Cell<MutableModel, MoveOnly>> && std::is_move_assignable_v<Cell<MutableModel, MoveOnly>>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().model()), const MutableModel&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().instance()), const Instance&>);

    MutableModel model;
    AddressModel address_model;
    auto address_cell = make_cell(address_model, Instance{4, "address"});
    assert(std::addressof(address_cell.model()) == std::addressof(address_model));
    assert(address_cell.forward() == 4);
    try { (void)C(nullptr, Instance{1, "null"}); assert(false); }
    catch (const std::invalid_argument&) {}
    C directly_bound(&model, Instance{2, "direct"});
    assert(&directly_bound.model() == &model);
    auto cell = make_cell(model, Instance{3, "bio:42"});
    auto copy = cell;
    assert(&copy.model() == &model && &copy.instance() != &cell.instance());
    cell.forward() = 7;
    assert(cell.instance().value == 7 && copy.instance().value == 3);
    assert(*cell.forward(4) == 11);
    cell.forward(true);
    assert(cell.forward(0.0).get() == 8);
    try { cell.forward("fail"); assert(false); } catch (const std::runtime_error&) {}
    assert(cell.instance().value == 8);
    auto edit = cell.edit<Edit>(); *edit.value = 9;
    assert(*cell.view<View>().value == 9);
    assert(cell.view<CellIdentityView<std::string>>().value == "bio:42");
    const ConstModel const_model;
    auto const_binding = make_cell(const_model, Instance{12, "id"});
    static_assert(std::is_same_v<decltype(const_binding.model()), const ConstModel&>);
    assert(std::as_const(const_binding).forward() == 12);
    auto owned = std::make_shared<ConstModel>();
    auto owned_cell = make_cell(owned, Instance{5, "owned"});
    assert(owned.use_count() == 2);
    owned.reset();
    assert(owned_cell.forward() == 5);
    auto temporary_owner = make_cell(std::make_shared<ConstModel>(), Instance{6, "temporary owner"});
    assert(temporary_owner.forward() == 6);
    try { (void)make_cell(std::shared_ptr<ConstModel>{}, Instance{1, "null"}); assert(false); }
    catch (const std::invalid_argument&) {}
    CopyOnly copy_instance;
    auto copy_only = make_cell(model, copy_instance);
    (void)copy_only;
}
