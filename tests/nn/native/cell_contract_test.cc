#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cell.hh>

#include <cassert>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

using cellerator::Cell;
using cellerator::CellAccess;
using cellerator::CellEdit;
using cellerator::CellIdentityView;
using cellerator::ViewTag;

struct NumberView { const int& value; };
struct MutableNumberView { int& value; };
struct CallerView { const std::string& value; };
struct UnsupportedView {};

template<class Identity>
class ExampleCell final
    : public CellAccess<NumberView>,
      public CellEdit<MutableNumberView>,
      public CellAccess<CallerView>,
      public CellAccess<CellIdentityView<Identity>> {
public:
    ExampleCell(int number, Identity identity, std::string text)
        : number_(number), identity_(std::move(identity)), text_(std::move(text)) {}

protected:
    NumberView read(ViewTag<NumberView>) const & override { return {number_}; }
    MutableNumberView write(ViewTag<MutableNumberView>) & override { return {number_}; }
    CallerView read(ViewTag<CallerView>) const & override { return {text_}; }
    CellIdentityView<Identity> read(ViewTag<CellIdentityView<Identity>>) const & override {
        return {identity_};
    }

private:
    int number_;
    Identity identity_;
    std::string text_;
};

template<class T, class View>
concept CanViewTemporary = requires(T&& cell) {
    static_cast<T&&>(cell).template view<View>();
};

template<class T, class View>
concept CanEditConst = requires(const T& cell) {
    cell.template edit<View>();
};

template<class T, class View>
concept CanEditTemporary = requires(T&& cell) {
    static_cast<T&&>(cell).template edit<View>();
};

int main() {
    ExampleCell<std::string> cell(3, "cell-A", "custom application view");
    static_assert(std::is_base_of_v<Cell, ExampleCell<std::string>>);
    static_assert(!CanViewTemporary<ExampleCell<std::string>, NumberView>);
    static_assert(!CanEditConst<ExampleCell<std::string>, MutableNumberView>);
    static_assert(!CanEditTemporary<ExampleCell<std::string>, MutableNumberView>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(
        std::declval<NumberView>().value)>>);

    const Cell& read_only = cell;
    assert(read_only.supports<NumberView>());
    assert(read_only.supports<CallerView>());
    assert(read_only.supports<CellIdentityView<std::string>>());
    assert(!read_only.supports<UnsupportedView>());
    assert(read_only.view<NumberView>().value == 3);
    assert(read_only.view<CallerView>().value == "custom application view");
    assert(read_only.view<CellIdentityView<std::string>>().value == "cell-A");
    assert(!read_only.supports_edit<UnsupportedView>());
    assert(read_only.supports_edit<MutableNumberView>());

    bool bad_cast = false;
    try {
        (void)read_only.view<UnsupportedView>();
    } catch (const std::bad_cast&) {
        bad_cast = true;
    }
    assert(bad_cast);

    bad_cast = false;
    try {
        (void)cell.edit<UnsupportedView>();
    } catch (const std::bad_cast&) {
        bad_cast = true;
    }
    assert(bad_cast);

    auto mutable_view = cell.edit<MutableNumberView>();
    mutable_view.value = 11;
    assert(cell.view<NumberView>().value == 11);

    // Applications add independent typed access without changing Cell.
    ExampleCell<int> integer_identity(17, 90210, "independent capability");
    assert(integer_identity.view<CellIdentityView<int>>().value == 90210);
    assert(integer_identity.view<CallerView>().value == "independent capability");
}
