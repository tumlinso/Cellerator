#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cell.hh>

#include <cassert>
#include <string>
#include <type_traits>
#include <utility>

namespace {

struct TokenView {
    const int* data = nullptr;
    std::size_t count = 0;
};

struct ExpressionView {
    const float* data = nullptr;
    std::size_t count = 0;
};

struct MutableExpressionView {
    float* data = nullptr;
    std::size_t count = 0;
};

struct CustomView {
    const char* data = nullptr;
};

struct MissingView {};

class MultiModalCell final : public cellerator::CellAccess<TokenView>,
                             public cellerator::CellAccess<ExpressionView>,
                             public cellerator::CellEdit<MutableExpressionView>,
                             public cellerator::CellAccess<CustomView> {
public:
    explicit MultiModalCell(std::string identity)
        : identity_(std::move(identity)) {}

protected:
    TokenView read(cellerator::ViewTag<TokenView>) const & override {
        return {tokens_, 3};
    }

    ExpressionView read(cellerator::ViewTag<ExpressionView>) const & override {
        return {expression_, 2};
    }

    MutableExpressionView write(
        cellerator::ViewTag<MutableExpressionView>) & override {
        return {expression_, 2};
    }

    CustomView read(cellerator::ViewTag<CustomView>) const & override {
        return {custom_};
    }

private:
    std::string identity_;
    int tokens_[3]{4, 4, 9};
    float expression_[2]{1.0f, 2.0f};
    const char* custom_ = "extended";
};

class StringIdentityCell final
    : public cellerator::CellAccess<cellerator::CellIdentityView<std::string>> {
public:
    explicit StringIdentityCell(std::string value) : value_(std::move(value)) {}

protected:
    cellerator::CellIdentityView<std::string> read(
        cellerator::ViewTag<cellerator::CellIdentityView<std::string>>) const & override {
        return {value_};
    }

private:
    std::string value_;
};

class IntegerIdentityCell final
    : public cellerator::CellAccess<cellerator::CellIdentityView<int>> {
public:
    explicit IntegerIdentityCell(int value) : value_(value) {}

protected:
    cellerator::CellIdentityView<int> read(
        cellerator::ViewTag<cellerator::CellIdentityView<int>>) const & override {
        return {value_};
    }

private:
    int value_;
};

class EmptyCell final : public cellerator::Cell {
};

template<class T, class = void>
struct HasRvalueView : std::false_type {};

template<class T>
struct HasRvalueView<T, std::void_t<decltype(
    std::declval<T&&>().template view<TokenView>())>> : std::true_type {};

template<class T, class = void>
struct HasConstEdit : std::false_type {};

template<class T>
struct HasConstEdit<T, std::void_t<decltype(
    std::declval<const T&>().template edit<MutableExpressionView>())>>
    : std::true_type {};

template<class T, class = void>
struct HasRvalueEdit : std::false_type {};

template<class T>
struct HasRvalueEdit<T, std::void_t<decltype(
    std::declval<T&&>().template edit<MutableExpressionView>())>>
    : std::true_type {};

template<class T, class View, class = void>
struct HasPublicRead : std::false_type {};

template<class T, class View>
struct HasPublicRead<T, View, std::void_t<decltype(
    std::declval<const T&>().read(cellerator::ViewTag<View>{}))>> : std::true_type {};

template<class T, class View, class = void>
struct HasPublicWrite : std::false_type {};

template<class T, class View>
struct HasPublicWrite<T, View, std::void_t<decltype(
    std::declval<T&>().write(cellerator::ViewTag<View>{}))>> : std::true_type {};

} // namespace

int main() {
    using namespace cellerator;
    static_assert(std::is_abstract<Cell>::value,
        "Cell must remain an interface");
    static_assert(!HasRvalueView<MultiModalCell>::value,
        "views must not be acquired from temporary cells");
    static_assert(!HasConstEdit<MultiModalCell>::value,
        "const cells must not expose edit access");
    static_assert(!HasRvalueEdit<MultiModalCell>::value,
        "edits must not be acquired from temporary cells");
    static_assert(!HasPublicRead<CellAccess<TokenView>, TokenView>::value,
        "read hook must be protected on the access contract");
    static_assert(!HasPublicWrite<CellEdit<MutableExpressionView>, MutableExpressionView>::value,
        "write hook must be protected on the edit contract");
    static_assert(!HasPublicRead<MultiModalCell, TokenView>::value,
        "concrete cell must not expose the read hook publicly");
    static_assert(!HasPublicWrite<MultiModalCell, MutableExpressionView>::value,
        "concrete cell must not expose the write hook publicly");

    MultiModalCell cell("cell-A");
    const MultiModalCell& const_cell = cell;

    assert(cell.supports<TokenView>());
    assert(cell.supports<ExpressionView>());
    assert(cell.supports<CustomView>());
    assert(!cell.supports<MutableExpressionView>());
    assert(cell.supports_edit<MutableExpressionView>());
    assert(!cell.supports_edit<ExpressionView>());
    assert(const_cell.supports<TokenView>());

    const TokenView tokens = const_cell.view<TokenView>();
    assert(tokens.count == 3);
    assert(tokens.data[0] == 4 && tokens.data[1] == 4 && tokens.data[2] == 9);

    const ExpressionView read = const_cell.view<ExpressionView>();
    assert(read.count == 2 && read.data[0] == 1.0f);
    const CustomView extension = cell.view<CustomView>();
    assert(std::string(extension.data) == "extended");

    const CellAccess<TokenView>& token_access = cell;
    assert(token_access.view<TokenView>().count == tokens.count);

    MutableExpressionView edit = cell.edit<MutableExpressionView>();
    edit.data[1] = 17.0f;
    assert(const_cell.view<ExpressionView>().data[1] == 17.0f);
    CellEdit<MutableExpressionView>& edit_access = cell;
    assert(edit_access.edit<MutableExpressionView>().data == edit.data);

    bool missing_read_threw = false;
    try {
        (void)cell.view<MissingView>();
    } catch (const std::bad_cast&) {
        missing_read_threw = true;
    }
    assert(missing_read_threw);

    bool missing_edit_threw = false;
    try {
        (void)cell.edit<ExpressionView>();
    } catch (const std::bad_cast&) {
        missing_edit_threw = true;
    }
    assert(missing_edit_threw);

    StringIdentityCell string_cell("bio:cell/42");
    const auto string_identity = string_cell.view<CellIdentityView<std::string>>();
    assert(string_identity.value == "bio:cell/42");

    IntegerIdentityCell integer_cell(8675309);
    const auto integer_identity = integer_cell.view<CellIdentityView<int>>();
    assert(integer_identity.value == 8675309);

    EmptyCell empty;
    assert(!empty.supports<TokenView>());
    assert(!empty.supports_edit<MutableExpressionView>());
    bool empty_lookup_threw = false;
    try {
        (void)empty.view<TokenView>();
    } catch (const std::bad_cast&) {
        empty_lookup_threw = true;
    }
    assert(empty_lookup_threw);
}
