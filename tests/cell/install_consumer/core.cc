#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cell.hh>
#include <Cellerator/cells/access.hh>
#include <Cellerator/modeling/cell.hh>

#include <cassert>
#include <string>
#include <type_traits>

struct ValueView { const int& value; };
struct ValueEdit { int& value; };

struct Instance {
    int value;
    std::string identity;

    ValueView read(cellerator::ViewTag<ValueView>) const & { return {value}; }
    ValueEdit write(cellerator::ViewTag<ValueEdit>) & { return {value}; }
    cellerator::CellIdentityView<std::string>
    read(cellerator::ViewTag<cellerator::CellIdentityView<std::string>>) const & {
        return {identity};
    }
};

struct Model {
    int operator()(Instance& instance, int delta) const {
        instance.value += delta;
        return instance.value;
    }
};

int main() {
    static_assert(!std::is_base_of_v<cellerator::CellViews,
                                     cellerator::Cell<Model, Instance>>);
    Model model;
    auto cell = cellerator::make_cell(model, Instance{17, "installed-cell"});
    assert((cellerator::Cell<Model, Instance>::supports<ValueView>()));
    assert((cellerator::Cell<Model, Instance>::supports_edit<ValueEdit>()));
    assert(cell.view<ValueView>().value == 17);
    assert(cell.view<cellerator::CellIdentityView<std::string>>().value ==
           "installed-cell");
    assert(cell.forward(5) == 22);

    auto runtime = cellerator::make_cell_views<
        cellerator::CellAccess<ValueView>, cellerator::CellEdit<ValueEdit>>(cell);
    cellerator::CellViews& runtime_views = runtime;
    auto edit = runtime_views.edit<ValueEdit>();
    edit.value = 31;
    assert(cell.view<ValueView>().value == 31);
}
