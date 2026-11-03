#ifdef NDEBUG
#undef NDEBUG
#endif
#include <ATen/ATen.h>
#include <Cellerator/cell.hh>
#include <Cellerator/cells/access.hh>

#include <cassert>
#include <memory>
#include <stdexcept>
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

struct BasicInstance {
    int number;
    std::string identity;
    std::string caller_text;

    NumberView read(ViewTag<NumberView>) const & { return {number}; }
    CallerView read(ViewTag<CallerView>) const & { return {caller_text}; }
    CellIdentityView<std::string>
    read(ViewTag<CellIdentityView<std::string>>) const & { return {identity}; }
    MutableNumberView write(ViewTag<MutableNumberView>) & { return {number}; }
};

struct BasicModel {
    explicit BasicModel(int& copies) : copies(&copies) {}
    BasicModel(const BasicModel& other) : copies(other.copies), calls(other.calls) {
        ++*copies;
    }
    BasicModel(BasicModel&&) noexcept = default;

    int operator()(BasicInstance& instance, int increment) {
        ++calls;
        instance.number += increment;
        return instance.number;
    }
    int operator()(const BasicInstance& instance, int increment) const {
        return instance.number + increment;
    }

    int* copies;
    int calls = 0;
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

template<class Model, class Instance>
concept CanBorrowTemporaryModel = requires(Model&& model, Instance&& instance) {
    cellerator::make_cell(static_cast<Model&&>(model),
                          static_cast<Instance&&>(instance));
};

struct RecurrentStateView { at::Tensor value; };

struct RecurrentInstance {
    at::Tensor value;

    RecurrentStateView read(ViewTag<RecurrentStateView>) const & {
        return {value};
    }
};

struct RecurrentModel {
    at::Tensor weight;
    int* calls;

    at::Tensor operator()(RecurrentInstance& state,
                          const at::Tensor& input) const {
        ++*calls;
        // Replacing this handle keeps prior autograd values alive through the
        // returned tensors and the graph, while the same Cell keeps advancing.
        state.value = state.value * weight + input;
        return state.value;
    }
};

int main() {
    int model_copies = 0;
    BasicModel model{model_copies};
    auto cell = cellerator::make_cell(
        model, BasicInstance{3, "cell-A", "caller-defined view"});
    using CellType = decltype(cell);

    static_assert(CellType::supports<NumberView>());
    static_assert(CellType::supports<CallerView>());
    static_assert(CellType::supports<CellIdentityView<std::string>>());
    static_assert(CellType::supports_edit<MutableNumberView>());
    static_assert(!CellType::supports<UnsupportedView>());
    static_assert(!CellType::supports_edit<UnsupportedView>());
    static_assert(!CanViewTemporary<CellType, NumberView>);
    static_assert(!CanEditConst<CellType, MutableNumberView>);
    static_assert(!CanEditTemporary<CellType, MutableNumberView>);
    static_assert(!CanBorrowTemporaryModel<BasicModel, BasicInstance>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(
        std::declval<NumberView>().value)>>);

    // A borrowed binding stores the model handle; it does not copy the model.
    assert(model_copies == 0);
    assert(cell.forward(4) == 7);
    assert(model.calls == 1);
    assert(cell.view<NumberView>().value == 7);
    assert(cell.view<CallerView>().value == "caller-defined view");
    assert(cell.view<CellIdentityView<std::string>>().value == "cell-A");

    // Cell copies copy their instance and keep the same borrowed model handle.
    auto copied_cell = cell;
    assert(model_copies == 0);
    assert(&copied_cell.model() == &model);
    assert(&copied_cell.instance() != &cell.instance());
    assert(copied_cell.forward(2) == 9);
    assert(cell.view<NumberView>().value == 7);
    assert(model.calls == 2);

    // A caller may explicitly copy its model and bind the copy.
    BasicModel explicitly_copied_model(model);
    assert(model_copies == 1);
    auto explicit_copy_cell = cellerator::make_cell(
        explicitly_copied_model, BasicInstance{17, "cell-B", "explicit copy"});
    assert(explicit_copy_cell.forward(1) == 18);
    assert(explicitly_copied_model.calls == 3);
    assert(model.calls == 2);

    const CellType& read_only = cell;
    assert(read_only.view<NumberView>().value == 7);
    assert(read_only.forward(5) == 12);
    auto mutable_view = cell.edit<MutableNumberView>();
    mutable_view.value = 11;
    assert(cell.view<NumberView>().value == 11);

    auto runtime = cellerator::make_cell_views<
        CellAccess<NumberView>, CellEdit<MutableNumberView>,
        CellAccess<CallerView>, CellAccess<CellIdentityView<std::string>>>(cell);
    cellerator::CellViews& runtime_views = runtime;
    assert(runtime_views.supports<NumberView>());
    assert(runtime_views.supports<CallerView>());
    assert(runtime_views.supports<CellIdentityView<std::string>>());
    assert(!runtime_views.supports<UnsupportedView>());
    assert(runtime_views.supports_edit<MutableNumberView>());
    assert(runtime_views.view<NumberView>().value == 11);
    assert(runtime_views.view<CallerView>().value == "caller-defined view");
    assert(runtime_views.view<CellIdentityView<std::string>>().value == "cell-A");

    auto runtime_edit = runtime_views.edit<MutableNumberView>();
    runtime_edit.value = 19;
    assert(cell.view<NumberView>().value == 19);

    bool bad_cast = false;
    try {
        (void)runtime_views.view<UnsupportedView>();
    } catch (const std::bad_cast&) {
        bad_cast = true;
    }
    assert(bad_cast);

    bad_cast = false;
    try {
        (void)runtime_views.edit<UnsupportedView>();
    } catch (const std::bad_cast&) {
        bad_cast = true;
    }
    assert(bad_cast);

    const auto& const_cell = cell;
    auto read_only_runtime = cellerator::make_cell_views<CellAccess<NumberView>>(
        const_cell);
    const cellerator::CellViews& const_views = read_only_runtime;
    assert(const_views.view<NumberView>().value == 19);
    assert(!const_views.supports_edit<MutableNumberView>());

    // Shared ownership is explicit and remains alive after the caller drops
    // its shared_ptr. Cell still stores only that owner and its instance.
    auto shared_model = std::make_shared<BasicModel>(model_copies);
    std::weak_ptr<BasicModel> weak_model = shared_model;
    auto shared_cell = cellerator::make_cell(
        shared_model, BasicInstance{1, "shared", "shared owner"});
    shared_model.reset();
    assert(!weak_model.expired());
    assert(shared_cell.forward(2) == 3);
    assert(weak_model.lock()->calls == 1);

    bool rejected_null_owner = false;
    try {
        std::shared_ptr<BasicModel> null_model;
        (void)cellerator::make_cell(
            null_model, BasicInstance{0, "null", "null owner"});
    } catch (const std::invalid_argument&) {
        rejected_null_owner = true;
    }
    assert(rejected_null_owner);

    // CPU ATen autograd: two calls mutate one logical Cell's instance by
    // replacing its Tensor handle. Earlier outputs remain valid for backward.
    auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
    auto initial = at::full({1}, 2.0, options).set_requires_grad(true);
    auto weight = at::full({1}, 3.0, options).set_requires_grad(true);
    const auto first_input = at::full({1}, 5.0, options);
    const auto second_input = at::full({1}, 7.0, options);
    int recurrent_calls = 0;
    RecurrentModel recurrent_model{weight, &recurrent_calls};
    auto recurrent = cellerator::make_cell(
        recurrent_model, RecurrentInstance{initial});
    auto* logical_instance = &recurrent.instance();

    auto first = recurrent.forward(first_input);
    const auto retained_state_after_first = recurrent.view<RecurrentStateView>().value;
    assert(first.item<double>() == 11.0);
    assert(retained_state_after_first.data_ptr() == first.data_ptr());
    assert(recurrent.instance().value.data_ptr() != initial.data_ptr());

    auto second = recurrent.forward(second_input);
    assert(&recurrent.instance() == logical_instance);
    assert(second.item<double>() == 40.0);
    assert(recurrent.view<RecurrentStateView>().value.data_ptr() == second.data_ptr());
    assert(second.data_ptr() != first.data_ptr());
    assert(first.item<double>() == 11.0);
    assert(retained_state_after_first.item<double>() == 11.0);
    assert(recurrent_calls == 2);

    (first + second).sum().backward();
    assert(weight.grad().defined());
    assert(weight.grad().item<double>() == 19.0);
    assert(initial.grad().defined());
    assert(initial.grad().item<double>() == 12.0);
}
