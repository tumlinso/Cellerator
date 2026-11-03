#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cellerator {

template<class View> struct ViewTag {};

template<class Identity> struct CellIdentityView {
    const Identity& value;
};

namespace detail {
template<class T> struct is_shared_ptr : std::false_type {};
template<class T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template<class Instance, class View, class = void>
struct can_read : std::false_type {};
template<class Instance, class View>
struct can_read<Instance, View, std::void_t<decltype(
    std::declval<const Instance&>().read(ViewTag<View>{}))>>
    : std::is_same<decltype(std::declval<const Instance&>().read(ViewTag<View>{})), View> {};

template<class Instance, class View, class = void>
struct can_write : std::false_type {};
template<class Instance, class View>
struct can_write<Instance, View, std::void_t<decltype(
    std::declval<Instance&>().write(ViewTag<View>{}))>>
    : std::is_same<decltype(std::declval<Instance&>().write(ViewTag<View>{})), View> {};
} // namespace detail

// The binding holds no state beyond the selected handle and instance. Owners
// determine their lifetimes, synchronization, and any model state or history.
namespace detail {
template<class Model, class Instance, class ModelHandle>
class cell_storage {
public:
    template<class H, class I,
             std::enable_if_t<std::is_constructible_v<ModelHandle, H&&> &&
                              std::is_constructible_v<Instance, I&&>, int> = 0>
    explicit cell_storage(H&& handle, I&& instance)
        : handle_(std::forward<H>(handle)), instance_(std::forward<I>(instance)) {
        if constexpr (std::is_pointer_v<ModelHandle> || detail::is_shared_ptr<ModelHandle>::value) {
            if (!handle_) throw std::invalid_argument("Cell requires a non-null model handle");
        }
    }

    Model& model() & { return *handle_; }
    const Model& model() const & { return *handle_; }
    Model& model() && = delete;
    const Model& model() const && = delete;
    Instance& instance() & { return instance_; }
    const Instance& instance() const & { return instance_; }
    Instance& instance() && = delete;
    const Instance& instance() const && = delete;

    template<class... Args, std::enable_if_t<
        std::is_invocable_v<Model&, Instance&, Args&&...>, int> = 0>
    decltype(auto) forward(Args&&... args) & {
        return std::invoke(model(), instance(), std::forward<Args>(args)...);
    }
    template<class... Args, std::enable_if_t<
        std::is_invocable_v<const Model&, const Instance&, Args&&...>, int> = 0>
    decltype(auto) forward(Args&&... args) const & {
        return std::invoke(model(), instance(), std::forward<Args>(args)...);
    }
    template<class... Args> void forward(Args&&...) && = delete;
    template<class... Args> void forward(Args&&...) const && = delete;

    template<class View> static constexpr bool supports() noexcept {
        return detail::can_read<Instance, View>::value;
    }
    template<class View> static constexpr bool supports_edit() noexcept {
        return detail::can_write<Instance, View>::value;
    }
    template<class View, std::enable_if_t<detail::can_read<Instance, View>::value, int> = 0>
    View view() const & { return instance_.read(ViewTag<View>{}); }
    template<class View> void view() const && = delete;
    template<class View, std::enable_if_t<detail::can_write<Instance, View>::value, int> = 0>
    View edit() & { return instance_.write(ViewTag<View>{}); }
    template<class View> void edit() && = delete;

private:
    ModelHandle handle_;
    Instance instance_;
};

// A deleted defaulted move is ignored by overload resolution, which would let
// a copy-only instance make Cell appear movable. Explicitly delete unavailable
// operations so all four traits track the stored handle and instance.
template<class Model, class Instance, class ModelHandle = Model*,
         bool CopyConstruct = std::is_copy_constructible_v<ModelHandle> && std::is_copy_constructible_v<Instance>,
         bool MoveConstruct = std::is_move_constructible_v<ModelHandle> && std::is_move_constructible_v<Instance>,
         bool CopyAssign = std::is_copy_assignable_v<ModelHandle> && std::is_copy_assignable_v<Instance>,
         bool MoveAssign = std::is_move_assignable_v<ModelHandle> && std::is_move_assignable_v<Instance>>
class cell_impl;

#define CELLERATOR_CELL_SPECIALIZATION(CC, MC, CA, MA, COPY_CTOR, MOVE_CTOR, COPY_ASSIGN, MOVE_ASSIGN) \
 template<class Model, class Instance, class ModelHandle> \
 class cell_impl<Model, Instance, ModelHandle, CC, MC, CA, MA> \
     : public detail::cell_storage<Model, Instance, ModelHandle> { \
     using base = detail::cell_storage<Model, Instance, ModelHandle>; \
 public: \
     using base::base; \
     cell_impl(const cell_impl&) = COPY_CTOR; \
     cell_impl(cell_impl&&) = MOVE_CTOR; \
     cell_impl& operator=(const cell_impl&) = COPY_ASSIGN; \
     cell_impl& operator=(cell_impl&&) = MOVE_ASSIGN; \
 };
CELLERATOR_CELL_SPECIALIZATION(false, false, false, false, delete, delete, delete, delete)
CELLERATOR_CELL_SPECIALIZATION(false, false, false, true, delete, delete, delete, default)
CELLERATOR_CELL_SPECIALIZATION(false, false, true, false, delete, delete, default, delete)
CELLERATOR_CELL_SPECIALIZATION(false, false, true, true, delete, delete, default, default)
CELLERATOR_CELL_SPECIALIZATION(false, true, false, false, delete, default, delete, delete)
CELLERATOR_CELL_SPECIALIZATION(false, true, false, true, delete, default, delete, default)
CELLERATOR_CELL_SPECIALIZATION(false, true, true, false, delete, default, default, delete)
CELLERATOR_CELL_SPECIALIZATION(false, true, true, true, delete, default, default, default)
CELLERATOR_CELL_SPECIALIZATION(true, false, false, false, default, delete, delete, delete)
CELLERATOR_CELL_SPECIALIZATION(true, false, false, true, default, delete, delete, default)
CELLERATOR_CELL_SPECIALIZATION(true, false, true, false, default, delete, default, delete)
CELLERATOR_CELL_SPECIALIZATION(true, false, true, true, default, delete, default, default)
CELLERATOR_CELL_SPECIALIZATION(true, true, false, false, default, default, delete, delete)
CELLERATOR_CELL_SPECIALIZATION(true, true, false, true, default, default, delete, default)
CELLERATOR_CELL_SPECIALIZATION(true, true, true, false, default, default, default, delete)
CELLERATOR_CELL_SPECIALIZATION(true, true, true, true, default, default, default, default)
#undef CELLERATOR_CELL_SPECIALIZATION
} // namespace detail

template<class Model, class Instance, class ModelHandle = Model*>
using Cell = detail::cell_impl<Model, Instance, ModelHandle>;

template<class Model, class Instance,
         std::enable_if_t<!detail::is_shared_ptr<std::remove_cv_t<Model>>::value, int> = 0>
auto make_cell(Model& model, Instance&& instance) {
    return Cell<Model, std::decay_t<Instance>>(
        std::addressof(model), std::forward<Instance>(instance));
}

// A deduced const Model& can bind an rvalue; borrowing it would dangle.
template<class Model, class Instance,
         std::enable_if_t<!std::is_lvalue_reference_v<Model> &&
                          !detail::is_shared_ptr<std::remove_cv_t<Model>>::value, int> = 0>
void make_cell(Model&&, Instance&&) = delete;

template<class Model, class Instance>
auto make_cell(std::shared_ptr<Model> model, Instance&& instance) {
    return Cell<Model, std::decay_t<Instance>, std::shared_ptr<Model>>(
        std::move(model), std::forward<Instance>(instance));
}

} // namespace cellerator
