#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace cellerator::modeling {

namespace detail {

template<class T>
struct is_reference_wrapper : std::false_type {};

template<class T>
struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type {};

template<class T>
inline constexpr bool is_reference_wrapper_v = is_reference_wrapper<std::decay_t<T>>::value;

template<class T>
struct is_shared_ptr : std::false_type {};

template<class T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template<class T>
inline constexpr bool is_shared_ptr_v = is_shared_ptr<std::decay_t<T>>::value;

// Information can be stored directly, borrowed through std::ref, shared, or
// supplied by a zero-argument provider. Resolve it only when a program runs so
// borrowed and provider-backed information remains live.
template<class T>
decltype(auto) resolve_information(T&& source) {
    using source_type = std::decay_t<T>;
    if constexpr (is_reference_wrapper_v<source_type>) {
        return resolve_information(source.get());
    } else if constexpr (is_shared_ptr_v<source_type>) {
        if (!source) throw std::invalid_argument("cell information source cannot be null");
        return resolve_information(*source);
    } else if constexpr (std::is_invocable_v<T>) {
        using result_type = std::invoke_result_t<T>;
        if constexpr (std::is_lvalue_reference_v<result_type>) {
            return resolve_information(std::invoke(std::forward<T>(source)));
        } else {
            return std::invoke(std::forward<T>(source));
        }
    } else {
        return std::forward<T>(source);
    }
}

// Program handles are unwrapped without invoking a zero-argument program.
template<class T>
decltype(auto) unwrap_program(T&& program) {
    using program_type = std::decay_t<T>;
    if constexpr (is_reference_wrapper_v<program_type>) {
        return program.get();
    } else if constexpr (is_shared_ptr_v<program_type>) {
        if (!program) throw std::invalid_argument("bound program cannot be null");
        return *program;
    } else {
        return std::forward<T>(program);
    }
}

namespace adl {
// The zero-argument fallback makes these dependent calls detectable while
// allowing associated namespaces to provide typed hooks such as nn::predict.
void predict() = delete;
void predict_query() = delete;
void query() = delete;

template<class Program, class Information, class... Args>
auto call_predict(Program&& program, Information&& information, Args&&... args)
    -> decltype(predict(std::forward<Program>(program),
                        std::forward<Information>(information),
                        std::forward<Args>(args)...)) {
    return predict(std::forward<Program>(program),
                   std::forward<Information>(information),
                   std::forward<Args>(args)...);
}

template<class Program, class Information, class... Args>
auto call_predict_query(Program&& program, Information&& information, Args&&... args)
    -> decltype(predict_query(std::forward<Program>(program),
                              std::forward<Information>(information),
                              std::forward<Args>(args)...)) {
    return predict_query(std::forward<Program>(program),
                         std::forward<Information>(information),
                         std::forward<Args>(args)...);
}

template<class Program, class Information, class... Args>
auto call_query(Program&& program, Information&& information, Args&&... args)
    -> decltype(query(std::forward<Program>(program),
                      std::forward<Information>(information),
                      std::forward<Args>(args)...)) {
    return query(std::forward<Program>(program),
                 std::forward<Information>(information),
                 std::forward<Args>(args)...);
}
} // namespace adl

template<class Program, class Information, class... Args>
using member_predict_result_t = decltype(std::declval<Program>().predict(
    std::declval<Information>(), std::declval<Args>()...));

template<class, class Program, class Information, class... Args>
struct has_member_predict_impl : std::false_type {};

template<class Program, class Information, class... Args>
struct has_member_predict_impl<std::void_t<member_predict_result_t<Program, Information, Args...>>,
                               Program, Information, Args...> : std::true_type {};

template<class Program, class Information, class... Args>
inline constexpr bool has_member_predict_v =
    has_member_predict_impl<void, Program, Information, Args...>::value;

template<class Program, class Information, class... Args>
using member_query_result_t = decltype(std::declval<Program>().query(
    std::declval<Information>(), std::declval<Args>()...));

template<class, class Program, class Information, class... Args>
struct has_member_query_impl : std::false_type {};

template<class Program, class Information, class... Args>
struct has_member_query_impl<std::void_t<member_query_result_t<Program, Information, Args...>>,
                             Program, Information, Args...> : std::true_type {};

template<class Program, class Information, class... Args>
inline constexpr bool has_member_query_v =
    has_member_query_impl<void, Program, Information, Args...>::value;

template<class Program, class Information, class... Args>
using adl_predict_result_t = decltype(adl::call_predict(
    std::declval<Program>(), std::declval<Information>(), std::declval<Args>()...));

template<class Program, class Information, class... Args>
using adl_predict_query_result_t = decltype(adl::call_predict_query(
    std::declval<Program>(), std::declval<Information>(), std::declval<Args>()...));

template<class, class Program, class Information, class... Args>
struct has_adl_predict_impl : std::false_type {};

template<class Program, class Information, class... Args>
struct has_adl_predict_impl<std::void_t<adl_predict_result_t<Program, Information, Args...>>,
                            Program, Information, Args...> : std::true_type {};

template<class Program, class Information, class... Args>
inline constexpr bool has_adl_predict_v =
    has_adl_predict_impl<void, Program, Information, Args...>::value;

template<class, class Program, class Information, class... Args>
struct has_adl_predict_query_impl : std::false_type {};

template<class Program, class Information, class... Args>
struct has_adl_predict_query_impl<
    std::void_t<adl_predict_query_result_t<Program, Information, Args...>>,
    Program, Information, Args...> : std::true_type {};

template<class Program, class Information, class... Args>
inline constexpr bool has_adl_predict_query_v =
    has_adl_predict_query_impl<void, Program, Information, Args...>::value;

template<class Program, class Information, class... Args>
using adl_query_result_t = decltype(adl::call_query(
    std::declval<Program>(), std::declval<Information>(), std::declval<Args>()...));

template<class, class Program, class Information, class... Args>
struct has_adl_query_impl : std::false_type {};

template<class Program, class Information, class... Args>
struct has_adl_query_impl<std::void_t<adl_query_result_t<Program, Information, Args...>>,
                          Program, Information, Args...> : std::true_type {};

template<class Program, class Information, class... Args>
inline constexpr bool has_adl_query_v =
    has_adl_query_impl<void, Program, Information, Args...>::value;

template<class Program, class Information, class... Args>
decltype(auto) invoke_prediction(Program&& program, Information&& information, Args&&... args) {
    auto&& callable = unwrap_program(std::forward<Program>(program));
    auto&& input = resolve_information(std::forward<Information>(information));
    if constexpr (has_adl_predict_query_v<decltype(callable), decltype(input), Args&&...>) {
        return adl::call_predict_query(std::forward<decltype(callable)>(callable),
                                       std::forward<decltype(input)>(input),
                                       std::forward<Args>(args)...);
    } else if constexpr (has_adl_predict_v<decltype(callable), decltype(input), Args&&...>) {
        return adl::call_predict(std::forward<decltype(callable)>(callable),
                                 std::forward<decltype(input)>(input),
                                 std::forward<Args>(args)...);
    } else if constexpr (has_member_predict_v<decltype(callable), decltype(input), Args&&...>) {
        return std::forward<decltype(callable)>(callable).predict(
            std::forward<decltype(input)>(input), std::forward<Args>(args)...);
    } else if constexpr (std::is_invocable_v<decltype(callable), decltype(input), Args...>) {
        return std::invoke(std::forward<decltype(callable)>(callable),
                           std::forward<decltype(input)>(input),
                           std::forward<Args>(args)...);
    } else {
        static_assert(std::is_invocable_v<decltype(callable), decltype(input), Args...>,
                      "bound program must accept the cell information, or provide an ADL predict hook");
    }
}

template<class Program, class Information, class... Args>
decltype(auto) invoke_query(Program&& program, Information&& information, Args&&... args) {
    auto&& callable = unwrap_program(std::forward<Program>(program));
    auto&& input = resolve_information(std::forward<Information>(information));
    // predict_query is the evaluation hook for typed NN requests. Check it
    // first because nn::query also constructs requests from three operands.
    if constexpr (has_adl_predict_query_v<decltype(callable), decltype(input), Args&&...>) {
        return adl::call_predict_query(std::forward<decltype(callable)>(callable),
                                       std::forward<decltype(input)>(input),
                                       std::forward<Args>(args)...);
    } else if constexpr (has_adl_query_v<decltype(callable), decltype(input), Args&&...>) {
        return adl::call_query(std::forward<decltype(callable)>(callable),
                               std::forward<decltype(input)>(input),
                               std::forward<Args>(args)...);
    } else if constexpr (has_member_query_v<decltype(callable), decltype(input), Args&&...>) {
        return std::forward<decltype(callable)>(callable).query(
            std::forward<decltype(input)>(input), std::forward<Args>(args)...);
    } else if constexpr (std::is_invocable_v<decltype(callable), decltype(input), Args...>) {
        return std::invoke(std::forward<decltype(callable)>(callable),
                           std::forward<decltype(input)>(input),
                           std::forward<Args>(args)...);
    } else {
        static_assert(std::is_invocable_v<decltype(callable), decltype(input), Args...>,
                      "bound program must accept the cell information, or provide an ADL query hook");
    }
}

template<class Program>
auto store_bound_program(Program&& program) {
    if constexpr (std::is_lvalue_reference_v<Program&&> &&
                  !is_reference_wrapper_v<Program> && !is_shared_ptr_v<Program>) {
        return std::ref(program);
    } else {
        return std::decay_t<Program>(std::forward<Program>(program));
    }
}

} // namespace detail

template<class CellType, class Program>
class BoundCell {
public:
    using cell_type = CellType;
    using program_storage_type = Program;

    BoundCell(const CellType& cell, Program program)
        : cell_(&cell), program_(std::move(program)) {}

    const CellType& cell() const noexcept { return *cell_; }
    Program& program() noexcept { return program_; }
    const Program& program() const noexcept { return program_; }

    template<class... Args>
    decltype(auto) predict(Args&&... args) {
        return detail::invoke_prediction(program_, cell_->information(),
                                         std::forward<Args>(args)...);
    }

    template<class... Args>
    decltype(auto) predict(Args&&... args) const {
        return detail::invoke_prediction(program_, cell_->information(),
                                         std::forward<Args>(args)...);
    }

    template<class... Args>
    decltype(auto) query(Args&&... args) {
        return detail::invoke_query(program_, cell_->information(),
                                    std::forward<Args>(args)...);
    }

    template<class... Args>
    decltype(auto) query(Args&&... args) const {
        return detail::invoke_query(program_, cell_->information(),
                                    std::forward<Args>(args)...);
    }

private:
    const CellType* cell_;
    Program program_;
};

// A Cell names an identity and a source of information. It imposes no clock,
// state-vector, model base, tensor representation, or parameter ownership.
// Value sources are retained; use std::ref for a borrowed source, shared_ptr
// for shared ownership, or a zero-argument provider for a live view.
template<class Information, class Identity = std::monostate>
class Cell {
public:
    using information_source_type = Information;
    using information_type = std::remove_cv_t<std::remove_reference_t<decltype(
        detail::resolve_information(std::declval<const Information&>()))>>;
    using identity_type = Identity;

    explicit Cell(Information information)
        : information_(std::move(information)), identity_{} {}

    Cell(Information information, Identity identity)
        : information_(std::move(information)), identity_(std::move(identity)) {}

    decltype(auto) information() const {
        return detail::resolve_information(information_);
    }

    decltype(auto) data() const { return information(); }

    const Information& information_source() const noexcept { return information_; }
    const Identity& id() const noexcept { return identity_; }
    const Identity& identity() const noexcept { return identity_; }

    template<class Program>
    auto bind(Program&& program) const & {
        auto stored = detail::store_bound_program(std::forward<Program>(program));
        return BoundCell<Cell, decltype(stored)>(*this, std::move(stored));
    }

    template<class Program>
    auto bind(Program&&) const && = delete;

    template<class Program>
    auto bind_owned(Program&& program) const & {
        using stored_type = std::decay_t<Program>;
        return BoundCell<Cell, stored_type>(*this, stored_type(std::forward<Program>(program)));
    }

    template<class Program>
    auto bind_owned(Program&&) const && = delete;

private:
    Information information_;
    Identity identity_;
};

template<class Information>
Cell(Information&&) -> Cell<std::decay_t<Information>>;

template<class Information, class Identity>
Cell(Information&&, Identity&&) -> Cell<std::decay_t<Information>, std::decay_t<Identity>>;

template<class Information>
auto make_cell(Information&& information) {
    return Cell<std::decay_t<Information>>(std::forward<Information>(information));
}

template<class Information, class Identity>
auto make_cell(Information&& information, Identity&& identity) {
    return Cell<std::decay_t<Information>, std::decay_t<Identity>>(
        std::forward<Information>(information), std::forward<Identity>(identity));
}

} // namespace cellerator::modeling
