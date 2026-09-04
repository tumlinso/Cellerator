#pragma once

#include <Cellerator/execution/operands.hh>

#include <type_traits>

namespace cellerator::compiler::sema::v1 {

template<typename T>
struct semantic_domain_traits {
    static constexpr bool is_domain = false;
};

template<typename T>
inline constexpr bool semantic_domain_v = semantic_domain_traits<T>::is_domain;

template<typename T>
struct execution_numeric_traits {
    static constexpr execution::numeric_type type = execution::numeric_type::invalid;
};
template<> struct execution_numeric_traits<float> {
    static constexpr execution::numeric_type type = execution::numeric_type::f32;
};
template<> struct execution_numeric_traits<double> {
    static constexpr execution::numeric_type type = execution::numeric_type::f64;
};

template<typename SourceDomain, typename DestinationDomain,
         typename Numeric, typename Layout>
struct relation_operation_instantiation {
    static_assert(semantic_domain_v<SourceDomain>, "source must be a semantic domain");
    static_assert(semantic_domain_v<DestinationDomain>, "destination must be a semantic domain");
    static_assert(execution_numeric_traits<Numeric>::type != execution::numeric_type::invalid,
                  "numeric type must have an execution mapping");

    using source_domain = SourceDomain;
    using destination_domain = DestinationDomain;
    using numeric_type = Numeric;
    using layout_type = Layout;
    static constexpr execution::numeric_type execution_numeric =
        execution_numeric_traits<Numeric>::type;
};

template<typename Operation, typename ExpectedLayout>
inline constexpr bool operation_uses_layout_v =
    std::is_same_v<typename Operation::layout_type, ExpectedLayout>;

const char *cxx_biological_constraints_revision() noexcept;

}  // namespace cellerator::compiler::sema::v1
