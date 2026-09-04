#include <Cellerator/compiler/sema/integrate_cxx_templates_and_concepts_with_biological_constraints_v1.hh>

#include <cassert>
#include <type_traits>

struct gene {};
struct cell {};
struct csr_layout {};
struct packed_layout {};

namespace cellerator::compiler::sema::v1 {
template<> struct semantic_domain_traits<::gene> { static constexpr bool is_domain = true; };
template<> struct semantic_domain_traits<::cell> { static constexpr bool is_domain = true; };
}  // namespace cellerator::compiler::sema::v1

template<typename D, typename N, typename L>
using self_relation = cellerator::compiler::sema::v1::relation_operation_instantiation<D, D, N, L>;

int main() {
    using namespace cellerator::compiler::sema::v1;
    using gene_f32 = self_relation<gene, float, csr_layout>;
    using cell_f64 = self_relation<cell, double, packed_layout>;
    static_assert(gene_f32::execution_numeric == cellerator::execution::numeric_type::f32);
    static_assert(cell_f64::execution_numeric == cellerator::execution::numeric_type::f64);
    static_assert(operation_uses_layout_v<gene_f32, csr_layout>);
    static_assert(!operation_uses_layout_v<gene_f32, packed_layout>);
    assert(cxx_biological_constraints_revision() != nullptr);
}
