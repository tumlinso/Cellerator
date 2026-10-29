#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/entity_layout.hh>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

using cellerator::execution::domain_id;
using cellerator::execution::order_id;
using cellerator::modeling::EntityLayout;

template<typename Exception, typename Callable>
void expect_throw(Callable &&callable) {
    bool thrown = false;
    try { callable(); }
    catch (const Exception &) { thrown = true; }
    assert(thrown);
}

int main() {
    const EntityLayout source({10u, 0u, 42u}, domain_id{7u, 8u}, order_id{1u, 0u});
    const EntityLayout target({42u, 10u, 0u}, domain_id{7u, 8u}, order_id{2u, 0u});
    assert(source.size() == 3u);
    assert(source.slot(0u) == 1u);
    assert(source.entity_ids()[2] == 42u);
    const auto permutation = source.permutation_to(target);
    assert((permutation == std::vector<std::size_t>{2u, 0u, 1u}));

    const auto reordered = source.reorder(
        std::vector<int>{100, 101, 0, 1, 420, 421}, 2u, target);
    assert((reordered == std::vector<int>{420, 421, 100, 101, 0, 1}));

    expect_throw<std::invalid_argument>([] {
        EntityLayout duplicate({1u, 1u});
    });
    expect_throw<std::invalid_argument>([&] {
        const EntityLayout other_domain({10u, 0u, 42u}, domain_id{99u, 0u});
        (void)source.permutation_to(other_domain);
    });
    expect_throw<std::invalid_argument>([&] {
        const EntityLayout different_set({10u, 0u, 99u}, domain_id{7u, 8u});
        (void)source.permutation_to(different_set);
    });
    expect_throw<std::out_of_range>([&] { (void)source.slot(99u); });
    expect_throw<std::invalid_argument>([&] {
        (void)source.reorder(std::vector<int>{1, 2, 3}, 0u, target);
    });
    expect_throw<std::invalid_argument>([&] {
        (void)source.reorder(std::vector<int>{1, 2}, 1u, target);
    });
    return 0;
}
