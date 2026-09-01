#include <Cellerator/execution/object_binding/multi_extent_v1.hh>

#include <cassert>

namespace binding = cellerator::execution::object_binding;

int main() {
    const std::uint64_t map[] = {2u, 0u, 3u, 1u};
    const binding::index_permutation_v1 permutation{map, 4u};
    assert(binding::validate_index_permutation_v1(permutation));

    const int canonical[] = {10, 20, 30, 40};
    int gathered[4]{};
    assert(binding::gather_permutation_v1(
        canonical, gathered, sizeof(int), permutation));
    assert(gathered[0] == 30 && gathered[1] == 10 &&
        gathered[2] == 40 && gathered[3] == 20);

    int scattered[4]{};
    assert(binding::scatter_permutation_v1(
        gathered, scattered, sizeof(int), permutation));
    assert(scattered[0] == 10 && scattered[1] == 20 &&
        scattered[2] == 30 && scattered[3] == 40);

    const std::uint64_t duplicate_map[] = {0u, 0u};
    assert(binding::validate_index_permutation_v1(
               {duplicate_map, 2u}).code ==
        binding::binding_status_code_v1::duplicate_atom);
}
