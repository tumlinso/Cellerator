#include <cassert>
struct version_pair { unsigned compiler; unsigned ceir; };
constexpr bool compatible(version_pair compiler, version_pair resource) { return compiler.compiler==resource.compiler && compiler.ceir==resource.ceir; }
int main() {
    static_assert(compatible({1,1},{1,1}));
    static_assert(!compatible({1,1},{2,1}));
    static_assert(!compatible({1,1},{1,2}));
    assert(!compatible({2,1},{1,1}));
}
