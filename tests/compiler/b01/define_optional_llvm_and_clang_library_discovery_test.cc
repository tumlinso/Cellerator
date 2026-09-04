#include <Cellerator/compiler/build/define_optional_llvm_and_clang_library_discovery_v1.hh>

int main() {
    using namespace Cellerator::compiler::build;
    static_assert(usable({true, "18.1.3", "abi-checks", true, false}));
    static_assert(!usable({false, {}, {}, true, false}));
    static_assert(!usable({true, "18.1.3", "abi-checks", false, true}));
}
