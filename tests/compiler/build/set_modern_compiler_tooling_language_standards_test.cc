#include <Cellerator/compiler/build/set_modern_compiler_tooling_language_standards_v1.hh>

int main() {
    using namespace Cellerator::compiler::build;
    static_assert(language_standards_contract_v1.compiler_implementation == 23);
    static_assert(language_standards_contract_v1.tooling_implementation == 23);
    static_assert(language_standards_contract_v1.public_header_minimum == 17);
    static_assert(language_standards_contract_v1.legacy_runtime == 17);
    static_assert(language_standards_contract_v1.legacy_cuda == 17);
    static_assert(language_standards_contract_v1.implementation_mode_is_private);
}
