#include <Cellerator/compiler/frontend/source/register_the_pragma_cellerator_preprocessor_contract_v1.hh>

#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        const source_location_v1 at{7, 41};
        const auto valid = handle_cellerator_pragma_v1({"0.1", at});
        const auto unversioned = handle_cellerator_pragma_v1({"", at});
        if (!valid.activate || valid.revision != "0.1" || valid.location.byte_offset != 41 ||
            !unversioned.activate || unversioned.revision != "0.1") {
            throw std::runtime_error("valid pragma was rejected");
        }
        if (handle_cellerator_pragma_v1({"0.1", at, true}).diagnostic != pragma_diagnostic_v1::duplicate ||
            handle_cellerator_pragma_v1({"0.1", at, false, true}).diagnostic != pragma_diagnostic_v1::late_activation ||
            handle_cellerator_pragma_v1({"0.1 extra", at}).diagnostic != pragma_diagnostic_v1::malformed ||
            handle_cellerator_pragma_v1({"0.1", at, false, false, true}).diagnostic != pragma_diagnostic_v1::macro_produced ||
            handle_cellerator_pragma_v1({"9.9", at}).diagnostic != pragma_diagnostic_v1::unknown_version) {
            throw std::runtime_error("pragma diagnostics were not deterministic");
        }
        std::cout << "validated pragma preprocessor contract and diagnostics\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
