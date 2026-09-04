#include <Cellerator/compiler/frontend/source/define_pragma_interaction_with_preprocessing_conditional_v1.hh>

#include <array>
#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        constexpr std::array sources{preprocessing_source_v1::textual,
                                     preprocessing_source_v1::include_replay,
                                     preprocessing_source_v1::precompiled_header,
                                     preprocessing_source_v1::module};
        for (auto source : sources) {
            if (!pragma_may_activate_v1({true, false, source, true}) ||
                pragma_may_activate_v1({false, true, source, true}) ||
                pragma_may_activate_v1({true, false, source, false})) {
                throw std::runtime_error("conditional or cached-source policy diverged");
            }
        }
        std::cout << "validated pragma preprocessing condition and cache replay policy\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
