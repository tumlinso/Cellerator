#include <Cellerator/compiler/frontend/source/build_a_lossless_raw_token_stream_v1.hh>

#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        const std::string bytes = "// raw\n#pragma cellerator 0.1\n<[a\t-[r]-> b]>;  \n";
        const auto activation = bytes.find("<[");
        auto stream = build_raw_token_stream_v1(9, bytes, activation, 17);
        if (reconstruct_raw_token_stream_v1(stream) != bytes ||
            !has_exact_byte_coverage_v1(stream, bytes.size())) {
            throw std::runtime_error("raw token stream was not lossless");
        }
        bool saw_inactive = false, saw_active = false;
        for (auto& token : stream.tokens) {
            saw_inactive |= !token.dialect_active;
            saw_active |= token.dialect_active;
            if (token.preprocessor_condition != 17) throw std::runtime_error("condition lost");
        }
        stream.tokens.back().macro_origin = source_span_v1{{3, 4}, {3, 5}};
        if (!saw_inactive || !saw_active || !stream.tokens.back().macro_origin) {
            throw std::runtime_error("token provenance fields missing");
        }
        std::cout << "validated lossless raw tokens and exact byte coverage\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
