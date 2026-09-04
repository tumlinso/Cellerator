#include <Cellerator/compiler/frontend/source/recognize_cellerator_execution_field_token_islands_v1.hh>

#include <array>
#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        constexpr std::string_view source =
            "std::vector<std::array<int, 2>> x; auto s=\"<[not]>\"; // <[no]>\n"
            "<[ outer <[nested]> (x << 1) [[attr]] ]>; /* ]> */";
        const auto scan = recognize_execution_field_islands_v1(4, source);
        if (!scan.balanced || scan.islands.size() != 1 ||
            source.substr(scan.islands[0].begin.byte_offset, scan.islands[0].size_bytes()).find("nested") == std::string_view::npos) {
            throw std::runtime_error("field-island recognition confused C++ tokens");
        }
        for (auto malformed : std::array<std::string_view, 2>{"<[ x", "x ]>"}) {
            if (recognize_execution_field_islands_v1(1, malformed).balanced)
                throw std::runtime_error("unbalanced delimiter accepted");
        }
        std::cout << "validated nesting-aware execution-field island recognition\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
