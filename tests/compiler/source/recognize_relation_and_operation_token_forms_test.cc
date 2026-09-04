#include <Cellerator/compiler/frontend/source/recognize_relation_and_operation_token_forms_v1.hh>

#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        const auto scan = recognize_operation_forms_v1(8,
            "x - /*a*/ [ relation<std::array<int,2>> ] /*b*/ - > y;");
        if (!scan.recovered || scan.forms.size() != 1 ||
            scan.forms[0].payload.find("relation<") == std::string::npos) {
            throw std::runtime_error("spaced/commented relation form not recognized");
        }
        const auto bad = recognize_operation_forms_v1(8, "x -[ relation ]- y; z -[ok]-> q;");
        if (bad.recovered || bad.forms.size() != 1) {
            throw std::runtime_error("operation-form recovery failed");
        }
        std::cout << "validated syntax-only relation operation recognition and recovery\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
