#include <Cellerator/compiler/tooling/implement_virtual_shadow_document_mapping_v1.hh>

#include <iostream>

int main() {
    std::string source;
    std::getline(std::cin, source, '\0');
    Cellerator::compiler::tooling::virtual_shadow_document_v1 document("stdin.cell", source);
    if (!document.append_original({0, source.size()})) return 1;
    std::cout << document.shadow_text();
    return 0;
}
