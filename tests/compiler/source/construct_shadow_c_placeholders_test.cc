#include <Cellerator/compiler/frontend/source/construct_shadow_c_placeholders_v1.hh>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        const std::string source = "int before=1; int result=<[before]>; int after=result;\n";
        const auto begin = source.find("<[");
        const auto shadow = construct_shadow_cxx_v1(5, source, {{{5, begin}, {5, begin + 10}}},
                                                    {{{"before", "int"}}});
        if (shadow.placeholders.size() != 1 || shadow.placeholders[0].captures.size() != 1 ||
            shadow.bytes.find("int before=1;") != 0 || shadow.bytes.find("int after=result;") == std::string::npos) {
            throw std::runtime_error("shadow metadata or verbatim C++ was lost");
        }
        const auto again = construct_shadow_cxx_v1(5, source, {{{5, begin}, {5, begin + 10}}});
        if (again.placeholders[0].stable_id != shadow.placeholders[0].stable_id) throw std::runtime_error("placeholder ID unstable");
        const char* path = "/tmp/ce_ccp1_b03_010_shadow.cc";
        std::ofstream output(path);
        output << "template<unsigned long long> int cellerator_shadow_field(){return 2;}\n" << shadow.bytes;
        output.close();
        if (std::system("clang++-18 -std=c++17 -fsyntax-only /tmp/ce_ccp1_b03_010_shadow.cc") != 0)
            throw std::runtime_error("upstream Clang rejected shadow C++");
        std::cout << "validated stable typed shadow-C++ placeholders with Clang\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
