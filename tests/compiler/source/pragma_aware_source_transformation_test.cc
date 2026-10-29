#include <Cellerator/compiler/frontend/source/pragma_aware_source_transformation_v1.hh>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        const std::vector<source_unit_v1> units{
            {1, "/tmp/cellerator_pragma_aware_source_transformation.hh", "#pragma once\ninline int ordinary(){return 1;}\n"},
            {2, "/tmp/cellerator_pragma_aware_source_transformation.hh", "#pragma once\n#pragma cellerator 0.1\ninline int activated(){return <[42]>;}\n"},
            {3, "/tmp/cellerator_pragma_aware_source_transformation.cc", "#include \"cellerator_pragma_aware_source_transformation.hh\"\n#include \"cellerator_pragma_aware_source_transformation.hh\"\n#pragma cellerator\nint main(){return <[ordinary()+activated()]>-2;}\n"},
        };
        const auto transformed = transform_sources_with_pragmas_v1(units);
        if (transformed.size() != 3 || transformed[0].dialect_activated ||
            !transformed[1].dialect_activated || !transformed[2].dialect_activated ||
            transformed[0].shadow_bytes != units[0].bytes || transformed[1].placeholders.size() != 1 ||
            transformed[2].placeholders.size() != 1) throw std::runtime_error("mixed source transformation failed");
        for (const auto& unit : transformed) { std::ofstream output(unit.path); output << unit.shadow_bytes; }
        std::ofstream support("/tmp/cellerator_pragma_aware_source_transformation.hh");
        support << "template<unsigned long long> constexpr int cellerator_shadow_field(){return 2;}\n";
        support.close();
        const char* host = "clang++-18 -std=c++17 -fsyntax-only -I/tmp -include /tmp/cellerator_pragma_aware_source_transformation.hh /tmp/cellerator_pragma_aware_source_transformation.cc";
        const char* cuda = "clang++-18 -std=c++17 -DCELLERATOR_CUDA_ENABLED=1 -fsyntax-only -I/tmp -include /tmp/cellerator_pragma_aware_source_transformation.hh /tmp/cellerator_pragma_aware_source_transformation.cc";
        if (std::system(host) != 0 || std::system(cuda) != 0) throw std::runtime_error("Clang rejected mixed shadow translation unit");
        std::cout << "validated pragma-aware mixed source transformation in host and CUDA-enabled modes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
