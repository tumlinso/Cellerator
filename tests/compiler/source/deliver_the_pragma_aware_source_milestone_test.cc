#include <Cellerator/compiler/frontend/source/deliver_the_pragma_aware_source_milestone_v1.hh>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        const std::vector<source_unit_v1> units{
            {1, "/tmp/ce_ccp1_b03_015_inactive.hh", "#pragma once\ninline int ordinary(){return 1;}\n"},
            {2, "/tmp/ce_ccp1_b03_015_active.hh", "#pragma once\n#pragma cellerator 0.1\ninline int activated(){return <[42]>;}\n"},
            {3, "/tmp/ce_ccp1_b03_015_main.cc", "#include \"ce_ccp1_b03_015_inactive.hh\"\n#include \"ce_ccp1_b03_015_active.hh\"\n#pragma cellerator\nint main(){return <[ordinary()+activated()]>-2;}\n"},
        };
        const auto transformed = transform_pragma_aware_sources_v1(units);
        if (transformed.size() != 3 || transformed[0].dialect_activated ||
            !transformed[1].dialect_activated || !transformed[2].dialect_activated ||
            transformed[0].shadow_bytes != units[0].bytes || transformed[1].placeholders.size() != 1 ||
            transformed[2].placeholders.size() != 1) throw std::runtime_error("mixed source transformation failed");
        for (const auto& unit : transformed) { std::ofstream output(unit.path); output << unit.shadow_bytes; }
        std::ofstream support("/tmp/ce_ccp1_b03_015_support.hh");
        support << "template<unsigned long long> constexpr int cellerator_shadow_field(){return 2;}\n";
        support.close();
        const char* host = "clang++-18 -std=c++17 -fsyntax-only -I/tmp -include /tmp/ce_ccp1_b03_015_support.hh /tmp/ce_ccp1_b03_015_main.cc";
        const char* cuda = "clang++-18 -std=c++17 -DCELLERATOR_CUDA_ENABLED=1 -fsyntax-only -I/tmp -include /tmp/ce_ccp1_b03_015_support.hh /tmp/ce_ccp1_b03_015_main.cc";
        if (std::system(host) != 0 || std::system(cuda) != 0) throw std::runtime_error("Clang rejected mixed shadow translation unit");
        std::cout << "validated pragma-aware mixed source milestone in host and CUDA-enabled modes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
