#include <Cellerator/compiler/tooling/freeze_the_celleratord_architecture_v1.hh>

#include <iostream>

int main() {
    const Cellerator::compiler::tooling::celleratord_architecture_v1 architecture;
    if (Cellerator::compiler::tooling::validate_celleratord_architecture_v1(architecture)
        != Cellerator::compiler::tooling::celleratord_architecture_status_v1::valid)
        return 1;
    std::cout << "celleratord architecture v1: libCellerator snapshots + upstream clangd\n";
    return 0;
}
