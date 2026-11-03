#include "../workloads.hh"

#include <cstdlib>
#include <iostream>
#include <string>

int main() {
    std::string error;
    if (!cgs::run_reference_checks(&error)) {
        std::cerr << "cgs reference checks failed: " << error << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "cgs reference checks passed\n";
    return EXIT_SUCCESS;
}
