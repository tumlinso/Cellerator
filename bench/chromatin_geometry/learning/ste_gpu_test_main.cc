#include "learning/ste.hh"

#include <iostream>
#include <string>

// Invoke only inside an admitted study GPU job.
int main() {
    std::string error;
    if (!cgs::run_ste_tests(nullptr, &error)) {
        std::cerr << "STE CUDA validation failed: " << error << '\n';
        return 1;
    }
    std::cout << "STE CUDA validation passed\n";
    return 0;
}
