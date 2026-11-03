#include "wire_input.hh"

#include <cstddef>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const auto paths = factorized_wire::input_paths(argc, argv);
        const auto bytes = factorized_wire::read_input_bytes(paths);
        if (!bytes.empty())
            std::cout.write(reinterpret_cast<const char*>(bytes.data()),
                            static_cast<std::streamsize>(bytes.size()));
        if (!std::cout) throw std::runtime_error("cannot write reconstructed stream");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
