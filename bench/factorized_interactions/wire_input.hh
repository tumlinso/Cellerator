#pragma once

// Host-only transport for the existing CFINT001 byte stream. The native parser
// consumes the concatenation; ordered shard boundaries have no semantic role.
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace factorized_wire {

inline std::vector<std::string> input_paths(int argc, char** argv) {
    std::vector<std::string> paths;
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option != "--input" && option != "--input-shard") continue;
        if (i + 1 >= argc || std::string(argv[i + 1]).empty())
            throw std::invalid_argument("input option requires a file path");
        if (option == "--input" && !paths.empty())
            throw std::invalid_argument("--input must occur once before its shards");
        if (option == "--input-shard" && paths.empty())
            throw std::invalid_argument("--input-shard requires a preceding --input");
        paths.emplace_back(argv[++i]);
    }
    return paths;
}

inline std::vector<std::byte> read_input_bytes(const std::vector<std::string>& paths) {
    if (paths.empty()) throw std::invalid_argument("case input is required");
    std::vector<std::size_t> lengths;
    lengths.reserve(paths.size());
    std::size_t total = 0;
    for (const auto& path : paths) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("cannot open case input: " + path);
        const auto end = input.tellg();
        if (end < 0 || static_cast<unsigned long long>(end) >
                           std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("cannot size case input: " + path);
        const auto length = static_cast<std::size_t>(end);
        if (length > std::numeric_limits<std::size_t>::max() - total ||
            length > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::overflow_error("case input length overflows the host");
        lengths.push_back(length);
        total += length;
    }
    std::vector<std::byte> bytes(total);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        std::ifstream input(paths[i], std::ios::binary);
        if (!input) throw std::runtime_error("cannot reopen case input: " + paths[i]);
        if (lengths[i])
            input.read(reinterpret_cast<char*>(bytes.data() + offset),
                       static_cast<std::streamsize>(lengths[i]));
        if (!input || input.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error("case input changed while reading: " + paths[i]);
        offset += lengths[i];
    }
    return bytes;
}

} // namespace factorized_wire
