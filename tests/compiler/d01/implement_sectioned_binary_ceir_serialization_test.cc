#include <Cellerator/compiler/ir/common/implement_sectioned_binary_ceir_serialization_v1.hh>

#include <cassert>
#include <cstring>

using namespace cellerator::compiler::ir;

int main() {
    const auto valid = build_binary_ceir({{1u, {1u, 2u, 3u}}, {9u, {0xffu}}}, 2u);
    assert(validate_binary_ceir(valid.data(), valid.size()) == binary_ceir_validation::ok);
    assert(validate_binary_ceir(valid.data(), sizeof(binary_ceir_header) - 1u)
        == binary_ceir_validation::too_small);

    for (const auto field : {offsetof(binary_ceir_header, magic),
             offsetof(binary_ceir_header, major), offsetof(binary_ceir_header, total_bytes),
             offsetof(binary_ceir_header, directory_offset),
             offsetof(binary_ceir_header, section_count),
             offsetof(binary_ceir_header, checksum)}) {
        auto corrupt = valid;
        corrupt[field] ^= 0x7fu;
        assert(validate_binary_ceir(corrupt.data(), corrupt.size()) != binary_ceir_validation::ok);
    }
    for (std::size_t field = 0u; field < sizeof(binary_ceir_section); field += 4u) {
        auto corrupt = valid;
        corrupt[sizeof(binary_ceir_header) + field] ^= 0x55u;
        assert(validate_binary_ceir(corrupt.data(), corrupt.size()) != binary_ceir_validation::ok);
    }
    auto payload_corrupt = valid;
    payload_corrupt.back() ^= 1u;
    assert(validate_binary_ceir(payload_corrupt.data(), payload_corrupt.size())
        == binary_ceir_validation::bad_checksum);
}
