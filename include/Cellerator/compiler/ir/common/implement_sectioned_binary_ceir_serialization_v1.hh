#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cellerator::compiler::ir {

inline constexpr std::uint32_t binary_ceir_magic = 0x52494543u; // CEIR, little endian
struct binary_ceir_header {
    std::uint32_t magic{};
    std::uint16_t major{};
    std::uint16_t minor{};
    std::uint32_t total_bytes{};
    std::uint32_t directory_offset{};
    std::uint32_t section_count{};
    std::uint32_t checksum{};
};
struct binary_ceir_section {
    std::uint32_t kind{};
    std::uint32_t offset{};
    std::uint32_t size{};
    std::uint32_t checksum{};
};
enum class binary_ceir_validation {
    ok, too_small, bad_magic, unsupported_version, bad_size,
    bad_directory, bad_section, bad_checksum
};
struct binary_section_input { std::uint32_t kind{}; std::vector<std::uint8_t> bytes; };

std::uint32_t binary_ceir_checksum(const std::uint8_t *data, std::size_t size) noexcept;
std::vector<std::uint8_t> build_binary_ceir(
    const std::vector<binary_section_input> &sections, std::uint16_t minor = 0u);
binary_ceir_validation validate_binary_ceir(
    const std::uint8_t *data, std::size_t size) noexcept;

} // namespace cellerator::compiler::ir
