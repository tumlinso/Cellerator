#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace cellerator::compiler::lto::v1 {
enum class elf_ceir_compression_v1:std::uint8_t{none=0,zlib,zstd};
struct elf_ceir_section_v1{std::string name,note,symbol;std::vector<std::uint8_t>payload;elf_ceir_compression_v1 compression=elf_ceir_compression_v1::none;bool allocatable=false,retain_when_stripped=true;};
[[nodiscard]] bool validate_elf_ceir_section_v1(const elf_ceir_section_v1&)noexcept;
[[nodiscard]] std::vector<std::uint8_t> emit_elf_ceir_section_v1(const elf_ceir_section_v1&);
[[nodiscard]] std::optional<elf_ceir_section_v1> extract_elf_ceir_section_v1(const std::vector<std::uint8_t>&);
[[nodiscard]] std::vector<std::uint8_t> strip_elf_runtime_symbols_v1(const std::vector<std::uint8_t>&);
}
