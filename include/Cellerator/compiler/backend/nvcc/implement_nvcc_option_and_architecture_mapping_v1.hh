#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace cellerator::compiler::backend::nvcc::v1 {
struct nvcc_options{std::vector<std::uint32_t> real_architectures,virtual_architectures;std::string host_compiler="c++";std::uint32_t cxx_standard=17,optimization=3;bool debug=false,line_info=true,rdc=false;std::vector<std::string> libraries,user_options;};
enum class option_status:std::uint8_t{ok=0,missing_architecture,unsupported_architecture,invalid_standard,invalid_optimization,unsafe_override};
[[nodiscard]] std::optional<std::vector<std::string>> make_nvcc_argv(const nvcc_options&,option_status* = nullptr) noexcept;
}
