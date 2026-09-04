#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace cellerator::compiler::lto::v1 {
using content_identity_v1=std::array<std::uint8_t,32>;
struct ceir_sidecar_v1{content_identity_v1 identity{};std::vector<std::uint8_t>payload;};
struct object_sidecar_reference_v1{content_identity_v1 identity{};std::string hint;};
[[nodiscard]] content_identity_v1 identify_sidecar_content_v1(const std::vector<std::uint8_t>&)noexcept;
[[nodiscard]] std::optional<std::size_t> resolve_sidecar_v1(const object_sidecar_reference_v1&,const std::vector<ceir_sidecar_v1>&)noexcept;
[[nodiscard]] std::string sidecar_filename_v1(content_identity_v1);
}
