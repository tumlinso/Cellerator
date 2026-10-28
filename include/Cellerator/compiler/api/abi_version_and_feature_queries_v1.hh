#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
namespace cellerator::compiler::api::v1 {
inline constexpr std::uint32_t compiler_api_abi_v1=1, source_language_revision_v1=1;
struct version_triplet_v1{std::uint32_t major=0,minor=0,patch=0;};
struct compiler_features_v1{version_triplet_v1 compiler{};std::vector<std::uint32_t> ceir_levels;std::vector<std::string> optional_features;};
struct provider_descriptor_v1{std::string name;version_triplet_v1 api{};std::vector<std::string> features;};
struct compatibility_result_v1{bool compatible=false;std::string diagnostic;};
[[nodiscard]] compiler_features_v1 query_compiler_features_v1();
[[nodiscard]] bool has_feature_v1(const compiler_features_v1&,std::string_view) noexcept;
[[nodiscard]] compatibility_result_v1 check_provider_compatibility_v1(const provider_descriptor_v1&);
}
