#include <Cellerator/compiler/driver/define_toolchain_override_precedence_v1.hh>
namespace cellerator::compiler::driver {
resolved_override_v1 resolve_toolchain_override_v1(const override_candidates_v1& candidates) noexcept { for (std::size_t i = 0; i != candidates.values.size(); ++i) if (!candidates.values[i].empty()) return {static_cast<override_source_v1>(i), candidates.values[i]}; return {}; }
std::string_view override_source_name_v1(override_source_v1 source) noexcept { constexpr std::array names{std::string_view{"command-line"}, std::string_view{"response-file"}, std::string_view{"environment"}, std::string_view{"build-configuration"}, std::string_view{"resource-manifest"}, std::string_view{"system-discovery"}, std::string_view{"unresolved"}}; return names[static_cast<std::size_t>(source)]; }
}  // namespace cellerator::compiler::driver
