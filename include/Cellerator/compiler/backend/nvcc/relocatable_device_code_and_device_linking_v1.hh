#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace cellerator::compiler::backend::nvcc::v1 {
struct device_object{std::string path;std::vector<std::string> definitions,references;};
struct device_link_request{std::vector<device_object> objects;std::vector<std::string> libraries;std::string linked_object,registration_object;std::vector<std::uint32_t> architectures;};
struct device_link_plan{std::vector<std::string> compile_actions;std::vector<std::string> nvlink_argv;std::string registration_action;};
enum class device_link_status:std::uint8_t{ok=0,insufficient_objects,invalid_path,duplicate_symbol,unresolved_symbol,missing_architecture};
[[nodiscard]] std::optional<device_link_plan> plan_device_link(const device_link_request&,device_link_status* = nullptr) noexcept;
}
