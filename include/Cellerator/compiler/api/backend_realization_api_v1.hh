#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::api::v1 {
struct target_description_v1{std::string architecture;std::string triple;};
struct physical_ir_v1{std::vector<std::string> operations;std::vector<std::string> source_maps;};
struct generated_artifact_v1{std::string kind;std::vector<std::uint8_t> bytes;};
using native_fragment_hook_v1=bool(*)(physical_ir_v1&,void*) noexcept;
using backend_emit_v1=bool(*)(const target_description_v1&,const physical_ir_v1&,generated_artifact_v1&,void*) noexcept;
struct backend_v1{std::string name;backend_emit_v1 emit=nullptr;native_fragment_hook_v1 fragment=nullptr;void* user_data=nullptr;};
class backend_registry_v1{public:bool add(backend_v1);const backend_v1* find(const std::string&)const noexcept;private:std::vector<backend_v1> entries_;};
[[nodiscard]] bool emit_object_v1(const backend_v1&,const target_description_v1&,physical_ir_v1,generated_artifact_v1&) noexcept;
}
