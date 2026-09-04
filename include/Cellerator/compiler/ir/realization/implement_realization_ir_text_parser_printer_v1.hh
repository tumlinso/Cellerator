#pragma once
#include <Cellerator/compiler/ir/realization/freeze_realization_ir_module_and_target_scopes_v1.hh>
#include <optional>
#include <string>
#include <vector>
namespace cellerator::compiler::ir::realization::v1 {
struct realization_text_record_v1{std::string kind;stable_identity_v1 identity{};std::string payload;};
struct realization_text_document_v1{stable_identity_v1 module{};std::vector<realization_text_record_v1>records;};
[[nodiscard]] std::string print_realization_text_v1(const realization_text_document_v1&);
[[nodiscard]] std::optional<realization_text_document_v1> parse_realization_text_v1(const std::string&,std::string*error=nullptr);
[[nodiscard]] bool equivalent_realization_text_v1(const realization_text_document_v1&,const realization_text_document_v1&)noexcept;
} // namespace cellerator::compiler::ir::realization::v1
