#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::diagnostics::v1 {struct structured_diagnostic{std::uint64_t stable_id=0,related_id=0,file_id=0;std::uint32_t line=0,column=0,severity=0;std::string code,message;};using diagnostic_callback=bool(*)(const structured_diagnostic&,void*);using cancellation_query=bool(*)(void*);[[nodiscard]] std::string to_lsp_json(const structured_diagnostic&);[[nodiscard]] std::size_t stream_diagnostics(const std::vector<structured_diagnostic>&,diagnostic_callback,void*,cancellation_query=nullptr,void* cancellation_context=nullptr);}
