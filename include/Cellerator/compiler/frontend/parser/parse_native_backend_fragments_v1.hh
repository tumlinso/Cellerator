#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

enum class native_backend_kind_v1 { generated_cxx, cuda, ptx, raw_native };

struct native_backend_fragment_v1 {
    native_backend_kind_v1 backend = native_backend_kind_v1::generated_cxx;
    std::string target;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> clobbers;
    std::vector<std::string> effects;
    std::string fallback;
    std::string payload;
    std::string spelling;
    parser_source_range_v1 range{};
};

struct native_backend_fragment_parse_v1 {
    std::vector<native_backend_fragment_v1> fragments;
    std::vector<declaration_diagnostic_v1> diagnostics;
    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

[[nodiscard]] native_backend_fragment_parse_v1 parse_native_backend_fragments_v1(
    std::string_view source);

} // namespace Cellerator::compiler::frontend::parser
