#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::api::v1 {

struct source_document_v1 { std::string name; std::string text; std::uint64_t revision = 1; };
struct parse_snapshot_v1 { std::uint64_t source_revision = 0; std::vector<std::string> tokens; };
struct semantic_snapshot_v1 {
    std::uint64_t source_revision = 0;
    std::vector<std::string> declared_symbols;
    bool planning_ran = false;
    bool code_generation_ran = false;
};
using analysis_cancelled_v1 = bool (*)(void*) noexcept;

[[nodiscard]] parse_snapshot_v1 parse_source_v1(const source_document_v1& source);
void update_source_v1(source_document_v1& source, std::string text);
[[nodiscard]] bool analyze_semantics_v1(const source_document_v1& source,
    semantic_snapshot_v1& output, analysis_cancelled_v1 cancelled = nullptr,
    void* user_data = nullptr) noexcept;

}  // namespace cellerator::compiler::api::v1
