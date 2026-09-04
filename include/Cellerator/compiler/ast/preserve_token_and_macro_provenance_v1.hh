#pragma once

#include <Cellerator/compiler/ast/assign_deterministic_source_identities_v1.hh>
#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ast {

enum class provenance_frame_kind_v1 : std::uint8_t {
    token_spelling = 1,
    macro_definition,
    macro_expansion,
    include_expansion,
    physical_file,
    shadow_placeholder,
    generated_source,
};

struct provenance_frame_v1 {
    provenance_frame_kind_v1 kind = provenance_frame_kind_v1::token_spelling;
    frontend::source::source_span_v1 span{};
    // Stable macro, include, placeholder, or generator identity. Zero is
    // permitted only for token spelling and physical-file frames.
    std::uint64_t producer_identity = 0;
};

struct token_provenance_record_v1 {
    compilation_source_identity_v1 token_identity{};
    // Ordered from the immediately observed token toward its physical source.
    std::vector<provenance_frame_v1> trace;
};

class token_provenance_sidecar_v1 {
public:
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const token_provenance_record_v1*
    find(compilation_source_identity_v1 identity) const noexcept;

private:
    std::vector<token_provenance_record_v1> records_;
    friend std::optional<token_provenance_sidecar_v1>
    freeze_token_provenance_v1(std::vector<token_provenance_record_v1>, std::string*);
};

[[nodiscard]] std::optional<token_provenance_sidecar_v1>
freeze_token_provenance_v1(std::vector<token_provenance_record_v1> records,
                           std::string* error = nullptr);

} // namespace Cellerator::compiler::ast
