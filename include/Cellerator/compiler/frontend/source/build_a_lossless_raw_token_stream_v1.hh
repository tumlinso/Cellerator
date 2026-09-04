#pragma once

#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::source {

struct raw_token_v1 {
    std::string leading_trivia;
    std::string spelling;
    source_span_v1 span{};
    std::optional<source_span_v1> macro_origin;
    bool dialect_active = false;
    std::uint64_t preprocessor_condition = 0;
};

struct raw_token_stream_v1 {
    source_space_id_v1 source = invalid_source_space_v1;
    std::vector<raw_token_v1> tokens;
    std::string trailing_trivia;
};

[[nodiscard]] raw_token_stream_v1 build_raw_token_stream_v1(
    source_space_id_v1 source, std::string_view bytes, std::uint64_t activation_offset,
    std::uint64_t preprocessor_condition = 0);
[[nodiscard]] std::string reconstruct_raw_token_stream_v1(const raw_token_stream_v1& stream);
[[nodiscard]] bool has_exact_byte_coverage_v1(const raw_token_stream_v1& stream,
                                              std::uint64_t source_size) noexcept;

} // namespace Cellerator::compiler::frontend::source
