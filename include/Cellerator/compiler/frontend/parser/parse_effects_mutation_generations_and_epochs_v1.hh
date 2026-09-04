#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

enum class effect_kind_v1 {
    reads, writes, mutates, preserves, invalidates, advances, publishes,
    canonicalizes, reorders, transfers, allocates, synchronizes, aliases,
    deterministic, pure, opaque
};

enum class generation_mode_v1 { none, automatic, explicit_value };

struct effect_syntax_v1 {
    effect_kind_v1 kind = effect_kind_v1::opaque;
    std::vector<std::string> arguments;
    generation_mode_v1 generation = generation_mode_v1::none;
};

enum class semantic_transition_kind_v1 {
    mutate_structure, mutate_values, mutate_support, mutate_order,
    publish_generation, end_epoch, advance_epoch, assert_generation,
    rebind_identity
};

struct semantic_transition_v1 {
    semantic_transition_kind_v1 kind = semantic_transition_kind_v1::mutate_values;
    std::vector<std::string> arguments;
    parser_source_range_v1 range{};
};

struct effects_parse_v1 {
    std::vector<effect_syntax_v1> effects;
    std::vector<semantic_transition_v1> transitions;
    std::vector<declaration_diagnostic_v1> diagnostics;
    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

[[nodiscard]] effects_parse_v1 parse_effects_and_transitions_v1(
    std::string_view source);

} // namespace Cellerator::compiler::frontend::parser
