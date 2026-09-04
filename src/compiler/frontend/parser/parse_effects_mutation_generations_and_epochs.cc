#include <Cellerator/compiler/frontend/parser/parse_effects_mutation_generations_and_epochs_v1.hh>

#include <array>
#include <cctype>
#include <optional>

namespace Cellerator::compiler::frontend::parser {
namespace {

std::string trim_copy(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
        value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        value.remove_suffix(1);
    return std::string(value);
}

std::size_t matching_parenthesis(std::string_view source, std::size_t open) {
    unsigned depth = 0;
    for (auto offset = open; offset < source.size(); ++offset) {
        depth += source[offset] == '(' ? 1u : 0u;
        if (source[offset] == ')' && --depth == 0)
            return offset;
    }
    return std::string_view::npos;
}

std::vector<std::string> split(std::string_view source) {
    std::vector<std::string> result;
    unsigned depth = 0;
    std::size_t begin = 0;
    for (std::size_t offset = 0; offset < source.size(); ++offset) {
        depth += source[offset] == '(' ? 1u : 0u;
        depth -= source[offset] == ')' && depth ? 1u : 0u;
        if (source[offset] == ',' && depth == 0) {
            result.push_back(trim_copy(source.substr(begin, offset - begin)));
            begin = offset + 1;
        }
    }
    auto tail = trim_copy(source.substr(begin));
    if (!tail.empty())
        result.push_back(std::move(tail));
    return result;
}

std::optional<effect_kind_v1> effect_kind(std::string_view name) {
    constexpr std::pair<std::string_view, effect_kind_v1> effects[] = {
        {"reads", effect_kind_v1::reads}, {"writes", effect_kind_v1::writes},
        {"mutates", effect_kind_v1::mutates}, {"preserves", effect_kind_v1::preserves},
        {"invalidates", effect_kind_v1::invalidates}, {"advances", effect_kind_v1::advances},
        {"publishes", effect_kind_v1::publishes}, {"canonicalizes", effect_kind_v1::canonicalizes},
        {"reorders", effect_kind_v1::reorders}, {"transfers", effect_kind_v1::transfers},
        {"allocates", effect_kind_v1::allocates}, {"synchronizes", effect_kind_v1::synchronizes},
        {"aliases", effect_kind_v1::aliases}, {"deterministic", effect_kind_v1::deterministic},
        {"pure", effect_kind_v1::pure}, {"opaque", effect_kind_v1::opaque},
    };
    for (const auto &[spelling, kind] : effects)
        if (name == spelling)
            return kind;
    return std::nullopt;
}

} // namespace

effects_parse_v1 parse_effects_and_transitions_v1(std::string_view source) {
    effects_parse_v1 result;
    const auto effects_begin = source.find("effects(");
    if (effects_begin != std::string_view::npos) {
        const auto open = effects_begin + 7;
        const auto close = matching_parenthesis(source, open);
        if (close == std::string_view::npos) {
            result.diagnostics.push_back({"unterminated effects contract",
                                          {effects_begin, source.size()}});
        } else {
            for (const auto &item : split(source.substr(open + 1, close - open - 1))) {
                const auto argument_open = item.find('(');
                const auto name = std::string_view(item).substr(0, argument_open);
                const auto kind = effect_kind(name);
                if (!kind) {
                    result.diagnostics.push_back({"unknown effect: " + std::string(name),
                                                  {effects_begin, close + 1}});
                    continue;
                }
                effect_syntax_v1 effect;
                effect.kind = *kind;
                if (argument_open != std::string::npos) {
                    if (item.back() != ')') {
                        result.diagnostics.push_back({"malformed effect arguments",
                                                      {effects_begin, close + 1}});
                        continue;
                    }
                    effect.arguments = split(std::string_view(item).substr(
                        argument_open + 1, item.size() - argument_open - 2));
                }
                if (*kind == effect_kind_v1::advances)
                    effect.generation = effect.arguments.size() > 1
                        ? generation_mode_v1::explicit_value : generation_mode_v1::automatic;
                result.effects.push_back(std::move(effect));
            }
        }
    }

    constexpr std::pair<std::string_view, semantic_transition_kind_v1> transitions[] = {
        {"ce::mutate_structure(", semantic_transition_kind_v1::mutate_structure},
        {"ce::mutate_values(", semantic_transition_kind_v1::mutate_values},
        {"ce::mutate_support(", semantic_transition_kind_v1::mutate_support},
        {"ce::mutate_order(", semantic_transition_kind_v1::mutate_order},
        {"ce::publish_generation(", semantic_transition_kind_v1::publish_generation},
        {"ce::end_epoch(", semantic_transition_kind_v1::end_epoch},
        {"ce::advance_epoch(", semantic_transition_kind_v1::advance_epoch},
        {"ce::assert_generation(", semantic_transition_kind_v1::assert_generation},
        {"ce::rebind_identity(", semantic_transition_kind_v1::rebind_identity},
    };
    for (const auto &[needle, kind] : transitions) {
        std::size_t at = 0;
        while ((at = source.find(needle, at)) != std::string_view::npos) {
            const auto open = at + needle.size() - 1;
            const auto close = matching_parenthesis(source, open);
            if (close == std::string_view::npos) {
                result.diagnostics.push_back({"unterminated semantic transition",
                                              {at, source.size()}});
                break;
            }
            result.transitions.push_back({kind,
                split(source.substr(open + 1, close - open - 1)), {at, close + 1}});
            at = close + 1;
        }
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::parser
