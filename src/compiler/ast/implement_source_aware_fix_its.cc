#include <Cellerator/compiler/ast/implement_source_aware_fix_its_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::ast {

std::optional<source_fix_v1>
generate_source_fix_v1(const source_fix_request_v1& request, std::uint64_t source_size,
                       std::string* error) {
    const auto fail = [&](std::string message) -> std::optional<source_fix_v1> {
        if (error) *error = std::move(message);
        return std::nullopt;
    };
    if (!request.physical_source || request.macro_expanded)
        return fail("automatic fix-it requires direct physical source");
    if (!request.source.valid() || request.source.end.byte_offset > source_size)
        return fail("fix-it source range is invalid");

    std::string replacement = request.replacement_hint;
    switch (request.kind) {
    case source_fix_kind_v1::missing_pragma:
        if (request.source.begin.byte_offset != 0 || request.source.size_bytes() != 0)
            return fail("missing pragma fix must insert at file start");
        replacement = "#pragma cellerator\n";
        break;
    case source_fix_kind_v1::absent_profile_binding:
        if (replacement.empty()) return fail("profile binding fix requires a profile expression");
        replacement = "given profile " + replacement + ";\n";
        break;
    case source_fix_kind_v1::effect_contract_omission:
        if (replacement.empty()) return fail("effect fix requires an effect list");
        replacement = " effects(" + replacement + ")";
        break;
    default:
        if (replacement.empty()) return fail("source replacement is not known uniquely");
        break;
    }
    if (error) error->clear();
    return source_fix_v1{request.kind, {request.source, std::move(replacement)},
                         request.promises_recompile};
}

std::optional<std::string>
apply_source_fixes_v1(std::string_view source,
                      frontend::source::source_space_id_v1 physical_file,
                      std::vector<source_fix_v1> fixes,
                      repaired_source_validator_v1 validator,
                      void* validator_context,
                      std::string* error) {
    const auto fail = [&](std::string message) -> std::optional<std::string> {
        if (error) *error = std::move(message);
        return std::nullopt;
    };
    if (physical_file == frontend::source::invalid_source_space_v1)
        return fail("physical source identity is invalid");
    bool promises_recompile = false;
    for (const auto& fix : fixes) {
        if (!fix.edit.source.valid() || fix.edit.source.begin.space != physical_file ||
            fix.edit.source.end.byte_offset > source.size())
            return fail("fix-it does not address the physical source buffer");
        promises_recompile |= fix.promises_recompile;
    }
    std::sort(fixes.begin(), fixes.end(), [](const auto& left, const auto& right) {
        return left.edit.source.begin.byte_offset < right.edit.source.begin.byte_offset;
    });
    for (std::size_t index = 1; index < fixes.size(); ++index)
        if (fixes[index - 1].edit.source.end.byte_offset >
            fixes[index].edit.source.begin.byte_offset)
            return fail("fix-it ranges overlap");

    std::string repaired{source};
    for (auto item = fixes.rbegin(); item != fixes.rend(); ++item) {
        const auto begin = item->edit.source.begin.byte_offset;
        repaired.replace(begin, item->edit.source.size_bytes(), item->edit.replacement);
    }
    if (promises_recompile && (!validator || !validator(repaired, validator_context)))
        return fail("repaired source did not pass frontend validation");
    if (error) error->clear();
    return repaired;
}

} // namespace Cellerator::compiler::ast
