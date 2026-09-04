#include <Cellerator/compiler/ast/implement_source_aware_fix_its_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ast;
using namespace Cellerator::compiler::frontend::source;

static bool equals_expected(std::string_view source, void* context) noexcept {
    return source == *static_cast<const std::string*>(context);
}

static source_span_v1 insertion(std::uint64_t offset) { return {{1, offset}, {1, offset}}; }
static source_span_v1 range(std::uint64_t begin, std::uint64_t end) {
    return {{1, begin}, {1, end}};
}

int main() {
    struct fixture { std::string broken; source_fix_request_v1 request; std::string expected; };
    const std::vector<fixture> fixtures{
        {"domain gene;\n", {source_fix_kind_v1::missing_pragma, insertion(0)},
         "#pragma cellerator\ndomain gene;\n"},
        {"field f() [[ x ]]", {source_fix_kind_v1::malformed_field_delimiter, range(10, 12), "<["},
         "field f() <[ x ]]"},
        {"cell -[r]-> cell", {source_fix_kind_v1::relation_endpoint_mismatch, range(12, 16), "gene"},
         "cell -[r]-> gene"},
        {"field f() <[ ]>", {source_fix_kind_v1::absent_profile_binding, insertion(13), "pbmc"},
         "field f() <[ given profile pbmc;\n]>"},
        {"native f()", {source_fix_kind_v1::effect_contract_omission, insertion(10), "reads(x)"},
         "native f() effects(reads(x))"},
        {"old_relation", {source_fix_kind_v1::deprecated_syntax, range(0, 12), "relation"},
         "relation"}};

    std::string error;
    for (const auto& fixture : fixtures) {
        auto fix = generate_source_fix_v1(fixture.request, fixture.broken.size(), &error);
        assert(fix && error.empty());
        auto repaired = apply_source_fixes_v1(fixture.broken, 1, {*fix}, equals_expected,
                                              const_cast<std::string*>(&fixture.expected), &error);
        assert(repaired && *repaired == fixture.expected && error.empty());
    }
    auto macro_request = fixtures[2].request;
    macro_request.macro_expanded = true;
    assert(!generate_source_fix_v1(macro_request, fixtures[2].broken.size(), &error));
    const source_fix_v1 overlap_a{source_fix_kind_v1::deprecated_syntax, {range(0, 3), "a"}, false};
    const source_fix_v1 overlap_b{source_fix_kind_v1::deprecated_syntax, {range(2, 4), "b"}, false};
    assert(!apply_source_fixes_v1("abcdef", 1, {overlap_a, overlap_b}, nullptr, nullptr, &error));

    std::cout << "recompiled_fixtures=" << fixtures.size() << '\n';
}
