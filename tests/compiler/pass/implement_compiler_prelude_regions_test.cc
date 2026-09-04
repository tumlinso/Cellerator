#include <Cellerator/compiler/pass/implement_compiler_prelude_regions_v1.hh>

#include <cassert>

namespace cp = cellerator::compiler::pass::v1;

int main() {
    const std::vector<cp::source_region_v1> regions{
        {false, {{cp::prelude_declaration_kind_v1::ordinary, "main", {}}}},
        {true, {{cp::prelude_declaration_kind_v1::transform, "normalize", {"schema"}}}},
        {true, {{cp::prelude_declaration_kind_v1::extension_schema, "schema", {}}}},
    };
    const auto receipt = cp::resolve_compiler_preludes_v1(regions);
    assert(receipt.status == cp::prelude_resolution_status_v1::success);
    assert(receipt.prelude_symbols.size() == 2);
    assert(receipt.ordinary_declarations.size() == 1
        && receipt.ordinary_declarations[0].name == "main");

    const std::vector<cp::source_region_v1> isolated{
        {false, {{cp::prelude_declaration_kind_v1::ordinary, "ordinary_only", {}}}},
        {true, {{cp::prelude_declaration_kind_v1::transform,
            "invalid_transform", {"ordinary_only"}}}},
    };
    assert(cp::resolve_compiler_preludes_v1(isolated).status
        == cp::prelude_resolution_status_v1::unresolved_prelude_reference);
}
