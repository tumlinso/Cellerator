#include <Cellerator/compiler/pass/implement_compiler_prelude_regions_v1.hh>

#include <algorithm>

namespace cellerator::compiler::pass::v1 {

prelude_resolution_receipt_v1 resolve_compiler_preludes_v1(
    const std::vector<source_region_v1>& regions) {
    prelude_resolution_receipt_v1 receipt;
    for (const auto& region : regions) {
        for (const auto& declaration : region.declarations) {
            if (declaration.name.empty()
                || (region.compiler_prelude
                    && declaration.kind == prelude_declaration_kind_v1::ordinary)
                || (!region.compiler_prelude
                    && declaration.kind != prelude_declaration_kind_v1::ordinary)) {
                receipt.status = prelude_resolution_status_v1::invalid_declaration;
                receipt.diagnostic = "declaration kind does not match its source region";
                return receipt;
            }
            if (!region.compiler_prelude) receipt.ordinary_declarations.push_back(declaration);
        }
    }
    // Collect every prelude declaration before resolving references, regardless of
    // its textual position relative to the ordinary compilation graph.
    for (const auto& region : regions) if (region.compiler_prelude) {
        for (const auto& declaration : region.declarations) {
            if (std::find(receipt.prelude_symbols.begin(), receipt.prelude_symbols.end(),
                    declaration.name) != receipt.prelude_symbols.end()) {
                receipt.status = prelude_resolution_status_v1::duplicate_prelude_symbol;
                receipt.diagnostic = declaration.name;
                return receipt;
            }
            receipt.prelude_symbols.push_back(declaration.name);
        }
    }
    for (const auto& region : regions) if (region.compiler_prelude) {
        for (const auto& declaration : region.declarations) {
            for (const auto& reference : declaration.references) {
                if (std::find(receipt.prelude_symbols.begin(), receipt.prelude_symbols.end(),
                        reference) == receipt.prelude_symbols.end()) {
                    receipt.status = prelude_resolution_status_v1::unresolved_prelude_reference;
                    receipt.diagnostic = reference;
                    return receipt;
                }
            }
        }
    }
    return receipt;
}

}  // namespace cellerator::compiler::pass::v1
