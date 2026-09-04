#include <Cellerator/sdk/expose_parse_and_semantic_analysis_apis_v1.hh>
#include <cassert>
namespace ca = cellerator::compiler::api::v1;
namespace { bool cancel(void* p) noexcept { return *static_cast<bool*>(p); } }
int main() {
    ca::source_document_v1 source{"sample.cell", "cell alpha", 1};
    assert(ca::parse_source_v1(source).tokens.size() == 2);
    ca::semantic_snapshot_v1 semantic;
    assert(ca::analyze_semantics_v1(source, semantic));
    assert(semantic.declared_symbols[0] == "alpha" && !semantic.planning_ran
        && !semantic.code_generation_ran);
    ca::update_source_v1(source, "gene beta");
    assert(ca::analyze_semantics_v1(source, semantic));
    assert(semantic.source_revision == 2 && semantic.declared_symbols[0] == "beta");
    bool stopped = true;
    assert(!ca::analyze_semantics_v1(source, semantic, cancel, &stopped));
}
