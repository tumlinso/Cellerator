#include <Cellerator/compiler/sema/implement_operation_kind_resolution_v1.hh>

#include <cassert>
#include <set>

int main() {
    using namespace cellerator::compiler::sema::v1;
    assert(operation_kind_coverage_count() == 14u);
    std::set<source_operation_kind> covered;
    for (std::uint32_t i = 0; i < operation_kind_coverage_count(); ++i) {
        const auto &entry = operation_kind_coverage_table()[i];
        assert(entry.syntax != nullptr && entry.syntax[0] != '\0');
        assert(covered.insert(entry.source).second);
        assert(resolve_operation_kind(entry.source) == &entry);
    }
    assert(resolve_operation_kind(source_operation_kind::relation_apply)->core ==
        cellerator::compute::operation::v2::operation_kind::relation_apply);
    assert(resolve_operation_kind(source_operation_kind::relation_chain)->requires_composite_lowering);
}
