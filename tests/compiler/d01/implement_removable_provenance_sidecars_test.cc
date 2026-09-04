#include <Cellerator/compiler/ir/common/implement_removable_provenance_sidecars_v1.hh>

#include <cassert>
#include <type_traits>

using namespace cellerator::compiler::ir;

int main() {
    static_assert(std::is_trivially_copyable<hot_operation_record>::value);
    static_assert(sizeof(hot_operation_record) == 24u);
    std::vector<hot_operation_record> hot{{7u, 0u, 2u, 2u, 1u, 0u}};
    const auto size_before = hot.size() * sizeof(hot_operation_record);
    const auto hash_before = executable_semantic_hash(hot);
    provenance_sidecars cold;
    cold.set(0u, {{"model.cc", 10u, 40u}, {1u, 2u}, {"profile:pbmc"},
        {"selected:csr"}, {"cuda:kernel_7"}});
    assert(cold.get(0u)->transform_lineage.size() == 2u);
    cold.strip();
    assert(cold.size() == 0u && cold.get(0u) == nullptr);
    assert(hot.size() * sizeof(hot_operation_record) == size_before);
    assert(executable_semantic_hash(hot) == hash_before);
}
