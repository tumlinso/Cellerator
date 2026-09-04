#include <Cellerator/compiler/ast/assign_deterministic_source_identities_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ast;

int main() {
    const source_identity_input_v1 original{11, 22, 4096, 33, 1};
    const auto stable = derive_source_identity_v1(original);
    assert(stable == derive_source_identity_v1(original));

    auto semantic_edit = original;
    ++semantic_edit.declaration_identity;
    assert(!(stable == derive_source_identity_v1(semantic_edit)));
    auto moved = original;
    ++moved.canonical_offset;
    assert(!(stable == derive_source_identity_v1(moved)));
    auto revised = original;
    ++revised.language_revision;
    assert(!(stable == derive_source_identity_v1(revised)));

    std::string error;
    const persistent_user_identity_v1 persistent{77, 88};
    auto first = assign_source_identities_v1(
        {original}, {{7, 1}}, {persistent}, &error);
    auto reparsed = assign_source_identities_v1(
        {original}, {{19, 900}}, {persistent}, &error);
    assert(first && reparsed && error.empty());
    assert((*first)[0].source_identity == (*reparsed)[0].source_identity);
    assert(!((*first)[0].transient_node == (*reparsed)[0].transient_node));
    assert((*first)[0].persistent_identity->high == 77);

    assert(!assign_source_identities_v1({original}, {}, {}, &error));
    assert(!assign_source_identities_v1(
        {original, original}, {{7, 1}, {7, 2}}, {persistent, std::nullopt}, &error));

    std::cout << "source_identity_high=" << stable.high
              << " source_identity_low=" << stable.low << '\n';
}
