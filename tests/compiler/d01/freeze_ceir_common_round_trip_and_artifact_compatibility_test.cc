#include <Cellerator/compiler/ir/common/freeze_ceir_common_round_trip_and_artifact_compatibility_v1.hh>

#include <cassert>

using namespace cellerator::compiler::ir;

int main() {
    std::vector<common_operation> corpus;
    for (unsigned index = 0; index < 64u; ++index) {
        common_operation operation;
        operation.namespace_name = "semantic";
        operation.operation_name = "op_" + std::to_string(index);
        operation.attributes = {{"index", std::to_string(index)}, {"order", "@genes"}};
        operation.unknown_extensions = {{"x.vendor", {
            static_cast<std::uint8_t>(index), 0u, 0xffu}}};
        corpus.push_back(std::move(operation));
    }
    const auto first = verify_common_round_trip(corpus);
    const auto second = verify_common_round_trip(corpus);
    assert(first.text_stable && first.binary_valid && first.binary_payload_equal);
    assert(first.unknown_extensions_preserved && first.standalone_resumed);
    assert(first.source_inline_parsed && first.diagnostic.empty());
    assert(first.canonical_hash == second.canonical_hash);
}
