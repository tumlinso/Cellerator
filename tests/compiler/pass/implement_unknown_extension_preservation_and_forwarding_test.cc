#include <Cellerator/compiler/pass/implement_unknown_extension_preservation_and_forwarding_v1.hh>

#include <cassert>

namespace cp = cellerator::compiler::pass::v1;

int main() {
    const std::string newer_ir = R"(future.splice %x {mode = #future<"lossless">})";
    cp::opaque_extension_node_v1 parsed;
    assert(cp::parse_unknown_extension_v1(7, "future.splice", newer_ir, parsed));
    parsed.opaque_payload = {0, 255, 17, 0};
    assert(cp::print_unknown_extension_v1(parsed) == newer_ir);
    const auto clone = cp::clone_unknown_extension_v1(parsed);
    assert(clone.exact_text == newer_ir && clone.opaque_payload == parsed.opaque_payload);
    const auto bytes = cp::serialize_unknown_extension_v1(clone);
    cp::opaque_extension_node_v1 restored;
    assert(cp::deserialize_unknown_extension_v1(bytes, restored));
    assert(restored.ir_level == 7 && restored.qualified_name == "future.splice");
    assert(restored.exact_text == newer_ir && restored.opaque_payload == parsed.opaque_payload);
    std::vector<cp::opaque_extension_node_v1> imported;
    assert(cp::forward_unknown_extension_v1(restored, imported));
    assert(imported.size() == 1 && cp::print_unknown_extension_v1(imported[0]) == newer_ir);
}
