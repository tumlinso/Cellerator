#include <Cellerator/compiler/pass/implement_extension_capability_negotiation_v1.hh>
#include <Cellerator/compiler/pass/implement_unknown_extension_preservation_and_forwarding_v1.hh>

#include <cassert>

namespace cp = cellerator::compiler::pass::v1;

int main() {
    const std::uint32_t required = cp::extension_reflection_v1 | cp::extension_lowering_v1;
    const auto understood = cp::negotiate_extension_capability_v1(
        {"future.splice", required, required, required, false});
    assert(understood.mode == cp::extension_handling_mode_v1::fully_understood);
    const auto external = cp::negotiate_extension_capability_v1(
        {"future.splice", required, required, cp::extension_reflection_v1, true});
    assert(external.mode == cp::extension_handling_mode_v1::external_lowered);
    const auto inspect = cp::negotiate_extension_capability_v1(
        {"future.splice", required, cp::extension_reflection_v1, 0, false});
    assert(inspect.mode == cp::extension_handling_mode_v1::inspect_only);
    const auto preserve = cp::negotiate_extension_capability_v1(
        {"future.splice", required, 0, 0, false});
    assert(preserve.mode == cp::extension_handling_mode_v1::preserve_only);
    cp::opaque_extension_node_v1 node;
    assert(cp::parse_unknown_extension_v1(9, "future.splice", "future.splice %x", node));
    assert(cp::print_unknown_extension_v1(node) == "future.splice %x");
}
