#include <Cellerator/compiler/ir/common/implement_common_type_and_attribute_interning_v1.hh>

#include <cassert>
#include <random>
#include <string>

using namespace cellerator::compiler::ir;

int main() {
    type_attribute_interner interner;
    const auto state = interner.intern(interned_kind::type, "state<f32,gene>", 17u);
    const auto same = interner.intern(interned_kind::type, "state<f32,gene>", 17u);
    assert(state.inserted && !same.inserted);
    assert(state.handle.slot == same.handle.slot);
    assert(interner.serialize(state.handle) == "0:15:state<f32,gene>");

    const auto collision = interner.intern(interned_kind::attribute, "ordered", 17u);
    assert(collision.identity_conflict && !collision.inserted);
    const auto opaque = interner.intern(
        interned_kind::opaque_extension, "x.vendor{bytes=00ff}");
    assert(interner.content(opaque.handle) == "x.vendor{bytes=00ff}");

    std::mt19937_64 random(9u);
    for (unsigned index = 0; index < 5000u; ++index) {
        const auto value = random();
        const auto text = std::string("type.") + std::to_string(value);
        const auto first = interner.intern(interned_kind::type, text);
        const auto repeat = interner.intern(interned_kind::type, text);
        assert(first.handle.slot == repeat.handle.slot);
        assert(interner.content(first.handle) == text);
        assert(!interner.serialize(first.handle).empty());
    }
}
