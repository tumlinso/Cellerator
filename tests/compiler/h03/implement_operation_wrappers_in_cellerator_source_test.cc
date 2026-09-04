#include <cassert>
enum class kind : unsigned { relation, transpose, contraction, segment, gate, update, bundle, chain, moments, hierarchy, exchange };
struct operation { kind value; unsigned inputs; unsigned outputs; };
constexpr operation make(kind value, unsigned inputs, unsigned outputs) { return {value, inputs, outputs}; }
int main() {
    constexpr auto wrapped = make(kind::contraction, 2, 1);
    constexpr operation handwritten{kind::contraction, 2, 1};
    static_assert(wrapped.value == handwritten.value && wrapped.inputs == handwritten.inputs && wrapped.outputs == handwritten.outputs);
    assert(sizeof(wrapped) == sizeof(handwritten));
}
