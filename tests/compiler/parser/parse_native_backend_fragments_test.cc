#include <Cellerator/compiler/frontend/parser/parse_native_backend_fragments_v1.hh>

#include <cassert>
#include <string>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    const std::string source = R"(
native<generated_cxx> target(host) inputs(graph) outputs(result)
    effects(reads(graph), writes(result)) fallback(reference) {
  result = lower(graph); // a preserved } in a comment
}
native<cuda> target(sm_70) inputs(x, relation<gene, cell>) outputs(y)
    clobbers(memory, condition_codes) fallback(csr_exact) {
  kernel<<<grid, block>>>(x, y); const char *brace = "}";
}
native<ptx> target(sm_70) inputs(%0) outputs(%1) clobbers(memory)
    fallback(cuda_exact) { { mov.u32 %1, %0; } }
native<raw_native> target(custom_v1) inputs(bytes) outputs(out)
    effects(opaque(out)) fallback(portable_exact) { opaque { payload } }
)";
    const auto parsed = parse_native_backend_fragments_v1(source);
    assert(parsed.accepted());
    assert(parsed.fragments.size() == 4);
    assert(parsed.fragments[0].backend == native_backend_kind_v1::generated_cxx);
    assert(parsed.fragments[1].backend == native_backend_kind_v1::cuda);
    assert(parsed.fragments[1].target == "sm_70");
    assert(parsed.fragments[1].inputs.size() == 2);
    assert(parsed.fragments[2].backend == native_backend_kind_v1::ptx);
    assert(parsed.fragments[3].backend == native_backend_kind_v1::raw_native);
    for (const auto &fragment : parsed.fragments) {
        assert(fragment.spelling == source.substr(fragment.range.begin,
                                                  fragment.range.end - fragment.range.begin));
        assert(!fragment.payload.empty());
    }

    assert(!parse_native_backend_fragments_v1(
        "native<cuda> target(sm_70) inputs(x) outputs(y) effects(writes(y)) {").accepted());
    assert(!parse_native_backend_fragments_v1(
        "native<cuda> inputs(x) outputs(y) effects(writes(y)) fallback(exact) {}").accepted());
    assert(!parse_native_backend_fragments_v1(
        "native<spirv> target(vulkan) inputs(x) outputs(y) effects(writes(y)) fallback(exact) {}").accepted());
}
