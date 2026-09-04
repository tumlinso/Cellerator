#include <Cellerator/compiler/ast/preserve_token_and_macro_provenance_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ast;
using namespace Cellerator::compiler::frontend::source;

static source_span_v1 span(source_space_id_v1 space, std::uint64_t begin,
                           std::uint64_t end) {
    return {{space, begin}, {space, end}};
}

int main() {
    const compilation_source_identity_v1 identity{1, 2};
    token_provenance_record_v1 nested{
        identity,
        {{provenance_frame_kind_v1::generated_source, span(8, 0, 5), 800},
         {provenance_frame_kind_v1::shadow_placeholder, span(7, 10, 15), 700},
         {provenance_frame_kind_v1::macro_expansion, span(6, 20, 25), 601},
         {provenance_frame_kind_v1::macro_definition, span(5, 30, 35), 501},
         {provenance_frame_kind_v1::macro_expansion, span(4, 40, 45), 401},
         {provenance_frame_kind_v1::macro_definition, span(3, 50, 55), 301},
         {provenance_frame_kind_v1::include_expansion, span(2, 60, 65), 201},
         {provenance_frame_kind_v1::physical_file, span(1, 70, 75), 0}}};
    std::string error;
    auto sidecar = freeze_token_provenance_v1({nested}, &error);
    assert(sidecar && error.empty());
    const auto* traced = sidecar->find(identity);
    assert(traced && traced->trace.size() == 8);
    assert(traced->trace[2].kind == provenance_frame_kind_v1::macro_expansion);
    assert(traced->trace[5].kind == provenance_frame_kind_v1::macro_definition);
    assert(traced->trace.back().span.begin.space == 1);

    auto no_physical = nested;
    no_physical.trace.pop_back();
    assert(!freeze_token_provenance_v1({no_physical}, &error));
    auto missing_producer = nested;
    missing_producer.trace[1].producer_identity = 0;
    assert(!freeze_token_provenance_v1({missing_producer}, &error));
    assert(!freeze_token_provenance_v1({nested, nested}, &error));

    std::cout << "tokens=" << sidecar->size()
              << " nested_trace_frames=" << traced->trace.size() << '\n';
}
