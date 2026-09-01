#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>

#include <cassert>

namespace resume = cellerator::execution::lowering_resumption;

int main() {
    const unsigned char bytes[] = {1u, 2u, 3u};
    const resume::lowering_identity_context_v1 context{
        {1u, 1u}, 2u, {3u, 1u}, {4u, 1u}, {5u, 1u}, 0u};
    resume::canonical_source_input_v1 source{};
    source.source_identity = {6u, 1u};
    source.bytes = bytes;
    source.byte_count = sizeof(bytes);
    source.content_hash[0] = 7u;
    resume::resumption_cursor_v1 cursor{};
    assert(resume::resume_from_canonical_source_v1(source, context, &cursor));
    assert(cursor.stage == resume::lowering_stage_v1::canonical_source);
    assert(cursor.payload == bytes && cursor.payload_bytes == sizeof(bytes));

    source.content_hash[0] = 0u;
    assert(resume::resume_from_canonical_source_v1(
               source, context, &cursor).code ==
        resume::compatibility_code_v1::invalid_argument);
}
