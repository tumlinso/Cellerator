#include <Cellerator/compiler/ast/preserve_token_and_macro_provenance_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::ast {
namespace {

bool less_identity(compilation_source_identity_v1 left,
                   compilation_source_identity_v1 right) noexcept {
    return left.high < right.high || (left.high == right.high && left.low < right.low);
}

bool requires_producer(provenance_frame_kind_v1 kind) noexcept {
    return kind != provenance_frame_kind_v1::token_spelling &&
           kind != provenance_frame_kind_v1::physical_file;
}

} // namespace

std::size_t token_provenance_sidecar_v1::size() const noexcept { return records_.size(); }

const token_provenance_record_v1*
token_provenance_sidecar_v1::find(compilation_source_identity_v1 identity) const noexcept {
    const auto found = std::lower_bound(
        records_.begin(), records_.end(), identity,
        [](const auto& record, auto sought) { return less_identity(record.token_identity, sought); });
    return found != records_.end() && found->token_identity == identity ? &*found : nullptr;
}

std::optional<token_provenance_sidecar_v1>
freeze_token_provenance_v1(std::vector<token_provenance_record_v1> records,
                           std::string* error) {
    auto fail = [&](std::string message) -> std::optional<token_provenance_sidecar_v1> {
        if (error) *error = std::move(message);
        return std::nullopt;
    };
    for (const auto& record : records) {
        if ((record.token_identity.high | record.token_identity.low) == 0 || record.trace.empty())
            return fail("token provenance requires an identity and a trace");
        if (record.trace.front().kind != provenance_frame_kind_v1::token_spelling &&
            record.trace.front().kind != provenance_frame_kind_v1::generated_source)
            return fail("trace must begin at token spelling or generated source");
        if (record.trace.back().kind != provenance_frame_kind_v1::physical_file)
            return fail("trace must terminate at a physical file");
        for (const auto& frame : record.trace) {
            if (!frame.span.valid()) return fail("provenance frame has an invalid source span");
            if (requires_producer(frame.kind) && frame.producer_identity == 0)
                return fail("derived provenance frame requires a producer identity");
        }
    }
    std::sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
        return less_identity(left.token_identity, right.token_identity);
    });
    for (std::size_t index = 1; index < records.size(); ++index)
        if (records[index - 1].token_identity == records[index].token_identity)
            return fail("duplicate token provenance identity");
    token_provenance_sidecar_v1 sidecar;
    sidecar.records_ = std::move(records);
    if (error) error->clear();
    return sidecar;
}

} // namespace Cellerator::compiler::ast
