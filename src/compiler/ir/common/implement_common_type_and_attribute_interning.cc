#include <Cellerator/compiler/ir/common/implement_common_type_and_attribute_interning_v1.hh>

#include <charconv>

namespace cellerator::compiler::ir {
namespace {
std::string key(interned_kind kind, std::string_view content) {
    return std::to_string(static_cast<unsigned>(kind)) + ":" + std::string(content);
}
} // namespace

intern_result type_attribute_interner::intern(interned_kind kind,
    std::string_view canonical_content, std::optional<std::uint64_t> identity) {
    const auto canonical_key = key(kind, canonical_content);
    if (const auto found = canonical_.find(canonical_key); found != canonical_.end()) {
        const auto &record = records_[found->second];
        return {{found->second, kind}, false,
            identity && record.identity && identity != record.identity};
    }
    if (identity) {
        if (const auto found = asserted_.find(*identity); found != asserted_.end()) {
            const auto &record = records_[found->second];
            if (record.kind != kind || record.content != canonical_content)
                return {{found->second, record.kind}, false, true};
        }
    }
    const auto slot = static_cast<std::uint32_t>(records_.size());
    records_.push_back({kind, std::string(canonical_content), identity});
    canonical_.emplace(canonical_key, slot);
    if (identity)
        asserted_.emplace(*identity, slot);
    return {{slot, kind}, true, false};
}

std::string_view type_attribute_interner::content(interned_handle handle) const noexcept {
    if (handle.slot >= records_.size() || records_[handle.slot].kind != handle.kind)
        return {};
    return records_[handle.slot].content;
}

std::optional<std::uint64_t> type_attribute_interner::asserted_identity(
    interned_handle handle) const noexcept {
    if (handle.slot >= records_.size() || records_[handle.slot].kind != handle.kind)
        return std::nullopt;
    return records_[handle.slot].identity;
}

std::string type_attribute_interner::serialize(interned_handle handle) const {
    if (handle.slot >= records_.size() || records_[handle.slot].kind != handle.kind)
        return {};
    const auto &record = records_[handle.slot];
    return std::to_string(static_cast<unsigned>(record.kind)) + ":"
        + std::to_string(record.content.size()) + ":" + record.content;
}

} // namespace cellerator::compiler::ir
