#include <Cellerator/compiler/ir/common/implement_removable_provenance_sidecars_v1.hh>

namespace cellerator::compiler::ir {

void provenance_sidecars::set(std::uint32_t operation, provenance_sidecar sidecar) {
    records_.insert_or_assign(operation, std::move(sidecar));
}

const provenance_sidecar *provenance_sidecars::get(
    std::uint32_t operation) const noexcept {
    const auto found = records_.find(operation);
    return found == records_.end() ? nullptr : &found->second;
}

void provenance_sidecars::strip() noexcept { records_.clear(); }

std::uint64_t executable_semantic_hash(
    const std::vector<hot_operation_record> &operations) noexcept {
    std::uint64_t hash = 1469598103934665603ull;
    const auto *bytes = reinterpret_cast<const unsigned char *>(operations.data());
    for (std::size_t index = 0; index < operations.size() * sizeof(hot_operation_record); ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

} // namespace cellerator::compiler::ir
