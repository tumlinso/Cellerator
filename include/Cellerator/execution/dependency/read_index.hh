#pragma once

#include <Cellerator/containers/posting_index.hh>
#include <Cellerator/geometry/dependency/explicit_pattern.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cellerator::execution {

// Prepared source lookup over an ExplicitPattern's immutable reverse postings.
// The pattern remains the sole owner of posting records; this object stores
// only the dense source-key table and range offsets needed by execution.
// Additional condition-only read indexes are separate semantic inputs.
class PreparedReadIndex final {
public:
    explicit PreparedReadIndex(
        std::shared_ptr<const geometry::dependency::ExplicitPattern> pattern);

    const std::shared_ptr<const geometry::dependency::ExplicitPattern>& pattern() const noexcept {
        return pattern_;
    }
    std::size_t source_ordinal(containers::PostingTerm key) const;
    const std::vector<containers::PostingTerm>& sources() const noexcept { return sources_; }
    const std::vector<std::uint64_t>& offsets() const noexcept { return offsets_; }
    const containers::PostingRecord& posting_record(std::size_t index) const;
    std::size_t posting_count() const noexcept { return pattern_->posting_records().size(); }
    containers::PostingIndexView posting_view() const noexcept {
        return pattern_->posting_view();
    }
    // Explicitly materializes a compact uint64 edge-ordinal staging array for
    // a device consumer such as the CUDA frontier. The canonical host records
    // remain owned by ExplicitPattern; retain this returned array through its
    // upload and device-use lifetime. Cost: one host allocation and one 8-byte
    // ordinal copy per posting.
    std::vector<std::uint64_t> make_device_ordinal_projection() const;

private:
    std::shared_ptr<const geometry::dependency::ExplicitPattern> pattern_;
    std::vector<containers::PostingTerm> sources_;
    std::vector<std::uint64_t> offsets_;
};

} // namespace cellerator::execution
