#include <Cellerator/execution/dependency/read_index.hh>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cellerator::execution {
namespace {
bool less(const containers::PostingTerm& a,
          const containers::PostingTerm& b) noexcept {
    return a.role < b.role || (a.role == b.role && a.entity_id < b.entity_id);
}
} // namespace

PreparedReadIndex::PreparedReadIndex(
    std::shared_ptr<const geometry::dependency::ExplicitPattern> pattern)
    : pattern_(std::move(pattern)), offsets_{0} {
    if (!pattern_) throw std::invalid_argument("read index pattern cannot be null");
    const auto& postings = pattern_->posting_records();
    sources_.reserve(postings.size());
    for (std::size_t i = 0; i < postings.size(); ++i) {
        const auto& record = postings[i];
        const containers::PostingTerm key{record.role, 0, record.entity_id};
        if (record.reserved != 0)
            throw std::invalid_argument("pattern posting reserved field must be zero");
        if (sources_.empty() || less(sources_.back(), key)) {
            if (!sources_.empty()) offsets_.push_back(i);
            sources_.push_back(key);
        } else if (less(key, sources_.back())) {
            throw std::invalid_argument("pattern postings are not ordered by source");
        }
    }
    if (!sources_.empty()) offsets_.push_back(postings.size());
}

std::size_t PreparedReadIndex::source_ordinal(containers::PostingTerm key) const {
    if (key.reserved) throw std::invalid_argument("source reserved field must be zero");
    const auto p = std::lower_bound(sources_.begin(), sources_.end(), key, less);
    if (p == sources_.end() || less(key, *p))
        throw std::out_of_range("unknown prepared read source");
    return static_cast<std::size_t>(p - sources_.begin());
}

const containers::PostingRecord&
PreparedReadIndex::posting_record(std::size_t index) const {
    return pattern_->posting_records().at(index);
}

std::vector<std::uint64_t> PreparedReadIndex::make_device_ordinal_projection() const {
    std::vector<std::uint64_t> result;
    result.reserve(posting_count());
    for (const auto& record : pattern_->posting_records())
        result.push_back(record.logical_edge_ordinal);
    return result;
}

} // namespace cellerator::execution
