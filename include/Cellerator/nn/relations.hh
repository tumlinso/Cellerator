#pragma once

#include <Cellerator/biology/coordinates.hh>

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cellerator::nn {

// Borrowed primitive CSR view. Preparation copies both arrays before
// returning, so callers retain their own allocation and lifetime policy.
template<class Index = std::int64_t>
struct CsrView {
    const Index* row_ptr = nullptr;
    std::size_t row_ptr_count = 0;
    const Index* col_indices = nullptr;
    std::size_t edge_count = 0;
};

struct CsrSupport {
    // Rows are targets; columns are source occurrence positions.
    std::vector<std::int64_t> row_ptr;
    std::vector<std::int64_t> col_indices;
};

// An immutable CSR relation bound to exact source and target occurrence
// declarations.  Host structure is copied during preparation, while device
// buffers are realized explicitly on the requested device.  Duplicate edges
// and empty target neighborhoods are retained.
class PreparedRelation {
public:
    static PreparedRelation from_csr(
        CsrSupport support,
        const biology::OccurrenceOrderToken& source_order,
        const biology::OccurrenceOrderToken& target_order,
        std::uint64_t structural_generation = 0,
        at::Device device = at::Device(at::kCPU));

    template<class Index,
             std::enable_if_t<std::is_integral<Index>::value
                                  && !std::is_same<Index, bool>::value,
                              int> = 0>
    static PreparedRelation from_csr(
        CsrView<Index> support,
        const biology::OccurrenceOrderToken& source_order,
        const biology::OccurrenceOrderToken& target_order,
        std::uint64_t structural_generation = 0,
        at::Device device = at::Device(at::kCPU)) {
        static_assert(sizeof(Index) <= sizeof(std::int64_t),
                      "CSR view indices must not exceed int64 width");
        if (support.row_ptr_count != 0 && support.row_ptr == nullptr) {
            throw std::invalid_argument("CSR row_ptr view has a null data pointer");
        }
        if (support.edge_count != 0 && support.col_indices == nullptr) {
            throw std::invalid_argument("CSR col_indices view has a null data pointer");
        }
        CsrSupport owned;
        owned.row_ptr.reserve(support.row_ptr_count);
        owned.col_indices.reserve(support.edge_count);
        const auto checked_index = [](Index value) {
            if constexpr (std::is_unsigned<Index>::value) {
                if (value > static_cast<std::make_unsigned_t<std::int64_t>>(
                                std::numeric_limits<std::int64_t>::max())) {
                    throw std::overflow_error("CSR index exceeds int64 range");
                }
            }
            return static_cast<std::int64_t>(value);
        };
        for (std::size_t i = 0; i < support.row_ptr_count; ++i) {
            owned.row_ptr.push_back(checked_index(support.row_ptr[i]));
        }
        for (std::size_t i = 0; i < support.edge_count; ++i) {
            owned.col_indices.push_back(checked_index(support.col_indices[i]));
        }
        return from_csr(std::move(owned), source_order, target_order,
                        structural_generation, device);
    }

    std::int64_t source_count() const noexcept { return n_sources_; }
    std::int64_t target_count() const noexcept { return n_targets_; }
    std::int64_t edge_count() const {
        validate_storage();
        return static_cast<std::int64_t>(host_col_indices_.size());
    }
    std::uint64_t generation() const noexcept { return structural_generation_; }
    at::Device device() const noexcept { return device_; }

    const biology::OccurrenceOrderToken& source_order_token() const noexcept {
        return source_order_;
    }
    const biology::OccurrenceOrderToken& target_order_token() const noexcept {
        return target_order_;
    }
    const std::vector<std::int64_t>& source_offsets() const {
        return source_order_.group_offsets();
    }
    const std::vector<std::int64_t>& target_offsets() const {
        return target_order_.group_offsets();
    }
    const std::optional<biology::AxisDescriptor>& source_axis() const {
        return source_order_.axis();
    }
    const std::optional<biology::AxisDescriptor>& target_axis() const {
        return target_order_.axis();
    }

    // Device tensors are borrowed read-only structure.  They remain owned by
    // this object and are not reconstructed or content-checked per forward.
    const at::Tensor& row_ptr() const { validate_storage(); return row_ptr_; }
    const at::Tensor& col_indices() const { validate_storage(); return col_indices_; }
    const at::Tensor& rows() const { validate_storage(); return rows_; }
    const at::Tensor& causal_mask() const { validate_storage(); return causal_mask_; }

    const std::vector<std::int64_t>& host_row_ptr() const {
        validate_storage();
        return host_row_ptr_;
    }
    const std::vector<std::int64_t>& host_col_indices() const {
        validate_storage();
        return host_col_indices_;
    }
    const std::vector<std::int64_t>& host_rows() const {
        validate_storage();
        return host_rows_;
    }
    const std::vector<std::uint8_t>& host_causal_mask() const {
        validate_storage();
        return host_causal_mask_;
    }

    // Check ATen version counters only.  This catches in-place mutation of a
    // borrowed Tensor handle without reading device contents or copying data.
    void validate_storage() const;
    bool has_future_edges() const;

    // Remove future edges once, on the host.  The returned relation retains
    // endpoint tokens and group declarations and is explicitly realized on
    // this relation's current device.
    PreparedRelation causal() const;

    PreparedRelation to(at::Device device) const;

    void validate_for(
        const biology::OccurrenceOrderToken& source_order,
        const biology::OccurrenceOrderToken& target_order,
        std::int64_t source_count,
        std::int64_t target_count,
        at::Device device) const;

    void validate_for(
        const biology::OccurrenceOrderToken& source_order,
        const biology::OccurrenceOrderToken& target_order,
        std::int64_t source_count,
        std::int64_t target_count,
        std::uint64_t structural_generation,
        at::Device device) const;

    PreparedRelation transpose() const;

private:
    PreparedRelation() = default;

    static PreparedRelation prepare(
        CsrSupport support,
        const biology::OccurrenceOrderToken& source_order,
        const biology::OccurrenceOrderToken& target_order,
        std::uint64_t structural_generation,
        at::Device device);

    std::int64_t n_sources_ = 0;
    std::int64_t n_targets_ = 0;
    std::uint64_t structural_generation_ = 0;
    at::Device device_ = at::Device(at::kCPU);
    biology::OccurrenceOrderToken source_order_;
    biology::OccurrenceOrderToken target_order_;
    std::vector<std::int64_t> host_row_ptr_;
    std::vector<std::int64_t> host_col_indices_;
    std::vector<std::int64_t> host_rows_;
    std::vector<std::uint8_t> host_causal_mask_;
    at::Tensor row_ptr_;
    at::Tensor col_indices_;
    at::Tensor rows_;
    at::Tensor causal_mask_;
    std::int64_t row_ptr_version_ = -1;
    std::int64_t col_indices_version_ = -1;
    std::int64_t rows_version_ = -1;
    std::int64_t causal_mask_version_ = -1;
};

} // namespace cellerator::nn
