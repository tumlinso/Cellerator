#pragma once

#include <Cellerator/cell.hh>
#include <Cellerator/memory/view.hh>

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace cellerator::cells {

// Feature identities are borrowed host metadata in the declared biological
// order; they are not inferred from numeric columns or physical row positions.
// FeatureIdentity or an additional Cell capability can carry domain identity.
template<class Scalar, class FeatureIdentity>
struct ExpressionView {
    memory::const_array_view<Scalar> values;
    std::span<const FeatureIdentity> feature_ids;
};

// Only values are writable. The caller's storage owner retains responsibility
// for synchronization and for publishing the correct value generation after
// changes. Cell never mutates identity, order, layout, or allocation.
template<class Scalar, class FeatureIdentity>
struct MutableExpressionView {
    memory::array_view<Scalar> values;
    std::span<const FeatureIdentity> feature_ids;
};

namespace detail {

template<class Descriptor, class Scalar, class FeatureIdentity, class Identity>
class ExpressionRowBase {
public:
    ExpressionRowBase(Descriptor& matrix, std::span<const FeatureIdentity> feature_ids,
                      std::size_t row, Identity identity)
        : matrix_(&matrix), feature_ids_(feature_ids), row_(row),
          identity_(std::move(identity)) {
        validate(matrix, feature_ids, row);
    }

    ExpressionRowBase(Descriptor&&, std::span<const FeatureIdentity>,
                      std::size_t, Identity) = delete;

    std::size_t row() const noexcept { return row_; }
    const Identity& identity() const & noexcept { return identity_; }
    const Identity& identity() const && = delete;

    // The new selection and feature axis are validated before changing the
    // handle. The owner remains responsible for declared biological identity.
    void rebind(Descriptor& matrix, std::span<const FeatureIdentity> feature_ids,
                std::size_t row) & {
        validate(matrix, feature_ids, row);
        matrix_ = &matrix;
        feature_ids_ = feature_ids;
        row_ = row;
    }

    void rebind(Descriptor&&, std::span<const FeatureIdentity>, std::size_t) & = delete;

    ExpressionView<Scalar, FeatureIdentity>
    read(ViewTag<ExpressionView<Scalar, FeatureIdentity>>) const & {
        const auto& matrix = *matrix_;
        validate(matrix, feature_ids_, row_);
        const Scalar* data = matrix.data;
        // No pointer arithmetic on a null or empty payload. A row of an empty
        // feature axis has no position-dependent storage to access.
        if (matrix.columns != 0 && row_ != 0 && matrix.row_stride != 0)
            data += row_ * matrix.row_stride;
        return {{data, matrix.columns, matrix.where}, feature_ids_};
    }
    ExpressionView<Scalar, FeatureIdentity>
    read(ViewTag<ExpressionView<Scalar, FeatureIdentity>>) const && = delete;

    CellIdentityView<Identity> read(ViewTag<CellIdentityView<Identity>>) const & {
        return {identity_};
    }
    CellIdentityView<Identity> read(ViewTag<CellIdentityView<Identity>>) const && = delete;

    template<class View> static constexpr bool supports() noexcept {
        return cellerator::detail::can_read<ExpressionRowBase, View>::value;
    }
    template<class View> static constexpr bool supports_edit() noexcept {
        return cellerator::detail::can_write<ExpressionRowBase, View>::value;
    }
    template<class View, std::enable_if_t<
        cellerator::detail::can_read<ExpressionRowBase, View>::value, int> = 0>
    View view() const & { return read(ViewTag<View>{}); }
    template<class View> void view() const && = delete;

protected:
    static void validate(const Descriptor& matrix,
                         std::span<const FeatureIdentity> feature_ids,
                         std::size_t row) {
        if (matrix.columns != feature_ids.size())
            throw std::invalid_argument("expression feature count does not match matrix columns");
        if (row >= matrix.rows)
            throw std::out_of_range("expression row is outside the matrix");
        if (matrix.row_stride < matrix.columns)
            throw std::invalid_argument("expression row stride is smaller than its width");
        if (matrix.columns != 0 && matrix.data == nullptr)
            throw std::invalid_argument("expression matrix has a null data pointer");
        const auto max_bytes = std::numeric_limits<std::size_t>::max() / sizeof(Scalar);
        const auto max_difference = static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max());
        const auto max_elements = max_bytes < max_difference ? max_bytes : max_difference;
        if (matrix.columns > max_elements ||
            (matrix.row_stride != 0 &&
             matrix.rows - 1 > (max_elements - matrix.columns) / matrix.row_stride))
            throw std::overflow_error("expression matrix extent exceeds the addressable range");
    }

    Descriptor& descriptor() noexcept { return *matrix_; }
    const Descriptor& descriptor() const noexcept { return *matrix_; }
    std::span<const FeatureIdentity> feature_ids() const noexcept { return feature_ids_; }
    std::size_t selected_row() const noexcept { return row_; }

private:
    Descriptor* matrix_;
    std::span<const FeatureIdentity> feature_ids_;
    std::size_t row_;
    Identity identity_;
};

} // namespace detail

// Borrow the descriptor object and feature identities; both must outlive this
// handle. Underlying payload ownership, synchronization, and valid extents are
// supplied by their owner. Read refreshes the descriptor's current pointer.
template<class Scalar, class FeatureIdentity, class Identity>
class DenseExpressionRow final
    : public detail::ExpressionRowBase<const memory::const_matrix_view<Scalar>,
                                       Scalar, FeatureIdentity, Identity> {
    using Base = detail::ExpressionRowBase<const memory::const_matrix_view<Scalar>,
                                           Scalar, FeatureIdentity, Identity>;
public:
    using Base::Base;
};

// This adapter borrows writable caller-owned storage. Requesting an edit does
// not publish a value generation: the owner must apply its own mutation and
// invalidation protocol, including holding it for the entire write lifetime.
template<class Scalar, class FeatureIdentity, class Identity>
class MutableDenseExpressionRow final
    : public detail::ExpressionRowBase<memory::matrix_view<Scalar>,
                                       Scalar, FeatureIdentity, Identity> {
    using Base = detail::ExpressionRowBase<memory::matrix_view<Scalar>,
                                           Scalar, FeatureIdentity, Identity>;
public:
    using Base::Base;

    template<class View> static constexpr bool supports_edit() noexcept {
        return cellerator::detail::can_write<MutableDenseExpressionRow, View>::value;
    }
    template<class View, std::enable_if_t<
        cellerator::detail::can_write<MutableDenseExpressionRow, View>::value, int> = 0>
    View edit() & { return write(ViewTag<View>{}); }
    template<class View> void edit() && = delete;

    MutableExpressionView<Scalar, FeatureIdentity>
    write(ViewTag<MutableExpressionView<Scalar, FeatureIdentity>>) & {
        auto& matrix = this->descriptor();
        const auto feature_ids = this->feature_ids();
        const auto row = this->selected_row();
        Base::validate(matrix, feature_ids, row);
        Scalar* data = matrix.data;
        if (matrix.columns != 0 && row != 0 && matrix.row_stride != 0)
            data += row * matrix.row_stride;
        return {{data, matrix.columns, matrix.where}, feature_ids};
    }
    MutableExpressionView<Scalar, FeatureIdentity>
    write(ViewTag<MutableExpressionView<Scalar, FeatureIdentity>>) && = delete;
};

} // namespace cellerator::cells
