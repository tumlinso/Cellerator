#pragma once

#include <stddef.h>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

namespace cellerator::rds {

enum class ReadStatus { ok, invalid_path, r_not_initialized, r_error, unsupported_matrix, invalid_matrix };
enum class MatrixStorage { none, dense_integer, dense_real, csc_real };
enum class NameEncoding { native, utf8, latin1, bytes };

struct AxisName {
    const char *text = nullptr;
    NameEncoding encoding = NameEncoding::native;
};

// Owns the R reference behind the borrowed views. Use the R main thread for
// reads, name lookup, and release. Release all owners before shutting down R.
class Matrix {
public:
    Matrix() noexcept = default;
    ~Matrix() noexcept;
    Matrix(Matrix&&) noexcept;
    Matrix& operator=(Matrix&&) noexcept;
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    // Deserialize one trusted RDS file. A failed read preserves the current matrix.
    [[nodiscard]] ReadStatus read(const char *path) noexcept;
    void reset() noexcept;

    MatrixStorage storage() const noexcept { return data_.storage; }
    size_t rows() const noexcept { return data_.rows; }
    size_t columns() const noexcept { return data_.columns; }

    // An accessor for a different storage type returns an empty view.
    [[nodiscard]] memory::const_matrix_view<int> dense_integer() const & noexcept;
    [[nodiscard]] memory::const_matrix_view<double> dense_real() const & noexcept;
    [[nodiscard]] tensor::base::CompressedView<int, const double> csc_real() const & noexcept;
    memory::const_matrix_view<int> dense_integer() const && = delete;
    memory::const_matrix_view<double> dense_real() const && = delete;
    tensor::base::CompressedView<int, const double> csc_real() const && = delete;

    // Names borrow the original encoded bytes. Missing or out-of-range names have null text.
    [[nodiscard]] AxisName row_name(size_t i) const & noexcept;
    [[nodiscard]] AxisName column_name(size_t i) const & noexcept;
    AxisName row_name(size_t) const && = delete;
    AxisName column_name(size_t) const && = delete;

private:
    struct Data {
        void *object;
        void *names[2];
        const void *values;
        const int *indices, *offsets;
        size_t rows, columns, nnz;
        MatrixStorage storage;
    } data_{};
    struct Load;
};

} // namespace cellerator::rds
