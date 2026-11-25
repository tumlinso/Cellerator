#include <Cellerator/rds/matrix.hh>

#define R_NO_REMAP
#include <Rinternals.h>

namespace cellerator::rds {
namespace {

static_assert(sizeof(int) == 4);
constexpr memory::placement host{memory::domain::host};

AxisName axis_name(void *names, size_t i, size_t count) noexcept {
    if (!names || i >= count) return {};
    SEXP name = STRING_ELT(static_cast<SEXP>(names), R_xlen_t(i));
    if (name == NA_STRING) return {};
    cetype_t encoding = Rf_getCharCE(name);
    return {CHAR(name), encoding == CE_UTF8 ? NameEncoding::utf8 :
                        encoding == CE_LATIN1 ? NameEncoding::latin1 :
                        encoding == CE_BYTES ? NameEncoding::bytes : NameEncoding::native};
}

} // namespace

struct Matrix::Load {
    const char *path;
    Data& data;
    ReadStatus status;

    static ReadStatus describe(SEXP object, Data& d) {
        int type = TYPEOF(object);
        bool sparse = Rf_isS4(object) && Rf_inherits(object, "dgCMatrix");
        if (!sparse && (Rf_isObject(object) || (type != INTSXP && type != REALSXP)))
            return ReadStatus::unsupported_matrix;

        SEXP dim = sparse ? R_do_slot(object, Rf_install("Dim")) : Rf_getAttrib(object, R_DimSymbol);
        if (dim == R_NilValue) return ReadStatus::unsupported_matrix;
        if (TYPEOF(dim) != INTSXP || XLENGTH(dim) != 2) return ReadStatus::unsupported_matrix;
        if (ALTREP(dim)) return ReadStatus::unsupported_matrix;
        const int *extent = INTEGER_RO(dim);
        if (extent[0] < 0 || extent[1] < 0) return ReadStatus::invalid_matrix;
        d.rows = size_t(extent[0]); d.columns = size_t(extent[1]);

        SEXP values = sparse ? R_do_slot(object, Rf_install("x")) : object;
        if (ALTREP(values)) return ReadStatus::unsupported_matrix;
        if (sparse) {
            SEXP p = R_do_slot(object, Rf_install("p")), i = R_do_slot(object, Rf_install("i"));
            if (TYPEOF(values) != REALSXP || TYPEOF(p) != INTSXP || TYPEOF(i) != INTSXP ||
                uint64_t(XLENGTH(p)) != uint64_t(d.columns) + 1)
                return ReadStatus::invalid_matrix;
            if (ALTREP(p) || ALTREP(i)) return ReadStatus::unsupported_matrix;

            d.offsets = INTEGER_RO(p); d.indices = INTEGER_RO(i);
            int nnz = d.offsets[d.columns];
            if (d.offsets[0] != 0 || nnz < 0 || nnz > XLENGTH(i) || nnz > XLENGTH(values))
                return ReadStatus::invalid_matrix;
            for (size_t col = 0; col < d.columns; ++col) {
                int first = d.offsets[col], end = d.offsets[col + 1];
                if (first < 0 || end < first || end > nnz) return ReadStatus::invalid_matrix;
                int prev = -1;
                for (int k = first; k < end; ++k) {
                    int row = d.indices[k];
                    if (row <= prev || row >= extent[0]) return ReadStatus::invalid_matrix;
                    prev = row;
                }
            }
            d.nnz = size_t(nnz);
            d.storage = MatrixStorage::csc_real;
        } else {
            if (uint64_t(d.rows)*d.columns != uint64_t(XLENGTH(values))) return ReadStatus::invalid_matrix;
            d.storage = type == INTSXP ? MatrixStorage::dense_integer : MatrixStorage::dense_real;
        }

        SEXP names = sparse ? R_do_slot(object, Rf_install("Dimnames")) : Rf_getAttrib(object, R_DimNamesSymbol);
        if (names != R_NilValue) {
            if (TYPEOF(names) != VECSXP || XLENGTH(names) != 2) return ReadStatus::invalid_matrix;
            if (ALTREP(names)) return ReadStatus::unsupported_matrix;
            for (int axis = 0; axis < 2; ++axis) {
                SEXP labels = VECTOR_ELT(names, axis);
                if (labels == R_NilValue) continue;
                if (TYPEOF(labels) != STRSXP || XLENGTH(labels) != extent[axis])
                    return ReadStatus::invalid_matrix;
                if (ALTREP(labels)) return ReadStatus::unsupported_matrix;
                d.names[axis] = labels;
            }
        }
        d.values = DATAPTR_RO(values);
        return ReadStatus::ok;
    }

    static void run(void *opaque) {
        // R errors can jump out of this callback. Its locals own no resources.
        auto& load = *static_cast<Load*>(opaque);
        SEXP path = PROTECT(Rf_mkString(load.path));
        SEXP call = PROTECT(Rf_lang2(Rf_install("readRDS"), path));
        SEXP object = PROTECT(Rf_eval(call, R_BaseEnv));
        load.status = describe(object, load.data);
        if (load.status == ReadStatus::ok) {
            R_PreserveObject(object);
            load.data.object = object;
        }
        UNPROTECT(3);
    }
};

Matrix::~Matrix() noexcept { reset(); }

Matrix::Matrix(Matrix&& other) noexcept : data_(other.data_) { other.data_ = {}; }

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this == &other) return *this;
    reset();
    data_ = other.data_;
    other.data_ = {};
    return *this;
}

ReadStatus Matrix::read(const char *path) noexcept {
    if (!path || !*path) return ReadStatus::invalid_path;
    if (!R_GlobalEnv) return ReadStatus::r_not_initialized;

    Matrix next;
    Load load{path, next.data_, ReadStatus::r_error};
    if (!R_ToplevelExec(Load::run, &load)) return ReadStatus::r_error;
    if (load.status == ReadStatus::ok) *this = static_cast<Matrix&&>(next);
    return load.status;
}

void Matrix::reset() noexcept {
    if (data_.object) R_ReleaseObject(static_cast<SEXP>(data_.object));
    data_ = {};
}

memory::const_matrix_view<int> Matrix::dense_integer() const & noexcept {
    if (storage() != MatrixStorage::dense_integer) return {};
    return {static_cast<const int*>(data_.values), rows(), columns(), 1, host, rows()};
}

memory::const_matrix_view<double> Matrix::dense_real() const & noexcept {
    if (storage() != MatrixStorage::dense_real) return {};
    return {static_cast<const double*>(data_.values), rows(), columns(), 1, host, rows()};
}

tensor::base::CompressedView<int, const double> Matrix::csc_real() const & noexcept {
    if (storage() != MatrixStorage::csc_real) return {};
    return {{rows(), columns(), data_.nnz, tensor::base::CompressedAxis::column, data_.offsets, data_.indices},
            static_cast<const double*>(data_.values)};
}

AxisName Matrix::row_name(size_t i) const & noexcept { return axis_name(data_.names[0], i, rows()); }

AxisName Matrix::column_name(size_t i) const & noexcept { return axis_name(data_.names[1], i, columns()); }

} // namespace cellerator::rds
