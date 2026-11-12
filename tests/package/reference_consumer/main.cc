#include <Cellerator/math/reference/operation.hh>
#include <Cellerator/math/reference/referee.hh>
#include <Cellerator/layout/projection/bell.hh>
#include <Cellerator/layout/projection/packed_dense_operand.hh>
#include <Cellerator/compute/projection/native_tile_view.hh>


int main() {
    namespace math = cellerator::compute::math;
    if (!math::scalar_is_zero(math::make_scalar(0.0f))
        || math::scalar_is_zero(math::make_scalar(1.0))) return 1;
    const double a[]{1.0, -2.0}, b[]{1.0, -2.0};
    math::numerical_comparison comparison;
    const math::logical_dense_view candidate{b, 1, 2, 2,
        math::dense_layout_kind::row_major, cellerator::real::value_f64};
    if (!math::compare_spmm_reference(a, 2, candidate, {}, &comparison)
        || !comparison.within_tolerance || comparison.mismatch_count) return 2;
    cellerator::layout::projection::CsrViewV1 projection;
    if (cellerator::layout::projection::build_execution_csr_view_host(
        {}, {}, {}, &projection)) return 3;
    return 0;
}
