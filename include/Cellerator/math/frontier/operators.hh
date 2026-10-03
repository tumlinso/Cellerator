#pragma once
#include <Cellerator/math/matrix/contracts.hh>
#include <Cellerator/math/effects/providers.hh>
namespace cellerator::math::frontier {
namespace mx=matrix;
namespace ex=execution;
namespace rel=compute::relation;
using Matrix=ce_moon::mechanisms::Matrix;
using generations=mx::generations;
enum class label { exact_identity,model_restriction,approximation }; // E/M/A
// Exact means algebraic identity for the declared family, not bitwise equality or
// biological equivalence. All kernels here are finite FP64 synchronous host calls.
enum class status { success,invalid_axes,invalid_binding,stale_generation,unsupported_policy,
                    ill_conditioned,residual_rejected,provider_failure };
struct square_axes { rel::axis_descriptor rows{},columns{}; };
struct polynomial_primal { const Matrix* X=nullptr;const Matrix* L=nullptr;const Matrix* R=nullptr;const Matrix* M=nullptr;
    const generations* current=nullptr; };
struct polynomial_tape { square_axes axes{};polynomial_primal primal{};generations saved{}; };
struct polynomial_result { Matrix value;polynomial_tape tape;label meaning=label::model_restriction; };
struct polynomial_adjoints { Matrix X,L,R,M; };
inline constexpr label polynomial_delta_meaning=label::exact_identity;
inline constexpr label polynomial_response_meaning=label::exact_identity;
inline constexpr mx::capabilities polynomial_capabilities{mx::nf::forward|mx::nf::jvp|mx::nf::vjp,
    ex::numeric_type::f64,false,false,true,true};
// F(X)=LX+XR+XMX; parameter roles remain separate even when storage aliases.
status polynomial_forward(square_axes,const polynomial_primal&,polynomial_result&) noexcept;
status polynomial_delta(const polynomial_tape&,const Matrix& D,Matrix&) noexcept;
status polynomial_jvp(const polynomial_tape&,const Matrix& dX,const Matrix& dL,const Matrix& dR,const Matrix& dM,Matrix&) noexcept;
status polynomial_vjp(const polynomial_tape&,const Matrix& cotangent,polynomial_adjoints&) noexcept;
struct solve_policy { double pivot_relative=1e-12,residual_relative=1e-10,max_rhs_amplification=1e12; };
struct solve_diagnostics { double normalized_residual=0,rhs_amplification=0; };
struct solve_primal { rel::axis_descriptor coordinates{},right_hand_sides{};
    const Matrix* A=nullptr;const Matrix* B=nullptr;const generations* current=nullptr; };
struct solve_tape { solve_primal primal{};generations saved{};Matrix solution;solve_policy policy{}; };
struct solve_result { Matrix value;solve_diagnostics diagnostics;solve_tape tape;label meaning=label::exact_identity; };
// Partial-pivot native multiple-RHS solve. RHS amplification is an observed guard,
// not a global condition estimate or certificate. No explicit inverse is formed.
status checked_solve(const solve_primal&,solve_policy,solve_result&) noexcept;
status solve_jvp(const solve_tape&,const Matrix& dA,const Matrix& dB,solve_result&) noexcept;
struct port_region {
    rel::axis_descriptor boundary{},interior{};
    const Matrix* Abb=nullptr;const Matrix* Abi=nullptr;const Matrix* Aib=nullptr;const Matrix* Aii=nullptr;
    const std::vector<double>* boundary_load=nullptr;const std::vector<double>* interior_load=nullptr;
    const generations* current=nullptr;
};
struct port_solution {
    std::vector<double> boundary;std::vector<std::vector<double>> interiors;
    std::vector<port_region> borrowed_regions;std::vector<generations> saved;
    solve_diagnostics diagnostics;double full_normalized_residual=0;
    label meaning=label::exact_identity;
};
// Independent interiors and shared boundary identity/order are required. Native
// port/coarse composition owners have a fixed 1e-12 pivot floor; lower policies
// are rejected explicitly. checked_solve alone supports smaller pivot tolerances.
status solve_port_regions(std::span<const port_region>,solve_policy,port_solution&) noexcept;
bool port_solution_is_current(const port_solution&) noexcept;
struct multilevel_descriptor { rel::axis_descriptor fine{},coarse{};label meaning=label::model_restriction; };
struct multilevel_primal { const Matrix* D=nullptr;const Matrix* P=nullptr;const Matrix* K=nullptr;const Matrix* R=nullptr;
    const std::vector<double>* state=nullptr;const generations* current=nullptr; };
struct multilevel_tape { multilevel_descriptor descriptor{};multilevel_primal primal{};generations saved{}; };
struct multilevel_result { std::vector<double> value;multilevel_tape tape;label meaning=label::model_restriction; };
// Declared D+PKR family; E requires caller-supplied exact decomposition.
status multilevel_apply(multilevel_descriptor,const multilevel_primal&,multilevel_result&) noexcept;
status multilevel_input_vjp(const multilevel_tape&,std::span<const double>,std::vector<double>&) noexcept;
struct correction_result { std::vector<double> value;double residual_before=0,residual_after=0;
    label meaning=label::approximation; };
// One actual coarse_direction correction. Rejects when full residual ratio exceeds
// the supplied bound. No convergence promise; original fine solve remains available.
status residual_correction(multilevel_descriptor,const Matrix& A,const Matrix& P,const Matrix& R,
    std::span<const double> x,std::span<const double> rhs,double alpha,double max_residual_ratio,
    solve_policy,correction_result&) noexcept;
} // namespace cellerator::math::frontier
