#pragma once
#include <Cellerator/math/matrix/contracts.hh>
namespace cellerator::math::process {
using matrix::status;
struct port_descriptor {
    matrix::rel::axis_descriptor actors{},ports{},hidden{},edges{};
    std::span<const matrix::rel::axis_descriptor> private_axes{};
    std::span<const std::int64_t> widths,source,destination;
    matrix::rel::arithmetic_policy policy=matrix::host_policy();
};
struct port_primal {
    std::span<const float> hidden,encoder,decoder,weights;
    const matrix::generations* current=nullptr; // roles h,E,D,edge weights
};
struct port_tape { port_descriptor descriptor{}; port_primal primal{}; matrix::generations saved{}; };
inline constexpr matrix::capabilities port_capabilities{matrix::nf::forward|matrix::nf::vjp|matrix::nf::jvp,
    matrix::ex::numeric_type::f32,false,false,true,true};
// Encoder E_i[P,H_i], decoder D_i[H_i,P], explicit ordered directed edges.
// Transport only: y_i=D_i sum_edges(dst=i) w_edge E_src h_src.
// Caller retains all borrowed values/topology and serialized generation records
// through responses. Local nonlinear laws, optimizer and publication stay caller
// owned. Native provider allocates temporary port arrays on every call.
status port_forward(const port_descriptor&,const port_primal&,std::span<float> output,port_tape&) noexcept;
status port_vjp(const port_tape&,std::span<const float> cotangent,std::span<float> hidden_gradient,
    std::span<float> encoder_gradient,std::span<float> decoder_gradient,std::span<float> edge_gradient) noexcept;
status port_jvp(const port_tape&,std::span<const float> hidden_direction,
    std::span<const float> encoder_direction,std::span<const float> decoder_direction,
    std::span<const float> edge_direction,std::span<float> response) noexcept;
} // namespace cellerator::math::process
