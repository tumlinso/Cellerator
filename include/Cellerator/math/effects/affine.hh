#pragma once
#include <Cellerator/math/effects/providers.hh>
#include <Cellerator/compute/operation/relation_semantics.hh>
namespace cellerator::math::effects {
namespace ex=execution;
namespace rel=compute::relation;
// Vocabulary metadata names numerical coordinates; the caller supplies meaning.
// Epoch versions the declared coordinate/control universe, not mutable values.
struct domain {
    rel::axis_descriptor state{},observable{};
    ex::structure_epoch epoch{};
};
inline bool same_axis(const rel::axis_descriptor& a,const rel::axis_descriptor& b) noexcept {
    return a.extent==b.extent && ex::same_identity(a.identity.domain,b.identity.domain)
        && ex::same_identity(a.identity.order,b.identity.order) && ex::same_identity(a.identity.geometry,b.identity.geometry)
        && ex::same_identity(a.identity.partition,b.identity.partition);
}
inline bool same_domain(const domain& a,const domain& b) noexcept {
    return same_axis(a.state,b.state) && same_axis(a.observable,b.observable) && a.epoch.value==b.epoch.value;
}
inline void validate_domain(const domain& d,std::size_t state,std::size_t observable) {
    if(!d.epoch.value || d.state.extent!=state || d.observable.extent!=observable
        || ex::validate_persistent_axis_identity(d.state.identity)!=ex::biological_validation_code::ok
        || ex::validate_persistent_axis_identity(d.observable.identity)!=ex::biological_validation_code::ok)
        throw std::invalid_argument("effect coordinate universe/epoch");
}
namespace detail {
inline double finite(double x) { return ce_moon::effects::finite(x); }
template<std::size_t N> void finite_input(const std::array<double,N>& x) {
    for(double value:x) if(!std::isfinite(value)) throw std::invalid_argument("nonfinite effect input/coefficient");
}
}
// h'=A h+b; q'=q+C h+d. This is closed for the declared additive q law.
// Owned coefficient arrays are effect summaries, not a canonical parameter master.
// Changed underlying parameters/readouts require caller recomposition. No runtime,
// cache, scheduler, precision adaptation or implicit approximation is supplied.
template<std::size_t N,std::size_t Q> struct accumulated_affine {
    static_assert(N>0);
    domain vocabulary{};
    std::array<std::array<double,N>,N> A{};
    std::array<double,N> b{};
    std::array<std::array<double,N>,Q> C{};
    std::array<double,Q> d{};
    static accumulated_affine identity(domain axes) {
        accumulated_affine effect; effect.vocabulary=axes;
        for(std::size_t i=0;i<N;++i) effect.A[i][i]=1;
        effect.validate(); return effect;
    }
    void validate() const {
        validate_domain(vocabulary,N,Q);
        for(const auto& row:A) detail::finite_input(row);
        for(const auto& row:C) detail::finite_input(row);
        detail::finite_input(b); detail::finite_input(d);
    }
    std::pair<std::array<double,N>,std::array<double,Q>> apply(
        const std::array<double,N>& h,const std::array<double,Q>& q) const {
        validate(); detail::finite_input(h); detail::finite_input(q);
        std::array<double,N> next_h=b;
        std::array<double,Q> next_q{};
        for(std::size_t i=0;i<N;++i) for(std::size_t j=0;j<N;++j)
            next_h[i]=detail::finite(next_h[i]+detail::finite(A[i][j]*h[j]));
        for(std::size_t i=0;i<Q;++i) {
            next_q[i]=detail::finite(q[i]+d[i]);
            for(std::size_t j=0;j<N;++j) next_q[i]=detail::finite(next_q[i]+detail::finite(C[i][j]*h[j]));
        }
        return {next_h,next_q};
    }
};
// Operational order throughout this API: visit left, then right. Real-arithmetic
// closure does not promise bitwise associativity of binary64 matrix composition.
template<std::size_t N,std::size_t Q> accumulated_affine<N,Q> compose(
    const accumulated_affine<N,Q>& left,const accumulated_affine<N,Q>& right) {
    left.validate(); right.validate();
    if(!same_domain(left.vocabulary,right.vocabulary)) throw std::invalid_argument("effect vocabulary mismatch");
    accumulated_affine<N,Q> out; out.vocabulary=left.vocabulary;
    for(std::size_t i=0;i<N;++i) {
        out.b[i]=right.b[i];
        for(std::size_t k=0;k<N;++k) out.b[i]=detail::finite(out.b[i]+detail::finite(right.A[i][k]*left.b[k]));
        for(std::size_t j=0;j<N;++j) for(std::size_t k=0;k<N;++k)
            out.A[i][j]=detail::finite(out.A[i][j]+detail::finite(right.A[i][k]*left.A[k][j]));
    }
    for(std::size_t i=0;i<Q;++i) {
        out.d[i]=detail::finite(left.d[i]+right.d[i]);
        for(std::size_t k=0;k<N;++k) out.d[i]=detail::finite(out.d[i]+detail::finite(right.C[i][k]*left.b[k]));
        for(std::size_t j=0;j<N;++j) {
            out.C[i][j]=left.C[i][j];
            for(std::size_t k=0;k<N;++k) out.C[i][j]=detail::finite(out.C[i][j]+detail::finite(right.C[i][k]*left.A[k][j]));
        }
    }
    return out;
}
// F(s,h)=(transition[s], A[s]h+b[s]), with fixed finite control vocabulary.
// Integer transitions have no ordinary gradient; route-learning policy is absent.
template<std::size_t S,std::size_t N> struct hybrid_affine {
    static_assert(S>0 && N>0);
    rel::axis_descriptor state{},control{};
    ex::structure_epoch epoch{};
    std::array<std::size_t,S> transition{};
    std::array<std::array<std::array<double,N>,N>,S> A{};
    std::array<std::array<double,N>,S> b{};
    static hybrid_affine identity(rel::axis_descriptor state_axis,rel::axis_descriptor control_axis,ex::structure_epoch version) {
        hybrid_affine out; out.state=state_axis; out.control=control_axis; out.epoch=version;
        for(std::size_t s=0;s<S;++s) { out.transition[s]=s; for(std::size_t i=0;i<N;++i) out.A[s][i][i]=1; }
        out.validate(); return out;
    }
    void validate() const {
        if(!epoch.value || state.extent!=N || control.extent!=S
            || ex::validate_persistent_axis_identity(state.identity)!=ex::biological_validation_code::ok
            || ex::validate_persistent_axis_identity(control.identity)!=ex::biological_validation_code::ok)
            throw std::invalid_argument("hybrid coordinate/control vocabulary");
        for(std::size_t s=0;s<S;++s) {
            if(transition[s]>=S) throw std::invalid_argument("hybrid transition outside vocabulary");
            for(const auto& row:A[s]) detail::finite_input(row);
            detail::finite_input(b[s]);
        }
    }
    std::pair<std::size_t,std::array<double,N>> apply(std::size_t control_state,const std::array<double,N>& h) const {
        validate(); detail::finite_input(h);
        if(control_state>=S) throw std::invalid_argument("hybrid incoming control state");
        auto value=b[control_state];
        for(std::size_t i=0;i<N;++i) for(std::size_t j=0;j<N;++j)
            value[i]=detail::finite(value[i]+detail::finite(A[control_state][i][j]*h[j]));
        return {transition[control_state],value};
    }
};
template<std::size_t S,std::size_t N> hybrid_affine<S,N> compose(
    const hybrid_affine<S,N>& left,const hybrid_affine<S,N>& right) {
    left.validate(); right.validate();
    if(!same_axis(left.state,right.state) || !same_axis(left.control,right.control) || left.epoch.value!=right.epoch.value)
        throw std::invalid_argument("hybrid vocabulary mismatch");
    hybrid_affine<S,N> out; out.state=left.state; out.control=left.control; out.epoch=left.epoch;
    for(std::size_t s=0;s<S;++s) {
        const auto middle=left.transition[s]; out.transition[s]=right.transition[middle];
        out.b[s]=right.b[middle];
        for(std::size_t i=0;i<N;++i) {
            for(std::size_t k=0;k<N;++k) out.b[s][i]=detail::finite(out.b[s][i]+detail::finite(right.A[middle][i][k]*left.b[s][k]));
            for(std::size_t j=0;j<N;++j) for(std::size_t k=0;k<N;++k)
                out.A[s][i][j]=detail::finite(out.A[s][i][j]+detail::finite(right.A[middle][i][k]*left.A[s][k][j]));
        }
    }
    return out;
}
} // namespace cellerator::math::effects
