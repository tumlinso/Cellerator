#pragma once
#include <Cellerator/math/adaptive/reuse.hh>
#include <Cellerator/math/frontier/operators.hh>
#include <stdexcept>
namespace cellerator::math::adaptive {
// Borrowed immutable coefficients and live generations must outlive callbacks.
// Caller declares those dependencies in the ledger context, including actual
// coefficient values. Callbacks execute synchronously; no tape escapes a call.
struct polynomial_providers {
    evaluator evaluate;
    delta_provider delta;
};
inline polynomial_providers bind_polynomial(frontier::square_axes axes,
        const frontier::Matrix& L,const frontier::Matrix& R,const frontier::Matrix& M,
        const frontier::generations& current) {
    auto primal=[axes,&L,&R,&M,&current](std::span<const double> x) {
        auto n=axes.rows.extent;
        if(n!=axes.columns.extent || !n || x.size()/n!=n || x.size()%n)
            throw std::invalid_argument("polynomial ledger extent");
        frontier::Matrix X(n,n,std::vector<double>(x.begin(),x.end()));
        frontier::polynomial_result out;
        if(frontier::polynomial_forward(axes,{&X,&L,&R,&M,&current},out)!=frontier::status::success)
            throw std::invalid_argument("polynomial ledger primal rejected");
        return out.value.data;
    };
    auto delta=[axes,&L,&R,&M,&current](std::span<const double> old,std::span<const double> dx,
                                    std::span<const double> output) {
        auto n=axes.rows.extent;
        if(n!=axes.columns.extent || !n || old.size()/n!=n || old.size()%n
            || dx.size()!=old.size() || output.size()!=old.size())
            throw std::invalid_argument("polynomial ledger delta extent");
        frontier::Matrix X(n,n,std::vector<double>(old.begin(),old.end()));
        frontier::Matrix D(n,n,std::vector<double>(dx.begin(),dx.end())),change;
        frontier::polynomial_result tape;
        if(frontier::polynomial_forward(axes,{&X,&L,&R,&M,&current},tape)!=frontier::status::success
            || frontier::polynomial_delta(tape.tape,D,change)!=frontier::status::success)
            throw std::invalid_argument("polynomial ledger delta rejected");
        for(std::size_t i=0;i<change.data.size();++i)change.data[i]+=output[i];
        return change.data;
    };
    return {std::move(primal),std::move(delta)};
}
} // namespace cellerator::math::adaptive
