#include <Cellerator/math/adaptive/reuse.hh>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace cellerator::math::adaptive {
namespace {
void require(bool b, const char* why) { if(!b) throw std::invalid_argument(why); }
bool finite(std::span<const double> x) { return std::all_of(x.begin(),x.end(),[](double v){return std::isfinite(v);}); }
void valid_context(const context& c) {
    const auto& g=c.generations;
    require(execution::valid_identity(g.structure) && g.epoch.value && g.values.value
        && g.activity.value && g.parameters.value,"invalid generations");
    require(!c.program_query_dependencies.empty() && finite(c.world) && finite(c.parameters),"incomplete context");
    for(const auto& x:c.coordinates) require(execution::valid_identity(x.actor) && x.incarnation,"invalid coordinate");
}
double difference(std::span<const double> a,std::span<const double> b) {
    require(a.size()==b.size(),"extent mismatch"); double d=0;
    for(std::size_t i=0;i<a.size();++i) { auto v=std::abs(a[i]-b[i]); require(std::isfinite(v),"discrepancy overflow"); d=std::max(d,v); } return d;
}
std::vector<double> product(std::span<const double> a,std::span<const double> b,
                           std::size_t m,std::size_t n,std::size_t k) {
    std::vector<double> out(m*k,0);
    for(std::size_t i=0;i<m;++i) for(std::size_t j=0;j<k;++j)
        for(std::size_t t=0;t<n;++t) out[i*k+j]+=a[i*n+t]*b[t*k+j];
    require(finite(out),"nonfinite product"); return out;
}
void validate(const linear_snapshot& s) {
    const auto n=s.values.size();
    require(n && s.outputs && s.coordinates.size()==n && s.law.size()==n*n
        && s.readout.size()==s.outputs*n && s.first_moment.size()==n
        && s.second_moment.size()==n && s.optimizer_steps.size()==n,"snapshot extents");
    context c{s.generations,s.coordinates,{1},{},{}}; valid_context(c);
    require(finite(s.values) && finite(s.law) && finite(s.readout)
        && finite(s.first_moment) && finite(s.second_moment),"nonfinite snapshot");
    for(auto v:s.second_moment) require(v>=0,"negative second moment");
    for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<i;++j) {
        auto a=s.coordinates[i],b=s.coordinates[j]; a.incarnation=b.incarnation=1;
        require(!state::same_coordinate(a,b),"duplicate logical slot");
    }
}
}
bool same_context(const context& a,const context& b) noexcept {
    if(!state::same_generations(a.generations,b.generations) || a.coordinates.size()!=b.coordinates.size()
        || a.program_query_dependencies!=b.program_query_dependencies || a.world!=b.world || a.parameters!=b.parameters) return false;
    for(std::size_t i=0;i<a.coordinates.size();++i) if(!state::same_coordinate(a.coordinates[i],b.coordinates[i])) return false;
    return true;
}
delta_result delta_ledger::update(const context& c,std::span<const double> x,double threshold,
                                const evaluator& evaluate,const delta_provider& delta) {
    valid_context(c); require(!x.empty() && x.size()==c.coordinates.size() && finite(x)
        && std::isfinite(threshold) && threshold>=0 && bool(evaluate),"invalid delta input");
    // Value generations advance as observations arrive; all other guards must
    // match. Sent values themselves provide the remembered value dependency.
    auto guard=c; guard.generations.values=context_.generations.values;
    bool reset=!ready_ || !same_context(guard,context_) || sent_.size()!=x.size();
    if(ready_ && execution::same_identity(c.generations.structure,context_.generations.structure))
        require(c.generations.epoch.value>=context_.generations.epoch.value
            && c.generations.values.value>=context_.generations.values.value
            && c.generations.activity.value>=context_.generations.activity.value
            && c.generations.parameters.value>=context_.generations.parameters.value,"generation rollback");
    auto next=reset?std::vector<double>(x.begin(),x.end()):sent_;
    std::vector<double> dx(x.size(),0); std::size_t count=reset?x.size():0;
    if(!reset) for(std::size_t i=0;i<x.size();++i) if(std::abs(x[i]-sent_[i])>threshold) {
        dx[i]=x[i]-sent_[i]; next[i]=x[i]; ++count;
    }
    require(finite(dx),"nonfinite delta");
    auto y=reset?evaluate(next):(count?(delta?delta(sent_,dx,output_):evaluate(next)):output_);
    auto actual=evaluate(x);
    require(!y.empty() && finite(y) && finite(actual),"nonfinite provider output");
    delta_result result{y,next,difference(actual,y),reset,count};
    context_=c; sent_=std::move(next); output_=std::move(y); ready_=true;
    return result;
}
publication::publication(linear_snapshot s) : snapshot_(std::move(s)) { validate(snapshot_); }
rewrite_report publication::publish(const supplied_rewrite& r) {
    require(!leases_,"drain tapes before publication"); validate(r.candidate);
    require(std::isfinite(r.tolerance) && r.tolerance>=0,"invalid tolerance");
    auto next=r.candidate; const auto& old=snapshot_; auto n=old.values.size(),m=next.values.size();
    require(next.outputs==old.outputs && r.forward.size()==m*n && r.backward.size()==n*m
        && finite(r.forward) && finite(r.backward) && r.optimizer_from.size()==m,"rewrite extents");
    require(execution::same_identity(next.generations.structure,old.generations.structure)
        && next.generations.epoch.value>old.generations.epoch.value
        && next.generations.parameters.value>old.generations.parameters.value
        && next.generations.values.value>old.generations.values.value
        && next.generations.activity.value>=old.generations.activity.value,"rewrite must advance native generations");
    auto identity=product(r.backward,r.forward,n,m,n);
    for(std::size_t i=0;i<n;++i) identity[i*n+i]-=1;
    require(finite(identity),"inverse overflow");
    double inv=0; for(auto v:identity) inv=std::max(inv,std::abs(v));
    auto transformed=product(r.forward,old.law,m,n,n);
    auto evolved=product(next.law,r.forward,m,m,n);
    auto read=product(next.readout,r.forward,next.outputs,m,n);
    auto old_read=product(old.readout,old.values,old.outputs,n,1);
    auto new_read=product(next.readout,next.values,next.outputs,m,1);
    rewrite_report report{r.kind,inv,difference(transformed,evolved),difference(read,old.readout),difference(old_read,new_read)};
    // Initialization is supplied, and must agree with q=T*s for these linear rewrites.
    require(difference(product(r.forward,old.values,m,n,1),next.values)<=r.tolerance,"state initialization disagrees with map");
    if(r.kind==rewrite_kind::exact_linear)
        require(n==m && report.inverse_residual<=r.tolerance && report.dynamics_residual<=r.tolerance
            && report.readout_residual<=r.tolerance && report.current_readout_discrepancy<=r.tolerance,"unverified exact rewrite");
    else require(r.kind==rewrite_kind::approximate_linear,"unknown rewrite kind");
    for(std::size_t i=0;i<m;++i) {
        auto j=r.optimizer_from[i];
        next.first_moment[i]=next.second_moment[i]=0; next.optimizer_steps[i]=0;
        if(j<0) { require(j==-1,"invalid optimizer reset marker"); continue; }
        require(std::size_t(j)<n && state::same_coordinate(next.coordinates[i],old.coordinates[j])
            && next.values[i]==old.values[j],"stale optimizer migration");
        // Conservative identity migration: prevent moments crossing changed
        // basis/law/readout semantics, including slot recycling.
        require(n==m && next.law==old.law && next.readout==old.readout,"optimizer migration requires unchanged law/readout");
        for(std::size_t k=0;k<n;++k) require(r.forward[i*n+k]==(k==std::size_t(j)?1.:0.),"optimizer basis mismatch");
        next.first_moment[i]=old.first_moment[j]; next.second_moment[i]=old.second_moment[j]; next.optimizer_steps[i]=old.optimizer_steps[j];
    }
    for(std::size_t i=0;i<m;++i) for(const auto& c:old.coordinates) {
        auto a=next.coordinates[i],b=c; a.incarnation=b.incarnation=1;
        if(state::same_coordinate(a,b)) {
            require(next.coordinates[i].incarnation>=c.incarnation,"incarnation rollback");
            auto old_index=std::size_t(&c-old.coordinates.data());
            bool unchanged_basis=true;
            for(std::size_t k=0;k<n;++k)
                unchanged_basis &= r.forward[i*n+k]==(k==old_index?1.:0.);
            require(unchanged_basis || next.coordinates[i].incarnation>c.incarnation,
                "changed coordinate basis requires new incarnation");
        }
    }
    snapshot_=std::move(next); return report;
}
} // namespace cellerator::math::adaptive
