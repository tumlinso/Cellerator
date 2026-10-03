#include <Cellerator/math/adaptive/reuse.hh>
#include <iostream>
#include <stdexcept>
#include <cmath>
namespace ad=cellerator::math::adaptive;
void check(bool ok,const char* s) { if(!ok) throw std::runtime_error(s); }
template<class F> void rejects(F f) { bool bad=false; try { f(); } catch(const std::invalid_argument&) { bad=true; } check(bad,"expected rejection"); }
bool near(double a,double b) { return std::abs(a-b)<1e-12; }
ad::context context() { return {{{10,1},{1},{1},{1},{1}},{{{1,1},{2,1},0,1}}, {7,8,9}, {1,0}, {2}}; }
void deltas(bool quadratic) {
    auto c=context(); ad::delta_ledger ledger;
    auto eval=[&](std::span<const double> x) { return std::vector<double>{c.parameters[0]*(quadratic?x[0]*x[0]:x[0])}; };
    auto delta=[&](std::span<const double> sent,std::span<const double> dx,std::span<const double> y) {
        return std::vector<double>{y[0]+c.parameters[0]*(quadratic?(2*sent[0]*dx[0]+dx[0]*dx[0]):dx[0])}; };
    auto update=[&](double x) { ++c.generations.values.value; return ledger.update(c,std::vector<double>{x},1,eval,delta); };
    check(update(0).reset,"initial reset");
    auto r=update(.6); check(r.transmitted==0 && r.sent[0]==0 && r.discrepancy>0,"suppressed delta");
    r=update(1.2); check(r.transmitted==1 && near(r.output[0],quadratic?2.88:2.4),"drift compared with last transmission");
    r=update(1.8); check(r.sent[0]==1.2,"not last observed");
    r=update(2.4); check(r.transmitted==1 && near(r.discrepancy,0),"second cumulative transmission");
    c.parameters[0]=3; r=update(2.5); check(r.reset,"actual weights invalidate without generation increment");
    ++c.generations.parameters.value; check(update(2.6).reset,"parameter generation invalidates");
    c.world[1]=1; check(update(2.7).reset,"whole world invalidates even same query");
    ++c.program_query_dependencies[2]; check(update(2.8).reset,"dependency context invalidates");
    ++c.coordinates[0].incarnation; check(update(2.9).reset,"incarnation invalidates");
    ++c.generations.epoch.value; check(update(3).reset,"structure invalidates");
    auto saved=c; --c.generations.parameters.value; rejects([&]{update(3.1);}); c=saved;
    rejects([&]{ledger.update(c,std::vector<double>{4},-1,eval);});
    rejects([&]{ledger.update(c,std::vector<double>{4},1,[](auto){return std::vector<double>{NAN};});});
    r=update(3.5); check(r.sent[0]==3,"failed provider leaves ledger unchanged");
}
ad::linear_snapshot initial() {
    auto c=context(); c.coordinates.push_back({{1,1},{2,1},1,1});
    return {c.generations,c.coordinates,{1,2},{2,0,0,3},{4,5},1,{6,7},{8,9},{10,11}};
}
void rewrites() {
    ad::publication owner(initial()); auto old=owner.snapshot();
    ad::supplied_rewrite r; r.candidate=old;
    ++r.candidate.generations.epoch.value; ++r.candidate.generations.values.value; ++r.candidate.generations.parameters.value;
    r.forward={2,0,0,.5}; r.backward={.5,0,0,2}; r.optimizer_from={-1,-1};
    r.candidate.values={2,1}; r.candidate.readout={2,10};
    ++r.candidate.coordinates[0].incarnation; ++r.candidate.coordinates[1].incarnation;
    { auto tape=owner.acquire_tape(); rejects([&]{owner.publish(r);}); check(owner.snapshot().generations.epoch.value==old.generations.epoch.value,"blocked publication unchanged"); }
    auto bad=r; bad.candidate.readout[0]+=1; rejects([&]{owner.publish(bad);});
    bad=r; bad.candidate.coordinates=old.coordinates; rejects([&]{owner.publish(bad);});
    auto report=owner.publish(r); check(report.inverse_residual==0 && report.dynamics_residual==0
        && report.readout_residual==0 && report.current_readout_discrepancy==0,"exact map/readout witness");
    check(owner.snapshot().first_moment==std::vector<double>({0,0}) && owner.snapshot().optimizer_steps[0]==0,"basis moments reset");
    // Repeated trajectory/readout equality supplements matrix intertwining check.
    auto x=old.values,q=owner.snapshot().values;
    for(int step=0;step<5;++step) {
        check(near(4*x[0]+5*x[1],2*q[0]+10*q[1]),"exact multistep readout");
        x[0]*=2; x[1]*=3; q[0]*=2; q[1]*=3;
    }
    ad::publication reduced(initial()); ad::supplied_rewrite approx;
    approx.candidate=initial(); auto& s=approx.candidate;
    s.coordinates.resize(1); s.values={1}; s.law={2}; s.readout={4}; s.first_moment={55}; s.second_moment={66}; s.optimizer_steps={77};
    ++s.generations.epoch.value; ++s.generations.values.value; ++s.generations.parameters.value;
    approx.forward={1,0}; approx.backward={1,0}; approx.optimizer_from={-1};
    rejects([&]{reduced.publish(approx);}); approx.kind=ad::rewrite_kind::approximate_linear;
    report=reduced.publish(approx); check(report.inverse_residual==1 && report.readout_residual==5
        && report.current_readout_discrepancy==10,"approximate reduction explicit discrepancy");
    check(reduced.snapshot().first_moment[0]==0,"supplied stale moments overwritten by reset");
    ad::publication migration(initial()); ad::supplied_rewrite identity;
    identity.candidate=initial(); ++identity.candidate.generations.epoch.value;
    ++identity.candidate.generations.values.value; ++identity.candidate.generations.parameters.value;
    identity.forward=identity.backward={1,0,0,1}; identity.optimizer_from={0,1};
    auto stale=identity; ++stale.candidate.coordinates[1].incarnation;
    rejects([&]{migration.publish(stale);});
    migration.publish(identity); check(migration.snapshot().first_moment==initial().first_moment,"identity optimizer migration");
    auto recycle=identity; recycle.candidate=migration.snapshot();
    ++recycle.candidate.generations.epoch.value; ++recycle.candidate.generations.values.value;
    ++recycle.candidate.generations.parameters.value; ++recycle.candidate.coordinates[1].incarnation;
    recycle.optimizer_from={0,-1}; migration.publish(recycle);
    check(migration.snapshot().optimizer_steps[0]==10 && migration.snapshot().optimizer_steps[1]==0,"selective slot reset");
    auto invalid=recycle; invalid.candidate=migration.snapshot();
    ++invalid.candidate.generations.epoch.value; ++invalid.candidate.generations.values.value;
    ++invalid.candidate.generations.parameters.value; --invalid.candidate.coordinates[1].incarnation;
    rejects([&]{migration.publish(invalid);});
    invalid.candidate=migration.snapshot(); ++invalid.candidate.generations.epoch.value;
    ++invalid.candidate.generations.values.value; ++invalid.candidate.generations.parameters.value;
    invalid.candidate.values[1]=0; rejects([&]{migration.publish(invalid);});
}
int main() { try { deltas(false); deltas(true); rewrites(); std::cout<<"adaptive native SDK: linear/quadratic remembered deltas; full guards; exact/approximate publication; tape drain; optimizer migration/reset PASS\n"; }
catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
