#include "learning.hpp"
#include <iostream>
#include <limits>

using namespace ce_moon::learning;
void check(bool x,const char* message) { if(!x) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
  bool failed=false; try { f(); } catch(const std::invalid_argument&) { failed=true; }
  check(failed,"invalid input accepted");
}
int main() {
  // Full objective gradient including positive-rate penalty, bias and L2.
  const double rows[]{-1,0.3, 0.2,-0.1, 0.8,1.2, 1.5,-0.3};
  const double ys[]{0,0,1,1};
  double w[]{0.4,-0.2,0.3},g[3];
  FitOptions o; o.l2=0.07; o.budget_weight=2; o.positive_budget=0.2;
  logistic_objective(rows,ys,4,2,w,o,g);
  for(unsigned j=0;j<3;++j) {
    const double saved=w[j],eps=1e-6;
    w[j]=saved+eps; double hi=logistic_objective(rows,ys,4,2,w,o);
    w[j]=saved-eps; double lo=logistic_objective(rows,ys,4,2,w,o);
    w[j]=saved;
    check(std::abs((hi-lo)/(2*eps)-g[j])<1e-8,"budget objective gradient");
  }
  // Separate held-out numeric rows check that fitted features, rather than labels,
  // determine inference. Repeated fits are identical.
  const double training[]{-3,-2,-1,1,2,3},targets[]{0,0,0,1,1,1};
  double model[2],repeat[2];
  auto fitted=fit_logistic(training,targets,6,1,model);
  fit_logistic(training,targets,6,1,repeat);
  check(model[0]==repeat[0] && model[1]==repeat[1],"deterministic fit");
  check(fitted.final_loss<fitted.initial_loss/5,"learning objective decrease");
  const double heldout[]{-1.5,1.5};
  check(predict_logistic(heldout,1,model)<0.1 && predict_logistic(heldout+1,1,model)>0.9,"held-out numerical rows");
  // A continuous expected-rate penalty changes decisions even with identical targets.
  const double zeros[]{0,0,0,0},allpositive[]{1,1,1,1};
  double free_w[2],budget_w[2];
  auto free=fit_logistic(zeros,allpositive,4,1,free_w);
  FitOptions budget; budget.budget_weight=20; budget.positive_budget=0.2;
  auto constrained=fit_logistic(zeros,allpositive,4,1,budget_w,budget);
  check(constrained.positive_rate<free.positive_rate-0.5,"rate penalty ignored");
  check(constrained.final_loss<constrained.initial_loss,"budget loss decrease");
  // Stable numerics at very large finite logits; rejects invalid shapes/data/options.
  check(sigmoid(1000)==1 && sigmoid(-1000)==0,"stable sigmoid");
  double extreme[]{0,1000};
  check(std::isfinite(logistic_objective(zeros,allpositive,4,1,extreme)),"stable BCE");
  rejects([&]{fit_logistic(training,targets,0,1,model);});
  FitOptions bad; bad.learning_rate=-1;
  rejects([&]{fit_logistic(training,targets,6,1,model,bad);});
  const double invalid[]{std::numeric_limits<double>::quiet_NaN()};
  rejects([&]{predict_logistic(invalid,1,model);});
  const double wrong[]{2};
  rejects([&]{fit_logistic(zeros,wrong,1,1,model);});
  rejects([&]{select_bank(zeros,1,model,0,1,1);});

  // E29 teacher, emitted immediate and independent exhaustive target comparisons.
  std::array<double,8> target{};
  for(unsigned i=0;i<8;++i) target[i]=((i>>2)&1)^(((i>>1)&1)&(i&1));
  auto teacher=fit_truth_table(target,200,0.5,7);
  auto circuit=harden(teacher,9);
  unsigned expected=0;
  for(unsigned i=0;i<8;++i) {
    expected |= unsigned(target[i])<<i;
    check(circuit.predict(i)==bool(target[i]),"hard LUT target disagreement");
    check((teacher.predict(i)>=0.5)==circuit.predict(i),"hard teacher disagreement");
    check(std::abs(teacher.predict(i)-target[i])<0.02,"continuous teacher fit");
  }
  check(circuit.immediate==expected && circuit.teacher_version==7 && circuit.circuit_version==9,"LUT/version");
  check(emit_lop3(circuit).find(std::to_string(expected))!=std::string::npos,"native immediate emission");
  rejects([&]{circuit.predict(8);});

  // E30 train both bank selectors; identical caller features under two states
  // select distinct banks and preserve explicit selection versions.
  const double states[]{-2,-1,1,2},bank0targets[]{1,1,0,0},bank1targets[]{0,0,1,1};
  double banks[4];
  fit_logistic(states,bank0targets,4,1,banks);
  fit_logistic(states,bank1targets,4,1,banks+2);
  const double state0=-0.8,state1=0.8;
  auto b0=select_bank(&state0,1,banks,2,11,13);
  auto b1=select_bank(&state1,1,banks,2,12,13);
  check(b0.index==0 && b1.index==1 && b0.state_version==11 && b1.weight_version==13,"learned bank selector");

  // E31 two Boolean forms with different costs share exact exhaustive semantics.
  auto a=input(0,42),b=input(1,42),c=input(2,42);
  auto expanded=binary(MaskOp::disjunction,binary(MaskOp::conjunction,a,b),binary(MaskOp::conjunction,a,c));
  auto factored=binary(MaskOp::conjunction,a,binary(MaskOp::disjunction,b,c));
  auto rewrite=minimize_boolean(expanded,3);
  check(rewrite.after<rewrite.before,"cost extraction did not reduce expression");
  check(truth_table(expanded)==truth_table(factored) && truth_table(rewrite.expression)==truth_table(expanded),"Boolean equivalence");
  for(unsigned i=0;i<8;++i) check(evaluate(expanded,i)==evaluate(rewrite.expression,i),"exhaustive rewrite rows");
  rejects([&]{binary(MaskOp::conjunction,a,input(1,99));});
  rejects([&]{minimize_boolean(a,5);});
  check(!reassociation_allowed(NumericalMode::strict_float) && reassociation_allowed(NumericalMode::approximate_float),"float rewrite tagging");
  const double fp_a=1e16,fp_b=-1e16,fp_c=1;
  check((fp_a+fp_b)+fp_c!=fp_a+(fp_b+fp_c),"unsafe reassociation witness");

  // E32 independent stale query, state, domain and weight invalidation.
  Guard context{7,2,3,42};
  auto cached=specialize(teacher,context,9);
  check(guarded_predict(cached,teacher,context,0).specialized,"guard matching");
  Guard changed=context; ++changed.query_version;
  check(!guarded_predict(cached,teacher,changed,0).specialized,"stale query");
  changed=context; ++changed.state_version;
  check(!guarded_predict(cached,teacher,changed,0).specialized,"stale state");
  changed=context; ++changed.domain_version;
  check(!guarded_predict(cached,teacher,changed,0).specialized,"stale domain");
  auto newer=teacher; newer.weight_version=8; newer.logits[0]=10;
  changed=context; changed.weight_version=8;
  auto fallback=guarded_predict(cached,newer,changed,0);
  check(!fallback.specialized && fallback.decision && !cached.lut.predict(0),"stale weight silently reused");
  rejects([&]{guarded_predict(cached,newer,context,0);});
  rejects([&]{specialize(newer,context);});
  std::cout << "E14 objective " << fitted.initial_loss << " -> " << fitted.final_loss
            << "; rate " << free.positive_rate << " -> " << constrained.positive_rate
            << "; E29 LUT " << expected << "; E30 banks 0/1; E31 cost " << rewrite.before
            << " -> " << rewrite.after << "; E32 four guard invalidations pass\n";
}
