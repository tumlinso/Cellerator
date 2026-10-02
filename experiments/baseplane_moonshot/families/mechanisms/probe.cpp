#include <ce_moon/mechanisms.hpp>
#include <iostream>
#include <string>
using namespace ce_moon::mechanisms;
static void check(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
static void near(double a,double b,double tolerance=1e-10) { check(std::abs(a-b)<=tolerance,"numeric comparison"); }
template<class F> static void rejects(F f) { bool caught=false; try { f(); } catch(const std::exception&) { caught=true; } check(caught,"expected rejection"); }
int main() {
  try {
    // Dimension overflow must be rejected before allocation or wrapped shape checks.
    const auto huge=std::numeric_limits<std::size_t>::max()/2+1;
    unsigned overflow_shapes=0;
    try { Matrix invalid(huge,2,{}); } catch(const std::overflow_error&) { ++overflow_shapes; }
    try { Matrix invalid(huge,2); } catch(const std::overflow_error&) { ++overflow_shapes; }
    check(overflow_shapes==2,"overflow dimensions");
    Matrix bounds(2,2,{1,2,3,4});const Matrix& const_bounds=bounds;
    unsigned rejected_coordinates=0;
    try { (void)bounds(0,2); } catch(const std::out_of_range&) { ++rejected_coordinates; }
    try { (void)const_bounds(0,2); } catch(const std::out_of_range&) { ++rejected_coordinates; }
    try { (void)bounds(2,0); } catch(const std::out_of_range&) { ++rejected_coordinates; }
    check(rejected_coordinates==3,"matrix axis bounds");
    // E41 4xp-xi=1, -xp+3xi=2. Exact (5/11,9/11).
    auto ports=condense_ports(Matrix(1,1,{4}),Matrix(1,1,{-1}),Matrix(1,1,{-1}),Matrix(1,1,{3}),{1},{2});
    auto xp=solve_ports(ports),xi=reconstruct_interior(ports,xp);
    auto full=solve(Matrix(2,2,{4,-1,-1,3}),std::vector<double>{1,2});
    near(xp[0],full[0]); near(xi[0],full[1]); near(xp[0],5.0/11); near(xi[0],9.0/11);
    near(ports.condensed(0,0),11.0/3); near(ports.load[0],5.0/3);
    auto composition=compose_port_system({ports,ports});
    near(solve(composition.stiffness,composition.load)[0],xp[0]);
    rejects([&]{compose_port_system({});});
    // Two interiors and pivoting; E11 numeric sensitivity to a unit port-load perturbation.
    Matrix app(2,2,{5,0,0,4}),api(2,2,{-1,0,0,-1}),aip=api,aii(2,2,{3,-1,-1,3});
    auto pair=condense_ports(app,api,aip,aii,{1,2},{3,4});
    auto pp=solve_ports(pair),ii=reconstruct_interior(pair,pp);
    Matrix block(4,4,{5,0,-1,0,0,4,0,-1,-1,0,3,-1,0,-1,-1,3});
    auto direct=solve(block,std::vector<double>{1,2,3,4});
    near(pp[0],direct[0]); near(pp[1],direct[1]); near(ii[0],direct[2]); near(ii[1],direct[3]);
    near(solve(Matrix(2,2,{0,1,1,2}),std::vector<double>{2,5})[0],1);
    double eps=1e-5;
    auto perturbed=condense_ports(Matrix(1,1,{4}),Matrix(1,1,{-1}),Matrix(1,1,{-1}),Matrix(1,1,{3}),{1+eps},{2});
    near((solve_ports(perturbed)[0]-xp[0])/eps,3.0/11,1e-9);
    rejects([&]{condense_ports(Matrix(1,1,{4}),Matrix(1,1,{1}),Matrix(1,1,{1}),Matrix(1,1,{0}),{1},{2});});
    std::cout<<"E41 full solve=("<<xp[0]<<","<<xi[0]<<"), sensitivity="<<3.0/11<<"\n";

    // E42 constant coarse mode corrects a coupled system; alpha is trainable scalar response.
    Matrix a(2,2,{2,-1,-1,2}),p(2,1,{1,1}),r(1,2,{0.5,0.5});
    std::vector<double> x{0,0},b{1,1};
    auto corrected=coarse_correct(a,p,r,x,b);
    near(corrected[0],1);near(corrected[1],1);near(objective(a,corrected,b),0);
    auto fit=fit_step_size(a,p,r,x,b,0.2);
    near(fit.alpha,1); near(fit.gradient_before,-1.6); check(fit.loss_after<fit.loss_before,"fit improvement");
    auto loss=[&](double alpha){return objective(a,coarse_correct(a,p,r,x,b,alpha),b);};
    near((loss(0.2+eps)-loss(0.2-eps))/(2*eps),fit.gradient_before,1e-9);
    // Poor but invertible restriction misses the residual entirely.
    Matrix bad_r(1,2,{1,-1}),bad_p(2,1,{1,0});
    auto missed=coarse_correct(a,bad_p,bad_r,x,b);
    near(objective(a,missed,b),objective(a,x,b));
    rejects([&]{coarse_correct(a,p,Matrix(1,2,{0,0}),x,b);});
    check(loss(3)>loss(0),"oversized step must worsen objective");
    std::cout<<"E42 loss fixed="<<loss(1)<<" fitted alpha="<<fit.alpha<<" bad restriction="<<objective(a,missed,b)<<"\n";

    // E43 sparse compatibility joins: 3*3*3 Cartesian tuples, just two survive equal keys.
    std::vector<RoleEntry> ra{{10,7,1},{11,8,2},{12,9,3}},rb{{20,7,4},{21,8,5},{22,10,6}},rc{{30,7,7},{31,8,8},{32,11,9}};
    FactorTuple output[2];auto jr=join_factors(ra,rb,rc,{0,1,2,3},output,2);
    check(jr.required==2 && jr.written==2 && !jr.overflow,"sparse join");
    check(output[0].ids[0]==10 && output[0].ids[1]==20 && output[0].ids[2]==30,"role provenance");
    near(output[0].score,30); near(output[1].score,36);
    std::size_t brute=0; for(auto u:ra)for(auto v:rb)for(auto w:rc)if(u.key==v.key&&u.key==w.key)++brute;
    check(brute==jr.required,"join brute oracle");
    auto short_join=join_factors(ra,rb,rc,{0,1,2,3},output,1);
    check(short_join.required==2 && short_join.written==1 && short_join.overflow,"bounded overflow");
    check(join_factors(ra,rb,rc,{0,1,2,3},nullptr,0).required==2,"count pass");
    check(join_factors({},rb,rc,{0,1,2,3},nullptr,0).required==0,"empty join");
    rejects([&]{join_factors(ra,rb,rc,{1},output,2);});
    bool score_overflow=false;
    try {
      join_factors({{1,0,std::numeric_limits<double>::max()}},{{2,0,0}},{{3,0,0}},
                   {0,2,0,0},output,1);
    } catch(const std::overflow_error&) { score_overflow=true; }
    check(score_overflow,"nonfinite factor tuple rejected");
    auto aggregates=aggregate_factors(ra,rb,rc,{0,1,2,3});
    check(aggregates.size()==2 && aggregates[0].count==1,"factorized cardinality");
    near(aggregates[0].score_sum,30);near(aggregates[1].score_sum,36);
    ra.push_back({13,7,2});rb.push_back({23,7,5});
    std::vector<FactorTuple> expanded(5);
    auto duplicate_join=join_factors(ra,rb,rc,{0,1,2,3},expanded.data(),expanded.size());
    auto duplicated_aggregates=aggregate_factors(ra,rb,rc,{0,1,2,3});
    double sum7=0;for(std::size_t i=0;i<duplicate_join.written;++i)if(expanded[i].key==7)sum7+=expanded[i].score;
    check(duplicated_aggregates[0].count==4,"duplicate posting products");
    near(duplicated_aggregates[0].score_sum,sum7);
    std::cout<<"E43 tuples=2/27 scores=(30,36), opaque role IDs retained\n";

    // E44 unchanged prefix, diverging node, then exact cancellation allows shared descendants.
    std::vector<Node> dag{{-1,-1,0,0,2},{0,-1,3,0,1},{1,-1,2,0,0},{2,1,1,-2,0},{3,-1,1,0,5}};
    std::vector<std::vector<WorldDelta>> worlds{{},{ {2,18} },{ {2,14} }};
    auto batch=evaluate_worlds(dag,worlds);
    for(std::size_t world=0;world<worlds.size();++world) {
      auto independent=evaluate_independent(dag,worlds[world]);
      check(independent==batch.states[world],"world independent oracle");
    }
    check(batch.evaluations==7,"baseline five plus two dirty descendants");
    near(batch.states[0][4],5); near(batch.states[1][4],9); near(batch.states[2][4],5);
    auto groups=exact_query_groups(batch.states,{4});check(groups[0]==0&&groups[1]==1&&groups[2]==0,"query equivalence");
    auto near_states=batch.states;near_states[2][4]+=1e-12;
    check(exact_query_groups(near_states,{4})[2]==2,"no tolerance merge");
    rejects([&]{evaluate_worlds(dag,{{{99,1}}});});
    rejects([&]{evaluate_worlds(dag,{{{2,1},{2,2}}});});
    rejects([&]{evaluate_independent({{0,-1,1,0,0}});});
    // E45 bilinear exact fixture and discontinuity approximation error.
    near(bilinear_response(0.25,0.75,{0,1,2,3}),1.75);
    near(bilinear_response(0.49,0.5,{0,1,0,1}),0.49); // hard-step oracle gives zero here
    rejects([&]{bilinear_response(1.1,0.5,{0,1,2,3});});
    std::cout<<"E44 query=(5,9,5) evaluations="<<batch.evaluations<<" independent=15; E45 step error=0.49\n";
    std::cout<<"CE-MOON-050 scalar references and shape/score overflow guards compared successfully\n";
  } catch(const std::exception& e) { std::cerr<<e.what()<<"\n";return 1; }
}
