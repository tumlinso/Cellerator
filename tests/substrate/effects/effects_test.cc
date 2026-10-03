#include <Cellerator/math/effects/affine.hh>
#include <iostream>
#include <type_traits>
namespace fx=cellerator::math::effects;
namespace ex=cellerator::execution;
int checks=0;
void require(bool b,const char* why) { ++checks; if(!b) throw std::runtime_error(why); }
void close(double a,double b,const char* why) { require(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<1e-10*(1+std::abs(b)),why); }
template<class Exception,class F> void rejects(F call,const char* why) {
    bool caught=false; try { call(); } catch(const Exception&) { caught=true; } require(caught,why);
}
fx::rel::axis_descriptor axis(std::uint64_t id,std::uint64_t n) {
    return {{{ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)},
        {id,1},{id,2},{id,3},{id,4}},n};
}
void existing_owners() {
    static_assert(std::is_same_v<fx::Dfa32,ce_moon::Dfa32>);
    static_assert(std::is_same_v<fx::ResidualTree,ce_moon::ResidualTree>);
    auto left=fx::Dfa32::identity(),right=left;
    for(unsigned s=0;s<32;++s) { left.to[s]=(s+3)%32; right.to[s]=(2*s)%32; }
    auto combined=fx::compose(left,right);
    for(unsigned s=0;s<32;++s) require(combined.to[s]==right.to[left.to[s]],"DFA sequential order");
    fx::CountedDfa32 counted_left{left,{}},counted_right{right,{}};
    for(unsigned s=0;s<32;++s) { counted_left.count[s]=s; counted_right.count[s]=2*s; }
    auto counted=fx::compose(counted_left,counted_right);
    for(unsigned s=0;s<32;++s) require(counted.count[s]==s+2*left.to[s],"count follows routed intermediate state");
    counted_left.count[0]=UINT64_MAX; counted_right.count[left.to[0]]=1;
    rejects<std::overflow_error>([&] { fx::compose(counted_left,counted_right); },"counted unsigned overflow");
    auto m=fx::MonomialAffine<2>::identity(),n=m; m.p={1,0}; m.d={2,-1}; m.b={1,3}; n.d={.5,2}; n.b={-2,1};
    std::array<double,2> x{4,5}; const auto direct=fx::apply(n,fx::apply(m,x)),composed=fx::apply(fx::compose(m,n),x);
    for(unsigned i=0;i<2;++i) close(composed[i],direct[i],"original monomial owner");
    auto block=fx::BlockAffine<1,2>::identity(),other=block;
    block.matrix[0][0][1]=2; block.bias[0][1]=1; other.matrix[0][1][0]=-1;
    const auto block_direct=fx::apply(other,fx::apply(block,x)),block_composed=fx::apply(fx::compose(block,other),x);
    for(unsigned i=0;i<2;++i) close(block_composed[i],block_direct[i],"original block affine owner");
    auto lift=fx::lift(3,7); const auto recovered=fx::unlift(lift);
    close(recovered.first,3,"lifting even reconstruction"); close(recovered.second,7,"lifting odd reconstruction");
    fx::ResidualTree residual({3,1,8,-2});
    for(unsigned i=0;i<4;++i) close(residual.reconstruct(i),std::array<double,4>{3,1,8,-2}[i],"residual reconstruction");
    require(residual.above(4)==std::vector<std::size_t>{2},"residual certified scalar-max query");
    fx::Relation16 a{},b{}; a[1][2]=1; b[2][3]=1;
    require(fx::compose_relation(a,b)[1][3]==1,"finite relation composition order");
    a[1][2]=2; rejects<std::invalid_argument>([&] { fx::compose_relation(a,b); },"finite relation binary vocabulary");
    fx::Jet j{0,1,2,.5,1},k{1,3,-1,.25,2}; const auto jet=fx::compose(j,k);
    close(jet.linear,-2,"jet chain derivative at recorded point"); close(jet.quadratic,.5,"jet second coefficient composition");
    const auto query=fx::query(jet,0); require(query.in_radius && !query.certified_error_bound,"jet query is not a Taylor certificate");
    k.center=2; rejects<std::invalid_argument>([&] { fx::compose(j,k); },"jet expansion point mismatch");
    fx::Matrix app(1,1,{4}),api(1,1,{-1}),aip(1,1,{-1}),aii(1,1,{3});
    const auto ports=fx::condense_ports(app,api,aip,aii,{1},{2}); const auto boundary=fx::solve_ports(ports);
    close(boundary[0],5./11,"original port condensation owner"); close(fx::reconstruct_interior(ports,boundary)[0],9./11,"interior recovery");
}
void accumulated() {
    using Effect=fx::accumulated_affine<2,1>;
    const fx::domain vocab{axis(1,2),axis(2,1),{3}};
    auto left=Effect::identity(vocab),right=left,third=left;
    left.A={{{1,2},{0,1}}}; left.b={1,-1}; left.C={{{2,-1}}}; left.d={.5};
    right.A={{{0,1},{-1,0}}}; right.b={.25,2}; right.C={{{-3,.5}}}; right.d={-.25};
    third.b={1,2}; third.C={{{.5,2}}};
    const std::array<double,2> h{2,3}; const std::array<double,1> q{7};
    auto middle=left.apply(h,q),direct=right.apply(middle.first,middle.second),joined=fx::compose(left,right).apply(h,q);
    close(joined.first[0],2.25,"affine state direct oracle0"); close(joined.first[1],-7,"affine state direct oracle1");
    close(joined.second[0],-17.75,"nonzero accumulated observable direct oracle");
    for(unsigned i=0;i<2;++i) close(joined.first[i],direct.first[i],"affine composed vs sequential state");
    close(joined.second[0],direct.second[0],"affine composed vs sequential observable");
    const auto reverse=fx::compose(right,left).apply(h,q);
    require(reverse.first!=joined.first && reverse.second!=joined.second,"noncommuting sequential order retained");
    const auto lhs=fx::compose(fx::compose(left,right),third).apply(h,q),rhs=fx::compose(left,fx::compose(right,third)).apply(h,q);
    for(unsigned i=0;i<2;++i) close(lhs.first[i],rhs.first[i],"three-region affine association within tolerance");
    close(lhs.second[0],rhs.second[0],"three-region observable association within tolerance");
    const auto empty=Effect::identity(vocab).apply(h,q); require(empty.first==h && empty.second==q,"empty identity effect");
    auto bad=right; ++bad.vocabulary.epoch.value;
    rejects<std::invalid_argument>([&] { fx::compose(left,bad); },"affine universe epoch mismatch");
    bad=right; ++bad.vocabulary.state.identity.order.low;
    rejects<std::invalid_argument>([&] { fx::compose(left,bad); },"affine domain order mismatch");
    bad=left; bad.A[0][0]=std::numeric_limits<double>::max();
    rejects<std::overflow_error>([&] { bad.apply(h,q); },"new affine finite intermediate overflow");
    rejects<std::invalid_argument>([&] { left.apply({std::numeric_limits<double>::infinity(),0},q); },"nonfinite bound state");
    const auto none=fx::accumulated_affine<1,0>::identity({axis(1,1),axis(2,0),{1}}).apply({4},{});
    require(none.first[0]==4 && none.second.empty(),"no compulsory stored readouts");
}
void hybrid() {
    using Effect=fx::hybrid_affine<3,1>;
    auto left=Effect::identity(axis(1,1),axis(3,3),{5}),right=left,third=left;
    left.transition={1,1,0}; right.transition={2,0,2}; third.transition={1,2,1};
    for(unsigned s=0;s<3;++s) { left.A[s][0][0]=s+1; left.b[s][0]=s;
        right.A[s][0][0]=s+2; right.b[s][0]=-static_cast<double>(s); third.b[s][0]=1; }
    auto joined=fx::compose(left,right);
    for(unsigned s=0;s<3;++s) {
        const auto middle=left.apply(s,{2}),direct=right.apply(middle.first,middle.second),composed=joined.apply(s,{2});
        require(composed.first==direct.first,"hybrid nonbijective routed discrete state"); close(composed.second[0],direct.second[0],"hybrid direct sequential continuous response");
        const auto a=fx::compose(fx::compose(left,right),third).apply(s,{2}),b=fx::compose(left,fx::compose(right,third)).apply(s,{2});
        require(a.first==b.first,"three-region hybrid control association"); close(a.second[0],b.second[0],"three-region hybrid continuous association");
    }
    auto mismatch=right; ++mismatch.control.identity.domain.low;
    rejects<std::invalid_argument>([&] { fx::compose(left,mismatch); },"control vocabulary mismatch");
    mismatch=right; mismatch.transition[1]=3;
    rejects<std::invalid_argument>([&] { fx::compose(left,mismatch); },"invalid transition");
    rejects<std::invalid_argument>([&] { left.apply(3,{1}); },"incoming control bounds");
    auto huge=left; huge.A[0][0][0]=std::numeric_limits<double>::max();
    rejects<std::overflow_error>([&] { huge.apply(0,{2}); },"hybrid overflow");
}
int main() try {
    existing_owners(); accumulated(); hybrid();
    std::cout<<"PASS native composable effects: "<<checks<<" checks\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
