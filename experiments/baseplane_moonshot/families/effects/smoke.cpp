#include <ce_moon/effects.hpp>
#include <iostream>
#include <random>
using namespace ce_moon::effects;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static void near(double a,double b){require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<1e-10*(1+std::abs(a)+std::abs(b)),"numerical mismatch");}
template<class E,class F>static void rejects(F f){bool caught=false;try{f();}catch(const E&){caught=true;}require(caught,"invalid input accepted");}
template<std::size_t B,std::size_t K>static std::array<double,B*K> dense_apply(const BlockAffine<B,K>& op,const std::array<double,B*K>& x){
    std::array<std::array<double,B*K>,B*K> dense{};
    for(std::size_t b=0;b<B;++b)for(std::size_t i=0;i<K;++i)for(std::size_t j=0;j<K;++j)dense[b*K+i][op.p[b]*K+j]=op.matrix[b][i][j];
    std::array<double,B*K> result{};
    for(std::size_t i=0;i<B*K;++i){result[i]=op.bias[i/K][i%K];for(std::size_t j=0;j<B*K;++j)result[i]+=dense[i][j]*x[j];}
    return result;
}
int main(){try{
    Weighted<2> a,b;a.weight={{{.6,.4},{.2,.8}}};b.weight={{{.1,.9},{.9,.1}}};
    auto probability=compose(a,b);near(probability.weight[0][0],.42);
    auto pruned=prune_below(a,.5);require(pruned.approximate&&pruned.removed==2,"pruning receipt");near(pruned.removed_probability_mass,.6);
    near(compose(pruned.effect,b).weight[0][0],.06);
    a.algebra=b.algebra=Algebra::max_plus;
    a.weight={{{std::log(.6),std::log(.4)},{std::log(.2),std::log(.8)}}};
    b.weight={{{std::log(.1),std::log(.9)},{std::log(.9),std::log(.1)}}};
    near(std::exp(compose(a,b).weight[0][0]),.36);
    require(compose(a,Weighted<2>::identity(Algebra::max_plus)).weight==a.weight,"max-plus identity");
    Weighted<2> boolean;boolean.algebra=Algebra::boolean;boolean.weight={{{1,1},{0,1}}};
    require(compose(boolean,boolean).weight==boolean.weight,"Boolean alternatives");
    rejects<std::invalid_argument>([&]{compose(a,boolean);});
    rejects<std::invalid_argument>([]{Weighted<2>x;x.weight[0][0]=-1;x.validate();});
    rejects<std::invalid_argument>([]{Weighted<2>x;x.algebra=Algebra::boolean;x.weight[0][0]=.2;x.validate();});
    rejects<std::overflow_error>([]{Weighted<2>x;x.weight[0][0]=std::numeric_limits<double>::max();compose(x,x);});
    std::mt19937 rng(2701);std::uniform_real_distribution<double> dist(-.7,.7);
    for(unsigned trial=0;trial<80;++trial){
        auto left=BlockAffine<3,2>::identity(),right=left,third=left;std::array<double,6>x{};
        for(unsigned block=0;block<3;++block){left.p[block]=(block+1)%3;right.p[block]=(block+2)%3;
            for(unsigned i=0;i<2;++i){left.bias[block][i]=dist(rng);right.bias[block][i]=dist(rng);third.bias[block][i]=dist(rng);
                for(unsigned j=0;j<2;++j){left.matrix[block][i][j]=dist(rng);right.matrix[block][i][j]=dist(rng);third.matrix[block][i][j]=dist(rng);}}}
        for(auto& v:x)v=dist(rng);
        auto composed=compose(compose(left,right),third);
        auto y=ce_moon::effects::apply(composed,x);
        auto reference=dense_apply(third,dense_apply(right,dense_apply(left,x)));
        for(unsigned i=0;i<6;++i)near(y[i],reference[i]);
    }
    auto mono_a=ce_moon::MonomialAffine<2>::identity(),mono_b=mono_a;mono_a.b[0]=2;mono_b.d[0]=3;
    std::array<double,2>x{1,2};
    near(ce_moon::apply(ce_moon::compose(mono_a,mono_b),x)[0],9.);
    near(ce_moon::apply(ce_moon::compose(mono_b,mono_a),x)[0],5.);
    auto invalid=BlockAffine<2,2>::identity();invalid.p[1]=0;
    rejects<std::invalid_argument>([&]{invalid.validate();});
    Jet square{1,1,2,1,.5};auto fourth=compose(square,square);
    auto inside=query(fourth,1.1),outside=query(fourth,2.);
    require(inside.in_radius&&!inside.certified_error_bound&&!outside.in_radius,"jet trust tags");
    near(inside.estimate,1.46);near(std::pow(1.1,4)-inside.estimate,.0041);
    rejects<std::invalid_argument>([&]{compose(square,Jet{0,0,1,0,.5});});
    rejects<std::overflow_error>([]{query(Jet{0,0,std::numeric_limits<double>::max(),0,1},2.);});
    std::vector<ce_moon::MonomialAffine<2>> operators(11,ce_moon::MonomialAffine<2>::identity());
    for(unsigned i=0;i<operators.size();++i){operators[i].p={i%2,(i+1)%2};operators[i].d={.9,.8};operators[i].b={.1*i,-.2*i};}
    Checkpoints<2> checkpoints(operators,4);
    for(unsigned boundary=0;boundary<=operators.size();++boundary){
        auto l=x,r=x;
        for(unsigned i=0;i<boundary;++i)l=ce_moon::apply(operators[i],l);
        for(unsigned i=boundary;i<operators.size();++i)r=ce_moon::apply(operators[i],r);
        std::size_t lp=0,rp=0;auto lc=checkpoints.from_left(boundary,x,&lp),rc=checkpoints.from_right(boundary,x,&rp);
        for(unsigned i=0;i<2;++i){near(lc[i],l[i]);near(rc[i],r[i]);}
        require(lp<4&&rp<4,"checkpoint bounded replay");
    }
    rejects<std::invalid_argument>([&]{Checkpoints<2> bad(operators,0);});
    rejects<std::out_of_range>([&]{checkpoints.from_left(12,x);});
    auto retained=ce_moon::lift(2.,8.);auto dropped=ce_moon::unlift({retained.coarse,0.});
    require(dropped.first!=2.,"omitted residual loses information");near(ce_moon::unlift(retained).first,2.);
    unsigned ambiguous=0;
    for(double value:{.1,.49,.51,.9}){
        auto coarse=precision_interval(value,1);require(coarse.lower<=value&&value<=coarse.upper,"interval enclosure");
        if(coarse.lower<=.5&&coarse.upper>.5){++ambiguous;auto refined=precision_interval(value,8);require(refined.lower>.5||refined.upper<=.5,"refinement decision");}
    }
    require(ambiguous==2,"ambiguous refinement count");
    rejects<std::invalid_argument>([]{precision_interval(1.1,3);});
    std::cout<<"{\"status\":\"effects_pass\",\"block_random_cases\":80,\"jet_truncation_error\":0.0041,\"checkpoint_maps\":"<<checkpoints.retained_maps()<<",\"gpu_run\":false}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
