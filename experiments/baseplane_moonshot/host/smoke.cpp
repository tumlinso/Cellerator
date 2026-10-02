#include <ce_moon/reference.hpp>
#include <iostream>
#include <random>

using namespace ce_moon;
static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
static void near(double a, double b) {
    require(std::abs(a-b) <= 1e-11*(1+std::abs(a)+std::abs(b)), "numeric mismatch");
}
template<class Exception, class F> static void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const Exception&) { caught = true; }
    require(caught, "invalid input accepted");
}
int main() { try {
    std::mt19937 rng(2701);
    for (unsigned trial=0; trial<100; ++trial) {
        Dfa32 a,b,c;
        for (unsigned i=0; i<32; ++i) { a.to[i]=rng()%32; b.to[i]=rng()%32; c.to[i]=rng()%32; }
        require(compose(compose(a,b),c).to == compose(a,compose(b,c)).to, "DFA associativity");
        require(compose(a,Dfa32::identity()).to == a.to, "DFA identity");
        auto result=compose(a,b);
        for (unsigned i=0; i<32; ++i) require(result.to[i]==b.to[a.to[i]], "DFA operational order");
        CountedDfa32 ca,cb; ca.state=a; cb.state=b;
        for (unsigned i=0; i<32; ++i) { ca.count[i]=i; cb.count[i]=2*i; }
        auto cc=compose(ca,cb);
        for (unsigned i=0; i<32; ++i) require(cc.count[i]==i+2*a.to[i], "counted composition");
    }
    auto a=MonomialAffine<8>::identity(),b=a,c=a;
    std::array<double,8> x{};
    for (unsigned i=0; i<8; ++i) { a.p[i]=(i+1)%8; b.p[i]=(i+3)%8; c.p[i]=(i+2)%8; a.d[i]=.5+.05*i; b.b[i]=.1*i; c.d[i]=.8; x[i]=i; }
    auto y=ce_moon::apply(compose(a,b),x),z=ce_moon::apply(b,ce_moon::apply(a,x));
    auto q=ce_moon::apply(compose(compose(a,b),c),x),r=ce_moon::apply(compose(a,compose(b,c)),x);
    for (unsigned i=0; i<8; ++i) { near(y[i],z[i]); near(q[i],r[i]); }
    std::vector<double> values{1,2,-3,7,0,1,9,2,4};
    ResidualTree tree(values);
    for (unsigned i=0; i<values.size(); ++i) near(tree.reconstruct(i),values[i]);
    require(tree.above(5)==std::vector<std::size_t>({3,6}), "residual refinement");
    auto pair=unlift(lift(3.,8.,.7,.2),.7,.2); near(pair.first,3.); near(pair.second,8.);
    Relation16 ra{},rb{};
    for (unsigned i=0; i<16; ++i) { ra[i][(i+1)%16]=1; rb[i][(i+2)%16]=1; }
    auto rc=compose_relation(ra,rb);
    for (unsigned i=0; i<16; ++i) for(unsigned j=0;j<16;++j) require(rc[i][j]==(j==(i+3)%16), "relation composition");
    rejects<std::invalid_argument>([]{ auto bad=Dfa32::identity(); bad.to[2]=32; compose(bad,Dfa32::identity()); });
    rejects<std::overflow_error>([]{ CountedDfa32 a,b; a.count[0]=std::numeric_limits<u64>::max(); b.count[0]=1; compose(a,b); });
    rejects<std::invalid_argument>([]{ ResidualTree tree({}); });
    rejects<std::invalid_argument>([]{ ResidualTree tree({std::numeric_limits<double>::infinity()}); });
    const double largest=std::numeric_limits<double>::max();
    rejects<std::overflow_error>([&]{ ResidualTree tree({largest,-largest}); });
    rejects<std::overflow_error>([&]{ lift(largest,0.,2.,.5); });
    rejects<std::overflow_error>([&]{ unlift({largest,-largest}); });
    rejects<std::invalid_argument>([]{ lift(1.,2.,std::numeric_limits<double>::quiet_NaN()); });
    ResidualTree large_finite({largest/4.,largest/2.});
    near(large_finite.reconstruct(0)/(largest/4.),1.);
    near(large_finite.reconstruct(1)/(largest/2.),1.);
    rejects<std::out_of_range>([&]{ tree.reconstruct(values.size()); });
    rejects<std::invalid_argument>([]{ auto bad=MonomialAffine<2>::identity(); bad.p[1]=0; bad.validate(); });
    rejects<std::invalid_argument>([]{ Relation16 a{},b{}; a[0][0]=2; compose_relation(a,b); });
    std::cout << "{\"status\":\"host_smoke_pass\",\"random_dfa_cases\":100,\"gpu_executed\":false,\"biological_validation\":false}\n";
    return 0;
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
