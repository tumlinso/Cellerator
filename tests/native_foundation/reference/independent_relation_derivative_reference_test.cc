#include "formulas.hh"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace ref=ce_nf1_reference;
namespace {
int checks=0,rejected=0;
void require(bool v,const char* why) {++checks;if(!v)throw std::runtime_error(why);}
bool close(double a,double b,double tolerance=3e-13) {
    return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance*(1+std::abs(b));
}
void near(double a,double b,const char* why,double tolerance=3e-13){require(close(a,b,tolerance),why);}
void reject(double wrong,double expected,const char* why){require(!close(wrong,expected,1e-8),why);++rejected;}
enum class Identity { A,B,C };
struct Slot {Identity id;double value;};
double lookup(const std::array<Slot,3>& slots,Identity id){
    for(const auto& s:slots)if(s.id==id)return s.value;
    throw std::runtime_error("unknown reference identity");
}
// Synthetic packed candidate, independently declared from the scalar referee.
// This is a fault-injection adapter, not a native backend.
struct Edge {Identity from;int output;double weight;};
std::array<double,2> candidate(const std::array<Slot,3>& slots,bool swapped=false) {
    const std::array<Edge,4> edges{{{Identity::C,0,-3},{Identity::A,1,4},
                                  {Identity::B,1,.5},{Identity::A,0,2}}};
    std::array<double,2> result{};
    for(auto e:edges){
        if(swapped){if(e.from==Identity::A)e.from=Identity::C;else if(e.from==Identity::C)e.from=Identity::A;}
        result[e.output]+=e.weight*lookup(slots,e.from);
    }
    return result;
}
void indexing(){
    for(int width:{0,1,16,33}) {
        int count=0;
        for(int i=0;i<width;++i){
            const double A=.2+.02*i,B=-.3+.01*i,C=.8-.005*i;
            const auto expected=ref::relation(A,B,C);
            std::array<Slot,3> slots{{{Identity::A,A},{Identity::B,B},{Identity::C,C}}};
            do {
                const auto got=candidate(slots);
                near(got[0],expected[0],"logical endpoint sink");near(got[1],expected[1],"logical endpoint report");++count;
            }while(std::next_permutation(slots.begin(),slots.end(),[](const Slot&a,const Slot&b){return a.id<b.id;}));
        }
        require(count==6*width,"exact empty/tail permutation inventory");
    }
    const std::array<Slot,3> slots{{{Identity::B,-.3},{Identity::C,.8},{Identity::A,.2}}};
    const auto wrong=candidate(slots,true),truth=ref::relation(.2,-.3,.8);
    reject(wrong[0],truth[0],"endpoint permutation detected independently");
    // A second path sharing the defect agrees; that agreement is not evidence.
    const auto same_bug=candidate(slots,true);
    near(same_bug[0],wrong[0],"shared faulty-map agreement control");
    reject(same_bug[1],truth[1],"independent referee defeats shared faulty map");
    const std::array<double,3> direction{{.4,-.7,.1}};
    const auto forward=ref::relation(direction[0],direction[1],direction[2]);
    const auto back=ref::relation_pullback(.3,-.8);
    near(.3*forward[0]-.8*forward[1],direction[0]*back[0]+direction[1]*back[1]+direction[2]*back[2],"relation adjoint identity");
}
ref::Point shifted(ref::Point q,const ref::Point& d,double step){for(int i=0;i<4;++i)q[i]+=step*d[i];return q;}
void derivatives(){
    const ref::Point q{{.4,-.7,.3,1.2}},v{{.2,.5,-.4,.3}};
    const auto g=ref::gradient(q);
    double dot=0;for(int i=0;i<4;++i)dot+=g[i]*v[i];
    near(dot,ref::first(q,v),"independent gradient and directional expression");
    // Basis finite differences prevent mutually consistent omitted JVP/VJP terms.
    for(int i=0;i<4;++i){
        ref::Point axis{};axis[i]=1.;
        const double h=1e-5;
        const auto fd=(ref::value(shifted(q,axis,h))-ref::value(shifted(q,axis,-h)))/(2*h);
        near(fd,g[i],"all derivative axes finite difference",3e-10);
    }
    const double h=1e-4;
    const double fd2=(ref::value(shifted(q,v,h))-2*ref::value(q)+ref::value(shifted(q,v,-h)))/(h*h);
    near(fd2,ref::second(q,v),"second direction independently differenced",2e-8);
    reject(ref::first(q,v)-q[0]*v[0],ref::first(q,v),"omitted quadratic derivative rejected");
    reject(ref::first(q,v)-q[0]*q[1]*v[3],ref::first(q,v),"omitted coefficient derivative rejected");
    reject(ref::second(q,v)-2*v[3]*(q[1]*v[0]+q[0]*v[1]),ref::second(q,v),"omitted mixed second direction rejected");
    // Repeated argument endpoints: left=right=A, so both incidences contribute.
    const double A=.6,Z=-.2,P=1.3;
    const ref::Point repeated{{A,A,Z,P}};
    const auto rg=ref::gradient(repeated);
    const double expected=(2*P+1)*A;
    near(rg[0]+rg[1],expected,"repeated input endpoint pullback assembly");
    reject(rg[0],expected,"missing repeated endpoint gradient rejected");
    // Zero coefficient kills product primal contribution but not its derivative.
    const ref::Point zero{{.4,-.7,0.,0.}};
    near(ref::gradient(zero)[3],-.28,"zero primal product preserves parameter sensitivity");
}
void nary(){
    // Independently stated unary, binary, and ternary formulas with two outputs.
    // Explicit ordered arguments include one repeated identity and additive ownership.
    const double A=.4,B=-.7,C=.3,P=1.2;
    const double unary=A*A;
    const double binary=B*C;
    const double ternary=ref::value({A,B,C,P});
    const std::array<double,2> assembled{{unary+ternary,binary+ref::value({A,A,C,P})}};
    const double expected0=A*A+P*A*B+std::sin(C)+.5*A*A;
    const double expected1=B*C+(P+.5)*A*A+std::sin(C);
    near(assembled[0],expected0,"irregular arity sum assembly");
    near(assembled[1],expected1,"repeated endpoint and second output");
    reject(ref::value({B,A,C,P})+unary,expected0,"ordered nary roles not commutative");
    reject(ternary,expected0,"overwrite instead of additive ownership rejected");
    require(!close(std::numeric_limits<double>::quiet_NaN(),0.),"nonfinite comparator rejects NaN");
}
}
int main()try{
    indexing();derivatives();nary();require(rejected==8,"all negative controls ran");
    std::cout<<"{\"task\":\"CE-NF1-T01\",\"evidence\":\"executed_host_test\",\"production_backend\":false,\"gpu\":false,\"checks\":"<<checks<<",\"rejected_variants\":"<<rejected<<"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
