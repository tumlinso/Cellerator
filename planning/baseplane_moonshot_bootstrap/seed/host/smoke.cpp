#include <bp_moon/reference.hpp>
#include <iostream>
#include <random>
using namespace bp_moon;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static void near(double a,double b){require(std::abs(a-b)<=1e-11*(1+std::abs(a)+std::abs(b)),"numeric mismatch");}
int main(){try{
    std::mt19937 rng(2701);unsigned checks=0;
    for(unsigned i=0;i<100;++i){u32 a=rng(),b=rng(),c=rng();require(lut3(a,b,c,0x80)==(a&b&c),"LUT AND");require(lut3(a,b,c,0x96)==(a^b^c),"LUT XOR");}++checks;
    for(u32 m:{u32{0},u32{1},u32{0x80000000},u32{0x5a51},~u32{0}})for(unsigned r=0;r<popcount(m);++r)require(rank_before(m,select_bit(m,r))==r,"rank/select");++checks;
    CounterPlanes cp;for(unsigned i=0;i<15;++i)cp.add(0x5a5a5a5a);require(!cp.overflow&&cp.at(1)==15&&cp.at(0)==0,"counter planes");cp.add(0x5a5a5a5a);require(cp.overflow==0x5a5a5a5a,"counter overflow");++checks;
    PackedFixture p(std::string(31,'A')+"GNCGT");require(p.words.size()==2&&!p.is_valid(32)&&!p.is_valid(36),"packed validity");auto pl=p.planes(1);require(pl.valid==14,"tail validity");++checks;
    auto a=summarize_sequence("ACGTAC"),b=summarize_sequence("TGGC"),c=summarize_sequence("NAC");require(compose(compose(a,b),c).to==compose(a,compose(b,c)).to,"DFA associativity");require(compose(a,b).to==summarize_sequence("ACGTACTGGC").to,"DFA concatenation");++checks;
    CountedDfa32 ca,cb;ca.state=a;cb.state=b;for(unsigned i=0;i<32;++i){ca.count[i]=i;cb.count[i]=2*i;}auto cc=compose(ca,cb);for(unsigned i=0;i<32;++i)require(cc.count[i]==i+2*a.to[i],"counted DFA");++checks;
    auto ma=MonomialAffine<8>::identity(),mb=ma,mc=ma;std::array<double,8>x{};
    for(unsigned i=0;i<8;++i){ma.p[i]=(i+1)%8;mb.p[i]=(i+3)%8;mc.p[i]=(i+2)%8;ma.d[i]=.5+.05*i;mb.b[i]=.1*i;mc.d[i]=.8;x[i]=i;}
    auto y=bp_moon::apply(compose(ma,mb),x),z=bp_moon::apply(mb,bp_moon::apply(ma,x));for(unsigned i=0;i<8;++i)near(y[i],z[i]);
    auto q=bp_moon::apply(compose(compose(ma,mb),mc),x),r=bp_moon::apply(compose(ma,compose(mb,mc)),x);for(unsigned i=0;i<8;++i)near(q[i],r[i]);++checks;
    std::vector<double> values{1,2,-3,7,0,1,9,2,4};ResidualTree tree(values);for(unsigned i=0;i<values.size();++i)near(tree.reconstruct(i),values[i]);require(tree.above(5)==std::vector<std::size_t>({3,6}),"refinement query");++checks;
    std::vector<Keyed> keys;for(unsigned i=0;i<97;++i)keys.push_back({i%3,i});auto groups=rendezvous(keys);require(groups.size()==3&&groups[0].ids.size()==33&&directed_pair_count(groups)==3040,"global peers across warp boundary");++checks;
    auto supports=canonical_support({{1,0,2},{1,5,6},{1,1,3},{2,1,3}});require(supports.size()==3&&supports[0].end==3&&supports[1].begin==5,"source support gaps");++checks;
    ExactInterner interner;require(interner.intern("ACGT")==interner.intern("ACGT")&&interner.intern("ACGT")!=interner.intern("ACGN"),"exact interning");
    auto dirty=dirty_closure({{2},{2},{3},{}},{0});require(dirty==std::vector<unsigned>({0,2,3}),"dirty cone");require(!(Versions{1,2,3,4,5}==Versions{1,2,3,4,6}),"version key");++checks;
    Relation16 ra{},rb{};for(unsigned i=0;i<16;++i){ra[i][(i+1)%16]=1;rb[i][(i+2)%16]=1;}auto rc=compose_relation(ra,rb);for(unsigned i=0;i<16;++i)require(rc[i][(i+3)%16]==1,"finite relation");++checks;
    std::cout<<"{\"status\":\"host_smoke_pass\",\"reference_case_groups_checked\":"<<checks<<",\"gpu_executed\":false,\"biological_validation\":false}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
