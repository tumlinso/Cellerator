#include "fixtures.hh"
#include <algorithm>
#include <iostream>
using namespace spine_verify;
int main() {
    try {
        double y[5], z[4];
        reference(demo.data(),demo.size(),4,5,demo_input,y);
        reference(demo.data(),demo.size(),4,5,demo_signal,z,true);
        const double ey[]={0,1.25,.5,0,1.5}, ez[]={.75,-4,.75,.125};
        for (int i=0;i<5;++i) require(y[i]==ey[i],"hand forward");
        for (int i=0;i<4;++i) require(z[i]==ez[i],"hand transpose");
        require(!near(y[1],ey[2]),"wrong-index control");
        double lhs=0,rhs=0;
        for(int i=0;i<5;++i) lhs+=y[i]*demo_signal[i];
        for(int i=0;i<4;++i) rhs+=z[i]*demo_input[i];
        require(lhs==rhs,"adjoint supplement");
        reference(duplicate.data(),duplicate.size(),3,4,second_input,y);
        reference(duplicate.data(),duplicate.size(),3,4,second_signal,z,true);
        require(y[0]==3&&y[1]==0&&y[2]==3&&y[3]==-1,"duplicate additive forward");
        require(z[0]==-1.5&&z[1]==-2&&z[2]==3,"independent duplicate transpose");
        auto permuted=duplicate; std::reverse(permuted.begin(),permuted.end());
        double reordered[4];reference(permuted.data(),permuted.size(),3,4,second_input,reordered);
        for(int i=0;i<4;++i)require(y[i]==reordered[i],"logical edge permutation");
        reference(nullptr,0,3,4,nullptr,y);for(int i=0;i<4;++i)require(y[i]==0,"empty support");
        reference(nullptr,0,0,0,nullptr,nullptr);
        const float ones[]={1,1,1};reference(cancellation.data(),3,3,1,ones,y);
        require(y[0]==1,"stored-value cancellation");
        require(!near(std::nan(""),0)&&!near(0,std::nan("")),"NaN control");
        require(!near(INFINITY,0)&&!near(INFINITY,-INFINITY)&&near(INFINITY,INFINITY),"infinity policy");
        require(half_value(1)==std::ldexp(1.0,-24),"half subnormal");
        reference(demo_generation_2.data(),9,4,5,demo_input,y);
        require(!near(y[0],ey[0]),"old generation output control");
        auto bad=second;bad[0].source=3;bool rejected=false;
        try {reference(bad.data(),bad.size(),3,4,second_input,y);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"exact endpoint bounds");
        std::cout<<"independent logical-edge oracle and negative controls passed\n";
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
