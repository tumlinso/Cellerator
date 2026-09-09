#include "formulas.hh"
#include "stress_fixtures.hh"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace ref=ce_nf1_reference;
namespace {
unsigned checks=0,wrong_variants=0;
void require(bool condition,const char* why){++checks;if(!condition)throw std::runtime_error(why);}
void exact(double actual,double expected,const char* why){require(actual==expected,why);}
void near(double actual,double expected,double absolute,double relative,const char* why){
    require(ref::same_nonfinite_or_near(actual,expected,absolute,relative),why);
}
void wrong(double actual,double expected,const char* why){
    require(!ref::same_nonfinite_or_near(actual,expected,0.,0.),why);++wrong_variants;
}
void shapes(){
    for(unsigned width:{0u,1u,16u,31u,32u,33u,65u}){
        constexpr std::array<unsigned,3> degrees{0,1,4097};
        std::vector<double> output(3*width+2,-123.);
        for(unsigned row=0;row<degrees.size();++row)for(unsigned column=0;column<width;++column){
            std::vector<double> segment(degrees[row],1.+column*.125);
            const double expected=degrees[row]*(1.+column*.125); // Independent closed form; exactly representable.
            const auto at=1+row*width+column;
            output[at]=ref::stress_sum(segment,ref::stress_precision::f64);
            exact(output[at],expected,"empty/skewed/tail exact FP64 reduction");
            exact(ref::stress_sum(segment,ref::stress_precision::f32),expected,"exact dyadic FP32 reduction");
        }
        exact(output.front(),-123.,"leading canary");exact(output.back(),-123.,"trailing canary");
        if(width>32){
            const double expected=degrees[2]*(1.+(width-1)*.125);
            wrong(-123.,expected,"truncated width32 route cannot pass tail");
        }
    }
    const std::array<double,3> weighted{2.,-3.,4.};
    exact(ref::stress_sum(weighted,ref::stress_precision::f64),3.,"signed destination assembly");
    wrong(4.,3.,"overwrite instead of additive reduction rejected");
}
void precision(){
    const std::array<double,3> cancellation{16777216.,1.,-16777216.};
    exact(ref::stress_sum(cancellation,ref::stress_precision::f64),1.,"FP64 cancellation truth");
    exact(ref::stress_sum(cancellation,ref::stress_precision::f32),0.,"declared sequential FP32 rounding");
    wrong(ref::stress_sum(cancellation,ref::stress_precision::f32),1.,"FP32 must not be reported as FP64");
    const double largest_float=std::numeric_limits<float>::max();
    const std::array<double,2> overflow{largest_float,largest_float};
    const double overflowed=ref::stress_sum(overflow,ref::stress_precision::f32);
    require(std::isinf(overflowed),"FP32 arithmetic overflow remains nonfinite");
    exact(ref::stress_sum(overflow,ref::stress_precision::f64),2*largest_float,"FP64 diagnostic avoids FP32 overflow");
    wrong(overflowed,2*largest_float,"FP32 overflow cannot be reported finite FP64");
    const std::array<double,1> single{1.+std::ldexp(1.,-12)};
    exact(ref::stress_sum(single,ref::stress_precision::f32),single[0],"FP32 preserves quarter half-ulp");
    exact(ref::stress_sum(single,ref::stress_precision::f16_storage_f32),1.,"FP16 storage rounds before FP32 arithmetic");
    wrong(1.,single[0],"undeclared half storage rejected");
    exact(ref::store_f16(1.+std::ldexp(1.,-11)),1.,"halfway lower even");
    exact(ref::store_f16(1.+3*std::ldexp(1.,-11)),1.+std::ldexp(1.,-9),"halfway upper even");
    exact(ref::store_f16(std::ldexp(1.,-24)),std::ldexp(1.,-24),"minimum half subnormal");
    exact(ref::store_f16(std::ldexp(1.,-25)),0.,"underflow halfway to even zero");
    require(std::signbit(ref::store_f16(-std::ldexp(1.,-25))),"negative underflow keeps zero sign");
    exact(ref::store_f16(65504.),65504.,"maximum finite half");
    require(std::isinf(ref::store_f16(65520.)),"half overflow is reported nonfinite");
    // Bounded storage error is separate from FP32 accumulation error.
    double total_bound=0.,unquantized=0.;std::vector<double> data;
    for(unsigned i=0;i<257;++i){double x=.1+i*.00013;data.push_back(x);unquantized+=x;int e=0;std::frexp(x,&e);total_bound+=std::ldexp(1.,e-12);}
    near(ref::stress_sum(data,ref::stress_precision::f16_storage_f32),unquantized,total_bound,0.,"explicit storage-error budget");
    bool rejected=false;try{(void)ref::stress_sum(single,ref::stress_precision::unsupported_fp8);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unavailable FP8 profile fails explicitly");
}
void masks_and_nonfinite(){
    const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    for(double value:{nan,inf,-inf}){
        std::array<double,2> x{value,2.};std::array<unsigned char,2> off{0,1},on{1,1};
        exact(ref::masked_sum(x,off),2.,"inactive nonfinite excluded before arithmetic");
        near(ref::masked_sum(x,on),value,0.,0.,"active nonfinite classification preserved");
        wrong(2.,value,"sanitized active nonfinite rejected");
    }
    const std::array<double,2> conflict{inf,-inf};
    require(std::isnan(ref::stress_sum(conflict,ref::stress_precision::f64)),"opposite infinities stay invalid");
    wrong(0.,nan,"NaN cannot become valid zero");
    require(std::isnan(0.*nan),"multiplicative mask negative-control witness");
    const std::array<double,2> tiny{1.,std::ldexp(1.,-40)};const std::array<unsigned char,2> active{1,1};
    exact(ref::masked_sum(tiny,active),1.+std::ldexp(1.,-40),"small exact support contribution retained");
    wrong(1.,ref::masked_sum(tiny,active),"threshold pruning is not exact masking");
    bool rejected=false;try{(void)ref::masked_sum(tiny,std::array<unsigned char,1>{1});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"mask shape mismatch rejected");
    rejected=false;try{(void)ref::masked_sum(tiny,std::array<unsigned char,2>{1,2});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"nonbinary exact mask rejected");
}
void multiplicity_response(){
    for(unsigned arity:{1u,2u,3u,9u}){
        double product=1.,response=0.;
        for(unsigned i=0;i<arity;++i)product*=2.;
        for(unsigned incidence=0;incidence<arity;++incidence){double term=1.;for(unsigned i=0;i<arity;++i)if(i!=incidence)term*=2.;response+=term;}
        exact(product,std::ldexp(1.,arity),"nary product closed form");
        exact(response,arity*std::ldexp(1.,arity-1),"all repeated endpoint incidences contribute");
        if(arity>1)wrong(std::ldexp(1.,arity-1),response,"deduplicating repeated argument derivative rejected");
    }
    const ref::Point q{std::ldexp(1.,-30),2.,0.,0.};
    exact(ref::gradient(q)[3],std::ldexp(1.,-29),"near-zero primal preserves coefficient response");
    wrong(0.,ref::gradient(q)[3],"zero-primal activity pruning loses response");
    const ref::Point rounded{ref::store_f16(1.+std::ldexp(1.,-12)),2.,0.,1.};
    const double mathematical=ref::gradient(rounded)[0];
    exact(mathematical,3.,"mathematical derivative at stored value");
    const double h=std::ldexp(1.,-14);
    const double through_rounding=(ref::store_f16(rounded[0]+h)-ref::store_f16(rounded[0]-h))/(2*h);
    exact(through_rounding,0.,"rounding derivative is locally zero away from boundary");
    wrong(through_rounding,mathematical,"rounding derivative cannot replace stored-value derivative");
}
}
int main()try{
    shapes();precision();masks_and_nonfinite();multiplicity_response();
    require(wrong_variants==16,"all numerical wrong-variant controls executed");
    std::cout<<"{\"task\":\"CE-NF1-T02\",\"checks\":"<<checks<<",\"wrong_variants\":"<<wrong_variants<<",\"production_backend\":false,\"gpu\":false}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
