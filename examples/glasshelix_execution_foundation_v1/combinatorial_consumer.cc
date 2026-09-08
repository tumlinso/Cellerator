#include <Cellerator/native_foundation.hh>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
int main() try {
 namespace ce=cellerator::native_foundation;
 ce::session session(ce::device::cuda);
 if(session.actual_device()!=ce::device::cuda)throw std::runtime_error("CUDA required; no fallback");
 constexpr std::size_t width=33;
 const std::array<std::uint32_t,3> index{0,1,2};
 // z0=x0; z1=x1; z2=x2; z3=tanh(z0); z4=z1*z1; z5=1;
 // z6=z4+z5; z7=z1/z6; z8=z3*z7; z9=z8*z2.
 const std::array<ce::term,10> terms{{{ce::op::argument,0},{ce::op::argument,1},{ce::op::argument,2},{ce::op::tanh,0},{ce::op::multiply,1,1},{ce::op::constant,0,0,1},{ce::op::add,4,5},{ce::op::divide,1,6},{ce::op::multiply,3,7},{ce::op::multiply,8,2}}};
 ce::prepared_nary plan(session,{index,terms,3},width);
 auto a=plan.bind_gain(1.5f),b=plan.bind_gain(2.0f),zero=plan.bind_gain(0.0f);
 std::vector<float> values(3*width);
 for(std::size_t i=0;i<width;++i){values[3*i]=.1f+.01f*i;values[3*i+1]=.2f+.02f*i;values[3*i+2]=.8f;}
 auto ra=plan.evaluate(a,values),rb=plan.evaluate(b,values),rd=plan.gain_jvp(zero,values,1.0f);
 ra.ready.wait();rb.ready.wait();rd.ready.wait();auto ya=ra.download(),yb=rb.download(),dy=rd.download();
 if(ya.size()!=width||yb.size()!=width||dy.size()!=width)throw std::runtime_error("tail width lost");
 for(std::size_t i=0;i<width;++i){const double x=values[3*i],y=values[3*i+1],z=values[3*i+2];const double r=std::tanh(x)*y/(1+y*y)*z;
   if(std::abs(ya[i]-1.5*r)>2e-6||std::abs(yb[i]-2*r)>2e-6||std::abs(dy[i]-r)>2e-6)throw std::runtime_error("n-ary/value/derivative mismatch");}
 const auto count=plan.inspect();if(count.structure_preparations!=1||count.value_instances!=3||count.hot_allocations!=0)throw std::runtime_error("preparation or hot-allocation contract violated");
 std::cout<<"Cellerator standalone n-ary consumer passed on CUDA, width 33\n";return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
