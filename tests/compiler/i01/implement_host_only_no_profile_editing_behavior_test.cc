#include <Cellerator/compiler/tooling/implement_host_only_no_profile_editing_behavior_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::tooling;
int main(){auto c=editor_capabilities_v1(false,false);for(int i=0;i<4;++i)assert(c[i].available);assert(!c[4].available&&c[4].reason=="profile not loaded");assert(!c[5].available&&c[5].reason=="CUDA unavailable");auto f=editor_capabilities_v1(true,true);assert(f[4].available&&f[5].available);}
