#include <Cellerator/compiler/tooling/expose_reusable_tooling_snapshot_apis_v1.hh>
int main(){return Cellerator::compiler::tooling::tooling_snapshot_v1({1,"",{}, {}}).revision()==1?0:1;}
