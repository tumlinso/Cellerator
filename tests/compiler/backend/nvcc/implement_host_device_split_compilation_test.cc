#include <Cellerator/compiler/backend/nvcc/implement_host_device_split_compilation_v1.hh>
#include <cassert>
int main(){using namespace cellerator::compiler::backend::nvcc::v1;split_request r{77,"a.cu","a.cc","a.so",{70},compilation_route::whole_cuda};auto whole=make_compilation_graph(r);r.route=compilation_route::split_host_device;auto split=make_compilation_graph(r);assert(whole&&split&&whole->semantic_identity==split->semantic_identity);assert(whole->jobs.size()==1&&split->jobs.size()==3);}
