#include <Cellerator/compiler/diagnostics/build_the_provenance_graph_model_v1.hh>
#include <cassert>
int main(){using namespace cellerator::compiler::diagnostics::v1;provenance_graph g;for(unsigned i=0;i<11;++i)g.nodes.push_back({i+1,static_cast<provenance_kind>(i)});for(unsigned i=1;i<11;++i)g.edges.push_back({i,i+1});assert(valid_provenance_graph(g));auto f=query_provenance(g,1,false),r=query_provenance(g,11,true);assert(f.size()==11&&f.back()==11);assert(r.size()==11&&r.back()==1);g.edges.push_back({99,1});assert(!valid_provenance_graph(g));}
