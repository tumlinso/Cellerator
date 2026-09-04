#include <Cellerator/compiler/composition/import_induced_grammar_as_experimental_search_v1.hh>
#include <algorithm>
namespace Cellerator::compiler::composition {
induced_grammar_search_v1 search_induced_grammar_v1(std::vector<induced_production_candidate_v1> c,std::size_t bound,double baseline,double confidence){induced_grammar_search_v1 r;std::stable_sort(c.begin(),c.end(),[](const auto&a,const auto&b){return a.confidence>b.confidence;});if(c.size()>bound)c.resize(bound);r.evaluated=c;for(const auto&x:c)if(!x.name.empty()&&!x.evidence.empty()&&!x.verifier.empty()&&x.exact&&x.confidence>=confidence&&x.total_cost<baseline)r.promoted.push_back(x);r.no_promotion=r.promoted.empty();return r;}
}
