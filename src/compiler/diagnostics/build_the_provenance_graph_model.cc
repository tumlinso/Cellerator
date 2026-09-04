#include <Cellerator/compiler/diagnostics/build_the_provenance_graph_model_v1.hh>
#include <set>
namespace cellerator::compiler::diagnostics::v1 {
bool valid_provenance_graph(const provenance_graph&g) noexcept{std::set<std::uint64_t> ids;for(auto n:g.nodes)if(!n.id||!ids.insert(n.id).second)return false;for(auto e:g.edges)if(!ids.count(e.from)||!ids.count(e.to)||e.from==e.to)return false;return true;}
std::vector<std::uint64_t> query_provenance(const provenance_graph&g,std::uint64_t root,bool reverse){if(!valid_provenance_graph(g))return{};std::vector<std::uint64_t> out{root};std::set<std::uint64_t> seen{root};for(std::size_t i=0;i<out.size();++i)for(auto e:g.edges){auto a=reverse?e.to:e.from,b=reverse?e.from:e.to;if(a==out[i]&&seen.insert(b).second)out.push_back(b);}return out;}
}
