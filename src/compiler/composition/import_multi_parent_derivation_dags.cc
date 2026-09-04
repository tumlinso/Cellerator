#include <Cellerator/compiler/composition/import_multi_parent_derivation_dags_v1.hh>
#include <algorithm>
#include <functional>
#include <map>
namespace Cellerator::compiler::composition {
derivation_validation_v1 validate_derivation_dag_v1(const derivation_dag_v1&g){derivation_validation_v1 r;std::map<std::string,const derivation_node_v1*> nodes;for(const auto&n:g.nodes){if(n.id.empty()||n.type.empty()||n.lineage.empty()||(!n.parents.empty()&&n.reconstruction.empty())){r.diagnostic="incomplete node "+n.id;return r;}if(!nodes.emplace(n.id,&n).second){r.diagnostic="duplicate node "+n.id;return r;}}
 std::map<std::string,int> mark;std::vector<std::string> stack;std::function<bool(const std::string&)> visit=[&](const std::string&id){if(mark[id]==2)return true;if(mark[id]==1){auto at=std::find(stack.begin(),stack.end(),id);r.diagnostic="cycle";for(;at!=stack.end();++at)r.diagnostic+=" -> "+*at;r.diagnostic+=" -> "+id;return false;}mark[id]=1;stack.push_back(id);for(const auto&p:nodes[id]->parents){if(!nodes.count(p.node)){r.diagnostic="missing parent "+p.node;return false;}if(p.production.empty()){r.diagnostic="missing production for "+p.node;return false;}if(!visit(p.node))return false;}stack.pop_back();mark[id]=2;r.topological_order.push_back(id);return true;};
 for(const auto&[id,node]:nodes){(void)node;if(!visit(id))return r;}r.valid=true;return r;}
}
