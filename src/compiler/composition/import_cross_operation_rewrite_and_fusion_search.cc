#include <Cellerator/compiler/composition/import_cross_operation_rewrite_and_fusion_search_v1.hh>
namespace Cellerator::compiler::composition {
std::vector<connected_rewrite_v1> select_connected_rewrites_v1(const std::vector<connected_rewrite_v1>&xs){std::vector<connected_rewrite_v1>r;for(const auto&x:xs)if(!x.name.empty()&&!x.traversal.empty()&&!x.persistent_order.empty()&&!x.common_output.empty()&&!x.partial_tree.empty()&&x.exact&&(!x.fusion||x.field_authorized)&&x.rewritten_cost<x.original_cost)r.push_back(x);return r;}
}
