#include <Cellerator/compiler/composition/create_semantic_differential_adapters_v1.hh>
#include <algorithm>
#include <set>
namespace Cellerator::compiler::composition {
semantic_differential_v1 compare_canonical_semantics_v1(const canonical_semantics_v1&a,const canonical_semantics_v1&b,const std::vector<intentional_difference_v1>&allowed){semantic_differential_v1 r;std::set<std::string>keys;for(const auto&x:a)keys.insert(x.first);for(const auto&x:b)keys.insert(x.first);for(const auto&k:keys){auto x=a.find(k),y=b.find(k);if(x!=a.end()&&y!=b.end()&&x->second==y->second)continue;auto permit=std::find_if(allowed.begin(),allowed.end(),[&](const auto&d){return d.field==k&&!d.reason.empty();});if(permit==allowed.end())r.unexpected.push_back(k);else r.intentional.push_back(*permit);}r.equivalent=r.unexpected.empty();return r;}
}
