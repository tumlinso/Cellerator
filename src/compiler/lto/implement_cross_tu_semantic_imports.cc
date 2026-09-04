#include <Cellerator/compiler/lto/implement_cross_tu_semantic_imports_v1.hh>
#include <algorithm>
namespace cellerator::compiler::lto::v1 {
semantic_import_status_v1 import_cross_tu_semantic_field_v1(const semantic_import_request_v1&q,const std::vector<exported_semantic_field_v1>&fs,imported_semantic_field_v1*r)noexcept{auto it=std::find_if(fs.begin(),fs.end(),[&](const auto&f){return f.identity.high==q.field.high&&f.identity.low==q.field.low;});if(it==fs.end()||!r)return semantic_import_status_v1::field_not_found;for(const auto&e:it->required_extensions)if(std::find(q.supported_extensions.begin(),q.supported_extensions.end(),e)==q.supported_extensions.end())return semantic_import_status_v1::extension_unsupported;if(q.depth==semantic_import_depth_v1::full_body&&it->body.empty())return semantic_import_status_v1::body_unavailable;r->field=*it;r->body_loaded=q.depth==semantic_import_depth_v1::full_body;if(!r->body_loaded)r->field.body.clear();return semantic_import_status_v1::valid;}
}
