#include <Cellerator/compiler/lto/implement_cross_tu_semantic_imports_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
int main(){exported_semantic_field_v1 f{{1,1},{2,1},{3,1},"shared","relation genes cells","body",{"ext"}};imported_semantic_field_v1 r;semantic_import_request_v1 q{{1,1},semantic_import_depth_v1::summary,{"ext"}};assert(import_cross_tu_semantic_field_v1(q,{f},&r)==semantic_import_status_v1::valid&&!r.body_loaded&&r.field.body.empty());q.depth=semantic_import_depth_v1::full_body;assert(import_cross_tu_semantic_field_v1(q,{f},&r)==semantic_import_status_v1::valid&&r.body_loaded&&r.field.source.high==2);q.supported_extensions.clear();assert(import_cross_tu_semantic_field_v1(q,{f},&r)==semantic_import_status_v1::extension_unsupported);}
