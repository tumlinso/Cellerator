#include <Cellerator/compiler/lto/implement_cellerator_link_driver_mode_v1.hh>
#include <algorithm>
namespace cellerator::compiler::lto::v1 {
link_driver_status_v1 build_cellerator_link_driver_plan_v1(const link_driver_request_v1&q,link_driver_plan_v1*r)noexcept{
 if(!r)return link_driver_status_v1::linker_missing;*r={};if(q.conventional_linker.empty())return link_driver_status_v1::linker_missing;
 std::vector<std::string>seen;for(const auto&i:q.inputs){if(i.path.empty()||std::find(seen.begin(),seen.end(),i.path)!=seen.end())return link_driver_status_v1::duplicate_input;seen.push_back(i.path);if(i.role!=link_input_role_v1::plain_cpp){if(i.field.high==0&&i.field.low==0)return link_driver_status_v1::field_identity_missing;r->program_ceir_fields.push_back(i.field);}}
 if(r->program_ceir_fields.size()>1&&q.authorization.authority==program_planning_authority_v1::none)return link_driver_status_v1::authorization_missing;
 r->final_linker_arguments.push_back(q.conventional_linker);r->final_linker_arguments.insert(r->final_linker_arguments.end(),q.linker_options.begin(),q.linker_options.end());
 if(r->program_ceir_fields.size()>1){r->authorized_lto_ran=true;r->replacement_objects.push_back("cellerator.program-lto.o");}
 for(const auto&i:q.inputs)if(i.role==link_input_role_v1::plain_cpp||!r->authorized_lto_ran)r->final_linker_arguments.push_back(i.path);
 r->final_linker_arguments.insert(r->final_linker_arguments.end(),r->replacement_objects.begin(),r->replacement_objects.end());return link_driver_status_v1::valid;}
}
