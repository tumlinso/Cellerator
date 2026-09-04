#include <Cellerator/compiler/composition/import_workload_family_representation_v1.hh>
namespace Cellerator::compiler::composition {
bool validate_workload_family_v1(const workload_family_v1&f,std::string*e){auto fail=[&](const char*m){if(e)*e=m;return false;};if(f.semantic_ir_family.empty()||f.profile_family.empty()||f.target_class.empty())return fail("semantic IR, profile, and target families are required");if(f.semantic_ir_family.find('/')!=std::string::npos||f.profile_family.find('/')!=std::string::npos)return fail("storage paths are forbidden");if(!f.recurrence||f.objectives.empty())return fail("recurrence and objectives are required");for(const auto&o:f.objectives)if(o.metric.empty()||o.weight<0)return fail("invalid objective");return true;}
}
