#include <Cellerator/compiler/driver/implement_plain_c_passthrough_planning_v1.hh>
#include <stdexcept>
namespace cellerator::compiler::driver {
passthrough_plan_v1 plan_plain_cxx_passthrough_v1(std::string compiler, std::vector<std::string> arguments, bool activated) { if (compiler.empty()) throw std::invalid_argument("downstream compiler is required"); passthrough_plan_v1 out{std::move(compiler), std::move(arguments)}; out.semantic_job_count = activated ? 1u : 0u; return out; }
}  // namespace cellerator::compiler::driver
