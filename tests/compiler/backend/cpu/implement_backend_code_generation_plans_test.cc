#include <Cellerator/compiler/backend/implement_backend_code_generation_plans_v1.hh>

#include <cassert>

namespace cb = cellerator::compiler::backend::v1;

int main() {
    cb::backend_codegen_plan_v1 plan;
    plan.target.triple = {"x86_64-unknown-linux-gnu", 24};
    plan.generated_files = {{"module.cc", "int answer(){return 42;}\n"}};
    plan.embedded_data = {{"ceir_blob", {std::byte{1}, std::byte{2}}}};
    plan.compile_jobs = {{"module.cc", "module.o", {"-std=c++17", "-O2"}}};
    plan.link_jobs = {{"module.so", {"module.o"}, {"libCellerator"}}};
    plan.source_maps = {{"module.cc", 1, 17, 4}};
    plan.keep_temporary_artifacts = true;
    assert(cb::validate_backend_codegen_plan_v1(plan) ==
           cb::backend_codegen_plan_status_v1::valid);
    const auto expected =
        "target=x86_64-unknown-linux-gnu keep_temps=1\n"
        "file module.cc 25\n"
        "data ceir_blob 2\n"
        "compile module.cc -> module.o\n"
        "link module.so 1 1\n"
        "map module.cc:1 <- 17:4\n";
    assert(cb::snapshot_backend_codegen_plan_v1(plan) == expected);
    auto dangling = plan;
    dangling.compile_jobs[0].source_path = "missing.cc";
    assert(cb::validate_backend_codegen_plan_v1(dangling) ==
           cb::backend_codegen_plan_status_v1::dangling_job_input);
}
