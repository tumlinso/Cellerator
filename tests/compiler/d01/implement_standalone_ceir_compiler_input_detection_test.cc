#include <Cellerator/compiler/ir/common/implement_standalone_ceir_compiler_input_detection_v1.hh>

#include <cassert>

using namespace cellerator::compiler::ir;

int main() {
    struct fixture { const char *level; ceir_input_level expected; ceir_resume_stage resume; const char *dump; };
    for (const auto &item : {fixture{"semantic", ceir_input_level::semantic,
             ceir_resume_stage::build_planning, "planning.ceir"},
             fixture{"planning", ceir_input_level::planning,
                 ceir_resume_stage::build_realization, "realization.ceir"},
             fixture{"realization", ceir_input_level::realization,
                 ceir_resume_stage::lower_executable, "executable.ceir"}}) {
        const auto input = std::string("ceir level ") + item.level + " version 1.0\n";
        const auto detected = detect_standalone_ceir("module.ceir", input);
        assert(detected.level == item.expected && detected.resume == item.resume);
        assert(next_ceir_dump_name(detected.level) == item.dump);
    }
    assert(detect_standalone_ceir("module.cpp", "ceir level semantic version 1.0").resume
        == ceir_resume_stage::reject);
    assert(detect_standalone_ceir("module.ceir", "ceir level semantic version 2.0").resume
        == ceir_resume_stage::reject);
    assert(detect_standalone_ceir("module.ceir", "semantic.apply").resume
        == ceir_resume_stage::reject);
}
