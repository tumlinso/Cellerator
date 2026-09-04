#include <Cellerator/compiler/pass/implement_pipeline_configuration_syntax_and_apis_v1.hh>

#include <cassert>

namespace cp = cellerator::compiler::pass::v1;

int main() {
    const std::string text = "profile=pbmc;discovery.before:discover,selection.after:audit";
    std::vector<cp::pipeline_configuration_v1> configurations;
    for (auto origin : {cp::pipeline_configuration_origin_v1::source_directive,
             cp::pipeline_configuration_origin_v1::inline_planning_ir,
             cp::pipeline_configuration_origin_v1::command_line,
             cp::pipeline_configuration_origin_v1::profile}) {
        cp::pipeline_configuration_v1 value{};
        assert(cp::parse_pipeline_configuration_v1(text, origin, &value));
        assert(cp::print_pipeline_configuration_v1(value) == text);
        configurations.push_back(std::move(value));
    }
    for (const auto& value : configurations)
        assert(value.profile == configurations[0].profile
            && value.passes == configurations[0].passes);
    const auto built = cp::pipeline_builder_v1{}.profile("pbmc")
        .add({cp::pipeline_phase_v1::discovery, cp::interception_side_v1::before}, "discover")
        .add({cp::pipeline_phase_v1::selection, cp::interception_side_v1::after}, "audit")
        .build();
    assert(cp::print_pipeline_configuration_v1(built) == text);
    cp::pipeline_configuration_v1 invalid{};
    assert(!cp::parse_pipeline_configuration_v1("unknown.before:x",
        cp::pipeline_configuration_origin_v1::command_line, &invalid));
}
