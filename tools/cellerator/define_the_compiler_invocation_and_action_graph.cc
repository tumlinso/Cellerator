#include <Cellerator/compiler/driver/define_the_compiler_invocation_and_action_graph_v1.hh>

#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    using namespace cellerator::compiler::driver;
    if (argc != 2) {
        std::cerr << "usage: cellerator-action-graph MODE\n";
        return 2;
    }
    const std::string_view mode = argv[1];
    compiler_invocation_v1 invocation{};
    invocation.input = "input.cc";
    invocation.output = "output";
    invocation.target = "native";
    invocation.host_toolchain = "c++";
    invocation.device_toolchain = "nvcc";
    invocation.source_kind = source_kind_v1::cellerator_cxx;
    if (mode == "-E") invocation.output_mode = output_mode_v1::preprocess;
    else if (mode == "-fsyntax-only") invocation.output_mode = output_mode_v1::syntax_only;
    else if (mode == "-S") invocation.output_mode = output_mode_v1::assembly;
    else if (mode == "-c") invocation.output_mode = output_mode_v1::object;
    else if (mode == "--emit-ceir") invocation.output_mode = output_mode_v1::ceir;
    else if (mode == "--inspect-profile") invocation.output_mode = output_mode_v1::profile_inspection;
    else if (mode == "link") invocation.output_mode = output_mode_v1::link;
    else {
        std::cerr << "cellerator: incompatible-options: unknown mode\n";
        return 2;
    }
    const auto result = define_action_graph_v1(invocation);
    if (!result) {
        std::cerr << "cellerator: " << diagnostic_name_v1(result.diagnostic) << '\n';
        return 1;
    }
    for (std::size_t index = 0; index != result.graph.job_count; ++index) {
        std::cout << index << ':' << action_name_v1(result.graph.jobs[index].kind)
                  << (result.graph.jobs[index].semantic_stage ? ":semantic" : ":backend")
                  << '\n';
    }
}
