#include <Cellerator/compiler/tooling/implement_compile_command_and_project_configuration_v1.hh>

#include <iostream>

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    Cellerator::compiler::tooling::compile_command_input_v1 input;
    input.directory = argv[1];
    input.file = argv[2];
    for (int index = 3; index < argc; ++index) input.arguments.emplace_back(argv[index]);
    const auto resolved = Cellerator::compiler::tooling::resolve_compile_command_v1(input);
    if (!resolved) return 1;
    std::cout << resolved->file << '\n';
    return 0;
}
