#include <Cellerator/compiler/tooling/implement_compile_command_and_project_configuration_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::tooling;

int main() {
    const std::vector<compile_command_input_v1> ninja_commands = {
        {"/tmp/cmake-ninja", "src/model.cc",
         "clang++ -std=c++17 -fcellerator --cellerator-profile=pbmc "
         "--cellerator-backend cuda -resource-dir /clang/18 @model.rsp -c src/model.cc", {}},
        {"/tmp/cmake-ninja", "src/plain.cc", "",
         {"clang++", "-std=c++17", "-c", "src/plain.cc"}},
    };
    const auto responses = [](std::string_view path, std::string_view directory)
        -> std::optional<std::string> {
        if (path == "model.rsp") {
            assert(directory == "/tmp/cmake-ninja");
            return "--cellerator-toolchain=clang18 -DNAME='cell model'";
        }
        return std::nullopt;
    };

    project_configuration_v1 project;
    assert(project.load(ninja_commands, responses));
    assert(project.commands().size() == 2);
    const auto *model = project.command_for("/tmp/cmake-ninja/src/model.cc");
    assert(model != nullptr);
    assert(model->cellerator_active);
    assert(model->profile == "pbmc");
    assert(model->backend == "cuda");
    assert(model->toolchain == "clang18");
    assert(model->resource_directory == "/clang/18");
    assert(model->arguments[model->arguments.size() - 3] == "-DNAME=cell model");
    const auto *plain = project.command_for("/tmp/cmake-ninja/src/plain.cc");
    assert(plain != nullptr && !plain->cellerator_active);

    assert(!resolve_compile_command_v1({"/tmp", "broken.cc", "clang++ @missing", {}}, responses));
    assert(!project.load({ninja_commands.front(), ninja_commands.front()}, responses));
}
