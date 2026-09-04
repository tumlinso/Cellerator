#include <Cellerator/compiler/backend/compile_generated_c_into_ordinary_objects_v1.hh>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace cb = cellerator::compiler::backend::v1;

int main() {
    const auto directory = std::filesystem::temp_directory_path()
        / "ce_ccp1_f02_009";
    std::filesystem::create_directories(directory);
    const auto source = directory / "generated.cc";
    const auto caller = directory / "caller.cc";
    { std::ofstream out(source); out << "extern \"C\" int generated_value(){return 42;}\n"; }
    { std::ofstream out(caller); out << "extern \"C\" int generated_value();\n"
        "int main(){return generated_value()==42?0:1;}\n"; }

    for (const char* compiler : {"g++", "clang++-18"}) {
        const std::string suffix = std::string(compiler).find("clang") == 0
            ? "clang" : "gcc";
        const auto object = directory / (suffix + ".o");
        const auto depfile = directory / (suffix + ".d");
        cb::compile_generated_cpp_receipt_v1 receipt{};
        assert(cb::compile_generated_cpp_object_v1({compiler, source.string(),
                   object.string(), depfile.string(), directory.string(),
                   {"-fPIC"}, {}, {"Cellerator"}}, &receipt)
            == cb::compile_object_status_v1::success);
        assert(receipt.format == cb::ordinary_object_format_v1::elf);
        assert(std::filesystem::file_size(object) > 0);
        assert(std::filesystem::file_size(depfile) > 0);
        const std::string symbol = "nm " + object.string()
            + " | grep -q generated_value";
        assert(std::system(symbol.c_str()) == 0);
        const auto executable = directory / (suffix + ".exe");
        const std::string link = std::string(compiler) + " " + caller.string()
            + " " + object.string() + " -o " + executable.string();
        assert(std::system(link.c_str()) == 0);
        assert(std::system(executable.c_str()) == 0);
    }
    std::filesystem::remove_all(directory);
}
