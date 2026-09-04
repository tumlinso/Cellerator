#include <Cellerator/compiler/driver/driver_v1.hh>
#include <Cellerator/compiler/driver/toolchain_v1.hh>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
namespace fs = std::filesystem;
namespace { std::string q(const fs::path& p) { return "'" + p.string() + "'"; } }
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const fs::path build_driver = argv[1];
    const auto root = fs::temp_directory_path() / "ce_ccp1_b02_014";
    fs::remove_all(root); fs::create_directories(root / "install/bin");
    const auto installed_driver = root / "install/bin/cellerator";
    fs::copy_file(build_driver, installed_driver, fs::copy_options::overwrite_existing);
    fs::permissions(installed_driver, fs::perms::owner_exec | fs::perms::owner_read | fs::perms::owner_write);
    const auto source = root / "plain.cc";
    { std::ofstream out(source); out << "#include <iostream>\nint main(){std::cout << \"ordinary-cxx\";}\n"; }
    for (const auto& driver : {build_driver, installed_driver}) {
        for (const auto* compiler : {"g++", "clang++"}) {
            if (std::system(("command -v " + std::string(compiler) + " >/dev/null").c_str()) != 0) continue;
            const auto object = root / (std::string(compiler) + (driver == build_driver ? "-build.o" : "-install.o"));
            const auto executable = root / (std::string(compiler) + (driver == build_driver ? "-build" : "-install"));
            const std::string prefix = q(driver) + " --driver " + compiler;
            if (std::system((prefix + " -std=c++17 -c " + q(source) + " -o " + q(object)).c_str()) != 0 || !fs::exists(object) || fs::file_size(object) == 0) return EXIT_FAILURE;
            if (std::system((prefix + " " + q(object) + " -o " + q(executable)).c_str()) != 0 || !fs::exists(executable)) return EXIT_FAILURE;
            if (std::system(("test \"$(" + q(executable) + ")\" = ordinary-cxx").c_str()) != 0) return EXIT_FAILURE;
        }
    }
    fs::remove_all(root);
    std::cout << "validated build-tree and installed-layout GCC/Clang ordinary object/executable passthrough\n";
}
