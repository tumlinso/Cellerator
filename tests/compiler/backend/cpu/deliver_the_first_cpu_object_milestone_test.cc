#include <Cellerator/compiler/backend/deliver_the_first_cpu_object_milestone_v1.hh>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace cb = cellerator::compiler::backend::v1;

int main() {
    const auto directory = std::filesystem::temp_directory_path()
        / "ce_ccp1_f02_013";
    std::filesystem::create_directories(directory);
    const auto cell = directory / "program.cell";
    { std::ofstream out(cell); out
        << "profile cpu_reference;\nrelation response = apply weights to state;\n"; }
    cb::first_cpu_object_receipt_v1 receipt{};
    assert(cb::deliver_first_cpu_object_v1({cell.string(), "cpu_reference",
               "profile_relation", "g++", directory.string(),
               {1, 2, 3, 4, 5}, {0, 2, 4}, {0, 2, 1, 2}, {2, 4, 3, 5}},
        &receipt) == cb::compile_object_status_v1::success);
    assert(receipt.profile_bound && receipt.source_to_native_provenance);
    assert(receipt.compilation.format == cb::ordinary_object_format_v1::elf);

    const auto main = directory / "main.cc";
    const auto executable = directory / "program";
    { std::ofstream out(main); out
        << "extern \"C\" void profile_relation(const float*,float*);\n"
           "int main(){float x[]={10,20,30}, y[2]={}; profile_relation(x,y);"
           "return y[0]==140 && y[1]==210 ? 0 : 1;}\n"; }
    const std::string link = "g++ " + main.string() + " " + receipt.object_path
        + " -o " + executable.string();
    assert(std::system(link.c_str()) == 0);
    assert(std::system(executable.c_str()) == 0);
    std::filesystem::remove_all(directory);
}
