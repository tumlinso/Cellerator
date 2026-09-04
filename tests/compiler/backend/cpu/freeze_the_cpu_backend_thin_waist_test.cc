#include <Cellerator/compiler/backend/freeze_the_cpu_backend_thin_waist_v1.hh>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace cb = cellerator::compiler::backend::v1;

int main() {
    const auto& receipt = cb::freeze_cpu_backend_thin_waist_v1();
    assert(receipt.backend_abi_version == 1 && receipt.cpu_backend_version == 1);
    assert(receipt.ordinary_objects && receipt.generated_cpp
        && receipt.runtime_binding && receipt.source_diagnostics
        && receipt.deterministic_fallbacks);

    const auto directory = std::filesystem::temp_directory_path()
        / "ce_ccp1_f02_014_install";
    const auto installed = directory / "include" / "Cellerator";
    std::filesystem::create_directories(installed.parent_path());
    std::filesystem::copy(CELLERATOR_TEST_INCLUDE_ROOT "/Cellerator", installed,
        std::filesystem::copy_options::recursive
            | std::filesystem::copy_options::overwrite_existing);
    const auto provider = directory / "provider.cc";
    const auto consumer = directory / "consumer.cc";
    { std::ofstream out(provider); out
        << "#include <Cellerator/compiler/backend/backend_v1.hh>\n"
           "int provider_version(){return cellerator::compiler::backend::v1::backend_thin_waist_version_v1;}\n"; }
    { std::ofstream out(consumer); out
        << "#include <Cellerator/compiler/backend/cpu/cpu_backend_v1.hh>\n"
           "int provider_version(); int main(){return provider_version()==1 && "
           "cellerator::compiler::backend::cpu::v1::cpu_backend_contract_version_v1==1?0:1;}\n"; }
    const auto executable = directory / "fixture";
    const std::string compile = "g++ -std=c++17 -I" + (directory / "include").string()
        + " " + provider.string() + " " + consumer.string() + " -o "
        + executable.string();
    assert(std::system(compile.c_str()) == 0);
    assert(std::system(executable.c_str()) == 0);
    std::filesystem::remove_all(directory);
}
