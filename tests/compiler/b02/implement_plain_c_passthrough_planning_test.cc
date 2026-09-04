#include <Cellerator/compiler/driver/implement_plain_c_passthrough_planning_v1.hh>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace cellerator::compiler::driver;
int main() {
    const std::vector<std::string> args{"-std=c++17", "-MMD", "-MF", "probe.d", "probe.cc", "-o", "probe"};
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i != 10000; ++i) { const auto plan = plan_plain_cxx_passthrough_v1("g++", args, false); if (plan.compiler != "g++" || plan.arguments != args || plan.semantic_job_count != 0 || plan.persistent_bytes != 0 || !plan.preserve_exit_code) return EXIT_FAILURE; }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin);
    if (elapsed.count() > 250) return EXIT_FAILURE;
    const auto root = std::filesystem::temp_directory_path() / "ce_ccp1_b02_009";
    std::filesystem::create_directories(root);
    const auto source = root / "probe.cc";
    { std::ofstream out(source); out << "#include <iostream>\nint exported_probe(){return 9;} int main(){std::cout << exported_probe();}\n"; }
    const auto direct = root / "direct"; const auto planned = root / "planned";
    const std::vector<std::string> compile_args{"-std=c++17", source.string(), "-MMD", "-MF", (root/"planned.d").string(), "-o", planned.string()};
    const auto plan = plan_plain_cxx_passthrough_v1("g++", compile_args, false);
    std::ostringstream command; command << plan.compiler; for (const auto& arg : plan.arguments) command << " '" << arg << "'";
    if (std::system(("g++ -std=c++17 '" + source.string() + "' -MMD -MF '" + (root/"direct.d").string() + "' -o '" + direct.string() + "'").c_str()) != 0 || std::system(command.str().c_str()) != 0) return EXIT_FAILURE;
    if (std::system(("nm -g '" + direct.string() + "' | grep -q exported_probe && nm -g '" + planned.string() + "' | grep -q exported_probe").c_str()) != 0 || std::system(("test \"$('" + direct.string() + "')\" = \"$('" + planned.string() + "')\"").c_str()) != 0 || !std::filesystem::exists(root/"planned.d")) return EXIT_FAILURE;
    std::filesystem::remove_all(root);
    std::cout << "validated symbols, depfiles, output and zero-semantic passthrough; 10000 plans in " << elapsed.count() << " ms, zero persistent bytes\n";
}
