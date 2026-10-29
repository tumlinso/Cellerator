#include <Cellerator/compiler/driver/discover_host_clang_toolchains_v1.hh>
#include <cstdlib>
#include <iostream>
#include <set>
using namespace cellerator::compiler::driver;
int main() {
    clang_discovery_input_v1 in{"/explicit", "/env", "/configured", {"/path"}, {"/default"}};
    const auto probe = [](const std::set<std::string>& files) { return [&files](std::string_view p) { return files.count(std::string(p)) != 0; }; };
    struct expected { std::set<std::string> files; discovery_source_v1 source; const char* root; } cases[] = {
        {{"/explicit/bin/clang++", "/env/bin/clang++"}, discovery_source_v1::explicit_override, "/explicit"},
        {{"/env/bin/clang++", "/configured/bin/clang++"}, discovery_source_v1::environment, "/env"},
        {{"/configured/bin/clang++", "/path/bin/clang++"}, discovery_source_v1::configured_resource, "/configured"},
        {{"/path/bin/clang++", "/default/bin/clang++"}, discovery_source_v1::path, "/path"},
        {{"/default/bin/clang++"}, discovery_source_v1::platform_default, "/default"}};
    for (const auto& test : cases) { const auto out = discover_host_clang_v1(in, probe(test.files)); if (!out || out.source != test.source || out.compiler != std::string(test.root) + "/bin/clang++" || out.linker.empty() || out.resource_directory.empty() || out.target_triple.empty() || out.version_identity.empty()) return EXIT_FAILURE; }
    if (discover_host_clang_v1(in, probe({}))) return EXIT_FAILURE;
    std::cout << "validated every host Clang discovery precedence branch\n";
}
