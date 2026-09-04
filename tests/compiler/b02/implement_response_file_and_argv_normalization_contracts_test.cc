#include <Cellerator/compiler/driver/implement_response_file_and_argv_normalization_contracts_v1.hh>
#include <cstdlib>
#include <iostream>
#include <map>
using namespace cellerator::compiler::driver;
int main() {
    const std::map<std::string, std::string> files{{"/work/outer.rsp", "-DNAME='cell state' @inner.rsp -Iinc/../include"}, {"/work/inner.rsp", "\"source file.cc\" -Xcompiler=-Wall"}};
    const auto normalized = normalize_argv_v1({"@outer.rsp", "-o", "out.o"}, "/work",
        [&](std::string_view path) -> std::optional<std::string> { auto it = files.find(std::string(path)); return it == files.end() ? std::nullopt : std::optional<std::string>(it->second); },
        {{"LC_ALL", "C"}, {"PATH", "/bin"}});
    const std::vector<std::string> expected{"-DNAME=cell state", "source file.cc", "-Xcompiler=-Wall", "-I/work/include", "-o", "out.o"};
    if (!normalized || normalized.arguments != expected || normalized.captured_environment.size() != 2) return EXIT_FAILURE;
    std::vector<std::string> round_trip;
    for (const auto& arg : expected) {
        const auto split = split_response_file_v1(quote_normalized_argument_v1(arg));
        if (split.size() != 1) return EXIT_FAILURE;
        round_trip.push_back(split.front());
    }
    if (round_trip != expected) return EXIT_FAILURE;
    const auto missing = normalize_argv_v1({"@absent"}, "/work", [](std::string_view) { return std::optional<std::string>{}; });
    if (missing || missing.diagnostic.find("/work/absent") == std::string::npos) return EXIT_FAILURE;
    std::cout << "validated nested response files and shell-independent normalized argv\n";
}
