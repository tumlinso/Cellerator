#include <Cellerator/compiler/driver/implement_response_file_and_argv_normalization_contracts_v1.hh>
#include <iostream>
int main(int argc, char** argv) {
    using namespace cellerator::compiler::driver;
    std::vector<std::string> args(argv + 1, argv + argc);
    const auto result = normalize_argv_v1(args, ".", [](std::string_view) { return std::optional<std::string>{}; });
    if (!result) { std::cerr << result.diagnostic << '\n'; return 1; }
    for (const auto& arg : result.arguments) std::cout << quote_normalized_argument_v1(arg) << '\n';
}
