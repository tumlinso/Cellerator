#include <Cellerator/compiler/frontend/cxx/create_the_c_compilation_invocation_bridge_v1.hh>

#include <array>
#include <cstdio>
#include <iostream>
#include <string>

namespace cxx = Cellerator::compiler::frontend::cxx;

namespace {

std::string capture(const char* command) {
    std::array<char, 4096> buffer{};
    std::string output;
    FILE* pipe = popen(command, "r");
    if (pipe == nullptr) {
        return {};
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output.append(buffer.data());
    }
    if (pclose(pipe) != 0) {
        return {};
    }
    return output;
}

bool contains(const std::string& text, const std::string& expected, const char* label) {
    if (text.find(expected) != std::string::npos) {
        return true;
    }
    std::cerr << "missing " << label << ": " << expected << '\n';
    return false;
}

}  // namespace

int main() {
    cxx::cxx_compilation_invocation_request_v1 request;
    request.llvm_major = 18;
    request.language = cxx::cxx_language_mode_v1::cxx20;
    request.clang_driver_path = "/usr/bin/clang++-18";
    request.target_triple = "x86_64-pc-linux-gnu";
    request.sysroot = "/";
    request.quote_include_paths = {"/usr/local/include"};
    request.system_include_paths = {"/usr/include"};
    request.macro_definitions = {"CELLERATOR_BRIDGE_PROBE=7"};
    request.normalized_driver_arguments = {"-fsyntax-only"};

    cxx::cxx_compilation_invocation_v1 invocation;
    const auto status = cxx::create_cxx_compilation_invocation_v1(request, &invocation);
    if (status != cxx::cxx_compilation_invocation_status_v1::success ||
        invocation.native_compiler_invocation() == nullptr ||
        invocation.target_triple() != request.target_triple ||
        invocation.sysroot() != request.sysroot ||
        invocation.language() != request.language) {
        std::cerr << "failed to construct normalized CompilerInvocation\n";
        return 1;
    }

    bool valid = true;
    std::string serialized;
    for (const auto& argument : invocation.clang_arguments()) {
        serialized += argument + '\n';
    }
    valid = contains(serialized, "-std=c++20", "language mode") && valid;
    valid = contains(serialized, "--sysroot=/", "sysroot") && valid;
    valid = contains(serialized, "-iquote\n/usr/local/include", "quote include") && valid;
    valid = contains(serialized, "-isystem\n/usr/include", "system include") && valid;
    valid = contains(serialized, "-DCELLERATOR_BRIDGE_PROBE=7", "macro") && valid;

    const auto macros = capture(
        "printf '' | /usr/bin/clang++-18 -x c++ -std=c++20 "
        "-target x86_64-pc-linux-gnu --sysroot=/ "
        "-DCELLERATOR_BRIDGE_PROBE=7 -dM -E - 2>/dev/null");
    valid = contains(macros, "#define __cplusplus 202002L", "direct Clang language macro") && valid;
    valid = contains(macros, "#define CELLERATOR_BRIDGE_PROBE 7", "direct Clang user macro") && valid;

    const auto includes = capture(
        "printf '' | /usr/bin/clang++-18 -x c++ -std=c++20 "
        "-target x86_64-pc-linux-gnu --sysroot=/ "
        "-iquote/usr/local/include -isystem/usr/include -E -v - 2>&1");
    valid = contains(includes, "/usr/local/include", "direct Clang quote include") && valid;
    valid = contains(includes, "/usr/include", "direct Clang system include") && valid;

    auto invalid = request;
    invalid.normalized_driver_arguments = {"-cc1"};
    valid = cxx::create_cxx_compilation_invocation_v1(invalid, &invocation) ==
            cxx::cxx_compilation_invocation_status_v1::invalid_argument && valid;
    invalid = request;
    invalid.target_triple.clear();
    valid = cxx::create_cxx_compilation_invocation_v1(invalid, &invocation) ==
            cxx::cxx_compilation_invocation_status_v1::missing_target && valid;

    return valid ? 0 : 1;
}
