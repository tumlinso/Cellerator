#include <Cellerator/compiler/backend/compile_generated_c_into_ordinary_objects_v1.hh>

#include <array>
#include <fstream>
#include <spawn.h>
#include <sys/wait.h>

extern char** environ;

namespace cellerator::compiler::backend::v1 {
namespace {

ordinary_object_format_v1 inspect_format(const std::string& path) {
    std::array<unsigned char, 4> magic{};
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(magic.data()), magic.size());
    if (input.gcount() != static_cast<std::streamsize>(magic.size()))
        return ordinary_object_format_v1::unknown;
    if (magic == std::array<unsigned char, 4>{0x7f, 'E', 'L', 'F'})
        return ordinary_object_format_v1::elf;
    const std::uint32_t word = static_cast<std::uint32_t>(magic[0])
        | static_cast<std::uint32_t>(magic[1]) << 8
        | static_cast<std::uint32_t>(magic[2]) << 16
        | static_cast<std::uint32_t>(magic[3]) << 24;
    if (word == 0xfeedfaceU || word == 0xfeedfacfU
        || word == 0xcefaedfeU || word == 0xcffaedfeU)
        return ordinary_object_format_v1::macho;
    return magic[2] == 0 && magic[3] == 0
        ? ordinary_object_format_v1::coff
        : ordinary_object_format_v1::unknown;
}

}  // namespace

compile_object_status_v1 compile_generated_cpp_object_v1(
    const compile_generated_cpp_request_v1& request,
    compile_generated_cpp_receipt_v1* receipt) noexcept {
    if (receipt == nullptr || request.compiler.empty()
        || request.source_path.empty() || request.object_path.empty()
        || request.depfile_path.empty())
        return compile_object_status_v1::invalid_argument;
    *receipt = {};
    receipt->compiler = request.compiler;
    receipt->source_path = request.source_path;
    receipt->object_path = request.object_path;
    receipt->depfile_path = request.depfile_path;
    receipt->support_libraries = request.support_libraries;
    receipt->arguments = {request.compiler, "-std=c++17", "-c",
        request.source_path, "-o", request.object_path, "-MMD", "-MF",
        request.depfile_path};
    if (!request.source_root.empty())
        receipt->arguments.push_back("-fdebug-prefix-map="
            + request.source_root + "=.");
    receipt->arguments.insert(receipt->arguments.end(),
        request.abi_flags.begin(), request.abi_flags.end());
    for (const auto& include : request.include_paths)
        receipt->arguments.push_back("-I" + include);
    // Support libraries are tracked in the receipt for the later link job;
    // compilation deliberately does not consume them.

    std::vector<char*> argv;
    argv.reserve(receipt->arguments.size() + 1);
    for (auto& argument : receipt->arguments)
        argv.push_back(argument.data());
    argv.push_back(nullptr);
    pid_t process = 0;
    const int spawn_status = posix_spawnp(&process, request.compiler.c_str(),
        nullptr, nullptr, argv.data(), environ);
    if (spawn_status != 0)
        return compile_object_status_v1::compiler_unavailable;
    int status = 0;
    if (waitpid(process, &status, 0) < 0 || !WIFEXITED(status))
        return compile_object_status_v1::compilation_failed;
    receipt->exit_code = WEXITSTATUS(status);
    if (receipt->exit_code != 0)
        return compile_object_status_v1::compilation_failed;
    receipt->format = inspect_format(request.object_path);
    return receipt->format == ordinary_object_format_v1::unknown
        ? compile_object_status_v1::object_unreadable
        : compile_object_status_v1::success;
}

}  // namespace cellerator::compiler::backend::v1
