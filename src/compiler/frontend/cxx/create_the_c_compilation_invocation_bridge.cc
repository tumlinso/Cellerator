#include <Cellerator/compiler/frontend/cxx/create_the_c_compilation_invocation_bridge_v1.hh>

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>

#include <algorithm>
#include <memory>
#include <utility>

namespace Cellerator::compiler::frontend::cxx {
namespace {

const char* language_standard(cxx_language_mode_v1 language) noexcept {
    switch (language) {
        case cxx_language_mode_v1::cxx17:
            return "-std=c++17";
        case cxx_language_mode_v1::cxx20:
            return "-std=c++20";
        case cxx_language_mode_v1::cxx23:
            return "-std=c++23";
    }
    return nullptr;
}

bool invalid_passthrough_argument(const std::string& argument) noexcept {
    return argument.empty() || argument == "-cc1" || argument == "-target" ||
           argument == "--target" || argument == "--sysroot" ||
           argument.rfind("--target=", 0) == 0 || argument.rfind("--sysroot=", 0) == 0 ||
           argument.rfind("-std=", 0) == 0 || argument == "-x";
}

void append_prefixed(std::vector<std::string>* arguments,
                     const std::vector<std::string>& values,
                     const char* prefix) {
    for (const auto& value : values) {
        arguments->emplace_back(std::string(prefix) + value);
    }
}

void append_paired(std::vector<std::string>* arguments,
                   const std::vector<std::string>& values,
                   const char* option) {
    for (const auto& value : values) {
        arguments->emplace_back(option);
        arguments->push_back(value);
    }
}

}  // namespace

struct cxx_compilation_invocation_v1::implementation {
    clang::CompilerInvocation invocation;
    std::vector<std::string> arguments;
    std::string target;
    std::string sysroot_path;
    cxx_language_mode_v1 language_mode = cxx_language_mode_v1::cxx20;
};

cxx_compilation_invocation_v1::cxx_compilation_invocation_v1() noexcept = default;
cxx_compilation_invocation_v1::~cxx_compilation_invocation_v1() = default;
cxx_compilation_invocation_v1::cxx_compilation_invocation_v1(
    cxx_compilation_invocation_v1&&) noexcept = default;
cxx_compilation_invocation_v1& cxx_compilation_invocation_v1::operator=(
    cxx_compilation_invocation_v1&&) noexcept = default;

const void* cxx_compilation_invocation_v1::native_compiler_invocation() const noexcept {
    return implementation_ == nullptr ? nullptr : &implementation_->invocation;
}

const std::vector<std::string>& cxx_compilation_invocation_v1::clang_arguments() const noexcept {
    static const std::vector<std::string> empty;
    return implementation_ == nullptr ? empty : implementation_->arguments;
}

std::string_view cxx_compilation_invocation_v1::target_triple() const noexcept {
    return implementation_ == nullptr ? std::string_view{} : implementation_->target;
}

std::string_view cxx_compilation_invocation_v1::sysroot() const noexcept {
    return implementation_ == nullptr ? std::string_view{} : implementation_->sysroot_path;
}

cxx_language_mode_v1 cxx_compilation_invocation_v1::language() const noexcept {
    return implementation_ == nullptr ? cxx_language_mode_v1::cxx20
                                      : implementation_->language_mode;
}

cxx_compilation_invocation_status_v1 create_cxx_compilation_invocation_v1(
    const cxx_compilation_invocation_request_v1& request,
    cxx_compilation_invocation_v1* invocation) noexcept {
    if (invocation == nullptr) {
        return cxx_compilation_invocation_status_v1::null_output;
    }
    if (request.schema_version != cxx_compilation_invocation_schema_version_v1) {
        return cxx_compilation_invocation_status_v1::schema_mismatch;
    }
    if (request.llvm_major != 17 && request.llvm_major != 18) {
        return cxx_compilation_invocation_status_v1::unsupported_llvm_major;
    }
    if (request.clang_driver_path.empty()) {
        return cxx_compilation_invocation_status_v1::missing_driver_path;
    }
    if (request.target_triple.empty()) {
        return cxx_compilation_invocation_status_v1::missing_target;
    }
    const char* standard = language_standard(request.language);
    if (standard == nullptr || std::any_of(
            request.normalized_driver_arguments.begin(),
            request.normalized_driver_arguments.end(),
            invalid_passthrough_argument)) {
        return cxx_compilation_invocation_status_v1::invalid_argument;
    }

    try {
        auto result = std::make_unique<cxx_compilation_invocation_v1::implementation>();
        result->arguments = {
            "-x", "c++", standard, "--target=" + request.target_triple};
        if (!request.sysroot.empty()) {
            result->arguments.emplace_back("--sysroot=" + request.sysroot);
        }
        append_paired(&result->arguments, request.quote_include_paths, "-iquote");
        append_paired(&result->arguments, request.system_include_paths, "-isystem");
        append_prefixed(&result->arguments, request.macro_definitions, "-D");
        if (!request.module_files.empty()) {
            result->arguments.emplace_back("-fmodules");
            append_prefixed(&result->arguments, request.module_files, "-fmodule-file=");
        }
        result->arguments.insert(
            result->arguments.end(),
            request.normalized_driver_arguments.begin(),
            request.normalized_driver_arguments.end());
        result->arguments.emplace_back("-");

        std::vector<std::string> cc1_arguments{
            "-x", "c++", standard, "-triple", request.target_triple};
        if (!request.sysroot.empty()) {
            cc1_arguments.emplace_back("-isysroot");
            cc1_arguments.push_back(request.sysroot);
        }
        append_paired(&cc1_arguments, request.quote_include_paths, "-iquote");
        append_paired(&cc1_arguments, request.system_include_paths, "-internal-isystem");
        append_prefixed(&cc1_arguments, request.macro_definitions, "-D");
        if (!request.module_files.empty()) {
            cc1_arguments.emplace_back("-fmodules");
            append_prefixed(&cc1_arguments, request.module_files, "-fmodule-file=");
        }
        cc1_arguments.insert(
            cc1_arguments.end(),
            request.normalized_driver_arguments.begin(),
            request.normalized_driver_arguments.end());
        cc1_arguments.emplace_back("-");

        std::vector<const char*> argument_views;
        argument_views.reserve(cc1_arguments.size());
        for (const auto& argument : cc1_arguments) {
            argument_views.push_back(argument.c_str());
        }
        auto diagnostic_ids = llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
            new clang::DiagnosticIDs());
        auto diagnostic_options = llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions>(
            new clang::DiagnosticOptions());
        clang::DiagnosticsEngine diagnostics(
            diagnostic_ids,
            diagnostic_options,
            new clang::IgnoringDiagConsumer(),
            true);
        if (!clang::CompilerInvocation::CreateFromArgs(
                result->invocation,
                argument_views,
                diagnostics,
                request.clang_driver_path.c_str())) {
            return cxx_compilation_invocation_status_v1::clang_rejected_arguments;
        }
        result->target = request.target_triple;
        result->sysroot_path = request.sysroot;
        result->language_mode = request.language;
        invocation->implementation_ = std::move(result);
        return cxx_compilation_invocation_status_v1::success;
    } catch (...) {
        return cxx_compilation_invocation_status_v1::clang_rejected_arguments;
    }
}

}  // namespace Cellerator::compiler::frontend::cxx
