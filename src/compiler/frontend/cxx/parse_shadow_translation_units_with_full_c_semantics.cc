#include <Cellerator/compiler/frontend/cxx/parse_shadow_translation_units_with_full_c_semantics_v1.hh>

#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Tooling/Tooling.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>

namespace Cellerator::compiler::frontend::cxx {
namespace {

std::vector<std::string> diagnostic_messages(
    clang::TextDiagnosticBuffer::const_iterator begin,
    clang::TextDiagnosticBuffer::const_iterator end) {
    std::vector<std::string> messages;
    messages.reserve(static_cast<std::size_t>(std::distance(begin, end)));
    for (auto iterator = begin; iterator != end; ++iterator) {
        messages.push_back(iterator->second);
    }
    return messages;
}

}  // namespace

struct shadow_translation_unit_v1::implementation {
    std::unique_ptr<clang::ASTUnit> ast;
    upstream_clang_adapter_v1 adapter_record{};
    std::vector<std::string> error_messages;
    std::vector<std::string> warning_messages;
    std::string filename;
};

shadow_translation_unit_v1::shadow_translation_unit_v1() noexcept = default;
shadow_translation_unit_v1::~shadow_translation_unit_v1() = default;
shadow_translation_unit_v1::shadow_translation_unit_v1(
    shadow_translation_unit_v1&&) noexcept = default;
shadow_translation_unit_v1& shadow_translation_unit_v1::operator=(
    shadow_translation_unit_v1&&) noexcept = default;

const upstream_clang_adapter_v1& shadow_translation_unit_v1::adapter() const noexcept {
    static const upstream_clang_adapter_v1 empty;
    return implementation_ == nullptr ? empty : implementation_->adapter_record;
}

const std::vector<std::string>& shadow_translation_unit_v1::errors() const noexcept {
    static const std::vector<std::string> empty;
    return implementation_ == nullptr ? empty : implementation_->error_messages;
}

const std::vector<std::string>& shadow_translation_unit_v1::warnings() const noexcept {
    static const std::vector<std::string> empty;
    return implementation_ == nullptr ? empty : implementation_->warning_messages;
}

std::string_view shadow_translation_unit_v1::virtual_filename() const noexcept {
    return implementation_ == nullptr ? std::string_view{} : implementation_->filename;
}

shadow_translation_unit_status_v1 parse_shadow_translation_unit_v1(
    const shadow_translation_unit_request_v1& request,
    shadow_translation_unit_v1* translation_unit) noexcept {
    if (translation_unit == nullptr) {
        return shadow_translation_unit_status_v1::null_output;
    }
    if (request.schema_version != shadow_translation_unit_schema_version_v1) {
        return shadow_translation_unit_status_v1::schema_mismatch;
    }
    if (request.llvm_major != 17 && request.llvm_major != 18) {
        return shadow_translation_unit_status_v1::unsupported_llvm_major;
    }
    if (request.invocation == nullptr ||
        request.invocation->native_compiler_invocation() == nullptr) {
        return shadow_translation_unit_status_v1::missing_invocation;
    }
    if (request.source.empty()) {
        return shadow_translation_unit_status_v1::empty_source;
    }

    try {
        auto result = std::make_unique<shadow_translation_unit_v1::implementation>();
        auto arguments = request.invocation->clang_arguments();
        if (!arguments.empty() && arguments.back() == "-") {
            arguments.pop_back();
        }
        arguments.erase(
            std::remove(arguments.begin(), arguments.end(), "-fsyntax-only"),
            arguments.end());
        clang::TextDiagnosticBuffer diagnostics;
        result->ast = clang::tooling::buildASTFromCodeWithArgs(
            request.source,
            arguments,
            request.virtual_filename,
            "cellerator-clang-bridge",
            std::make_shared<clang::PCHContainerOperations>(),
            clang::tooling::getClangStripDependencyFileAdjuster(),
            clang::tooling::FileContentMappings(),
            &diagnostics);
        result->error_messages = diagnostic_messages(diagnostics.err_begin(), diagnostics.err_end());
        result->warning_messages = diagnostic_messages(diagnostics.warn_begin(), diagnostics.warn_end());
        result->filename = request.virtual_filename;
        if (result->ast == nullptr) {
            translation_unit->implementation_ = std::move(result);
            return shadow_translation_unit_status_v1::clang_parse_failed;
        }

        upstream_clang_adapter_request_v1 adapter_request;
        adapter_request.llvm_major = request.llvm_major;
        adapter_request.ast_context = {
            &result->ast->getASTContext(), request.llvm_major,
            upstream_clang_object_kind_v1::ast_context, {}};
        adapter_request.sema = {
            &result->ast->getSema(), request.llvm_major,
            upstream_clang_object_kind_v1::sema, {}};
        adapter_request.preprocessor = {
            &result->ast->getPreprocessor(), request.llvm_major,
            upstream_clang_object_kind_v1::preprocessor, {}};
        adapter_request.diagnostics = {
            &result->ast->getDiagnostics(), request.llvm_major,
            upstream_clang_object_kind_v1::diagnostics, {}};
        adapter_request.tooling = {
            result->ast.get(), request.llvm_major,
            upstream_clang_object_kind_v1::tooling, {}};
        if (bind_upstream_clang_adapter_v1(
                adapter_request, &result->adapter_record) !=
            upstream_clang_adapter_status_v1::success) {
            return shadow_translation_unit_status_v1::clang_parse_failed;
        }
        const bool has_errors = !result->error_messages.empty() ||
                                result->ast->getDiagnostics().hasErrorOccurred();
        translation_unit->implementation_ = std::move(result);
        return has_errors ? shadow_translation_unit_status_v1::semantic_errors
                          : shadow_translation_unit_status_v1::success;
    } catch (...) {
        return shadow_translation_unit_status_v1::clang_parse_failed;
    }
}

}  // namespace Cellerator::compiler::frontend::cxx
