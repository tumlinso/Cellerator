#include <Cellerator/compiler/frontend/cxx/bind_source_captures_to_c_declarations_and_expressions_v1.hh>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Lex/Lexer.h>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace Cellerator::compiler::frontend::cxx {
namespace {

struct candidate {
    const void* node = nullptr;
    source_capture_ast_kind_v1 ast_kind = source_capture_ast_kind_v1::declaration;
    std::string spelling;
    std::string type;
    source_provenance_v1 provenance;
};

source_provenance_v1 provenance(
    const clang::SourceManager& source_manager,
    const clang::LangOptions& language,
    clang::SourceRange range) {
    source_provenance_v1 result;
    const auto begin = source_manager.getSpellingLoc(range.getBegin());
    const auto end = clang::Lexer::getLocForEndOfToken(
        source_manager.getSpellingLoc(range.getEnd()), 0, source_manager, language);
    if (begin.isInvalid() || end.isInvalid()) {
        return result;
    }
    result.file = source_manager.getFilename(begin).str();
    result.begin_offset = source_manager.getFileOffset(begin);
    result.end_offset = source_manager.getFileOffset(end);
    result.line = source_manager.getSpellingLineNumber(begin);
    result.column = source_manager.getSpellingColumnNumber(begin);
    return result;
}

class capture_visitor : public clang::RecursiveASTVisitor<capture_visitor> {
public:
    capture_visitor(clang::ASTContext& context, const source_capture_request_v1& request)
        : context_(context), request_(request) {}

    bool VisitNamedDecl(clang::NamedDecl* declaration) {
        if (!is_declaration_capture() || declaration->isImplicit()) {
            return true;
        }
        const auto name = declaration->getQualifiedNameAsString();
        if (request_.spelling != name && request_.spelling != declaration->getNameAsString()) {
            return true;
        }
        auto item = make_candidate(declaration, declaration->getSourceRange());
        item.spelling = name;
        if (const auto* value = llvm::dyn_cast<clang::ValueDecl>(declaration)) {
            item.type = value->getType().getAsString();
        } else {
            item.type = declaration->getDeclKindName();
        }
        add_if_offset_matches(std::move(item));
        return true;
    }

    bool VisitExpr(clang::Expr* expression) {
        if (is_declaration_capture() || expression->isImplicitCXXThis()) {
            return true;
        }
        if (request_.kind == source_capture_kind_v1::native_call &&
            !llvm::isa<clang::CallExpr>(expression)) {
            return true;
        }
        auto item = make_candidate(expression, expression->getSourceRange());
        item.ast_kind = source_capture_ast_kind_v1::expression;
        item.type = expression->getType().getAsString();
        if (const auto* call = llvm::dyn_cast<clang::CallExpr>(expression)) {
            if (const auto* callee = call->getDirectCallee()) {
                item.spelling = callee->getQualifiedNameAsString();
            }
        }
        if (!request_.spelling.empty() && item.spelling != request_.spelling &&
            item.spelling.substr(item.spelling.rfind(':') == std::string::npos
                                     ? 0 : item.spelling.rfind(':') + 1) != request_.spelling) {
            return true;
        }
        add_if_offset_matches(std::move(item));
        return true;
    }

    std::vector<candidate> candidates;

private:
    bool is_declaration_capture() const noexcept {
        return request_.kind == source_capture_kind_v1::domain ||
               request_.kind == source_capture_kind_v1::state ||
               request_.kind == source_capture_kind_v1::relation;
    }

    candidate make_candidate(const void* node, clang::SourceRange range) const {
        candidate result;
        result.node = node;
        result.provenance = provenance(
            context_.getSourceManager(), context_.getLangOpts(), range);
        return result;
    }

    void add_if_offset_matches(candidate item) {
        if (request_.source_offset == unspecified_source_offset_v1 ||
            (item.provenance.begin_offset <= request_.source_offset &&
             request_.source_offset < item.provenance.end_offset)) {
            candidates.push_back(std::move(item));
        }
    }

    clang::ASTContext& context_;
    const source_capture_request_v1& request_;
};

source_provenance_v1 request_provenance(
    clang::ASTContext& context, std::uint32_t offset) {
    if (offset == unspecified_source_offset_v1) {
        return {};
    }
    auto& source_manager = context.getSourceManager();
    const auto file = source_manager.getMainFileID();
    const auto location = source_manager.getLocForStartOfFile(file).getLocWithOffset(offset);
    return provenance(
        source_manager, context.getLangOpts(), clang::SourceRange(location, location));
}

}  // namespace

source_capture_binding_status_v1 bind_source_captures_v1(
    std::uint32_t schema_version,
    const upstream_clang_adapter_v1& adapter,
    const std::vector<source_capture_request_v1>& requests,
    source_capture_binding_result_v1* result) noexcept {
    if (result == nullptr || requests.empty()) {
        return source_capture_binding_status_v1::invalid_request;
    }
    result->captures.clear();
    result->diagnostics.clear();
    if (schema_version != source_capture_binding_schema_version_v1) {
        return source_capture_binding_status_v1::schema_mismatch;
    }
    if (validate_upstream_clang_adapter_v1(adapter) !=
        upstream_clang_adapter_status_v1::success) {
        return source_capture_binding_status_v1::invalid_adapter;
    }

    try {
        auto* ast_unit = static_cast<clang::ASTUnit*>(const_cast<void*>(adapter.tooling.address));
        auto& context = ast_unit->getASTContext();
        auto overall = source_capture_binding_status_v1::success;
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const auto& request = requests[index];
            capture_visitor visitor(context, request);
            visitor.TraverseDecl(context.getTranslationUnitDecl());
            if (request.source_offset != unspecified_source_offset_v1 &&
                visitor.candidates.size() > 1 &&
                request.kind != source_capture_kind_v1::native_call) {
                const auto best = std::min_element(
                    visitor.candidates.begin(), visitor.candidates.end(),
                    [](const candidate& lhs, const candidate& rhs) {
                        return lhs.provenance.end_offset - lhs.provenance.begin_offset <
                               rhs.provenance.end_offset - rhs.provenance.begin_offset;
                    });
                candidate selected = *best;
                visitor.candidates.clear();
                visitor.candidates.push_back(std::move(selected));
            }
            if (visitor.candidates.size() != 1) {
                const auto code = visitor.candidates.empty()
                    ? source_capture_binding_status_v1::missing_capture
                    : source_capture_binding_status_v1::ambiguous_capture;
                result->diagnostics.push_back({
                    static_cast<std::uint32_t>(index),
                    code,
                    request_provenance(context, request.source_offset),
                    visitor.candidates.empty()
                        ? "capture '" + request.spelling + "' was not found"
                        : "capture '" + request.spelling + "' is ambiguous",
                });
                if (overall == source_capture_binding_status_v1::success ||
                    code == source_capture_binding_status_v1::ambiguous_capture) {
                    overall = code;
                }
                continue;
            }
            const auto& selected = visitor.candidates.front();
            result->captures.push_back({
                request.kind,
                selected.ast_kind,
                selected.node,
                selected.spelling.empty() ? request.spelling : selected.spelling,
                selected.type,
                selected.provenance,
            });
        }
        return overall;
    } catch (...) {
        return source_capture_binding_status_v1::invalid_adapter;
    }
}

}  // namespace Cellerator::compiler::frontend::cxx
