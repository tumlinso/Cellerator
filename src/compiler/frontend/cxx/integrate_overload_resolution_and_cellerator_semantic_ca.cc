#include <Cellerator/compiler/frontend/cxx/integrate_overload_resolution_and_cellerator_semantic_ca_v1.hh>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Expr.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Lex/Lexer.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {
namespace {

struct resolved_call {
    clang::CallExpr* expression = nullptr;
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
};

class call_visitor : public clang::RecursiveASTVisitor<call_visitor> {
public:
    call_visitor(clang::ASTContext& context, std::uint32_t offset)
        : context_(context), offset_(offset) {}

    bool VisitCallExpr(clang::CallExpr* expression) {
        const auto& sources = context_.getSourceManager();
        const auto begin_location = sources.getSpellingLoc(expression->getBeginLoc());
        const auto end_location = clang::Lexer::getLocForEndOfToken(
            sources.getSpellingLoc(expression->getEndLoc()), 0,
            sources, context_.getLangOpts());
        if (begin_location.isInvalid() || end_location.isInvalid()) {
            return true;
        }
        const auto begin = sources.getFileOffset(begin_location);
        const auto end = sources.getFileOffset(end_location);
        if (begin <= offset_ && offset_ < end) {
            calls.push_back({expression, begin, end});
        }
        return true;
    }

    std::vector<resolved_call> calls;

private:
    clang::ASTContext& context_;
    std::uint32_t offset_;
};

std::pair<std::string, std::string> biological_annotation(
    const clang::FunctionDecl& declaration) {
    for (const auto* attribute : declaration.specific_attrs<clang::AnnotateAttr>()) {
        const std::string annotation = attribute->getAnnotation().str();
        constexpr const char* prefix = "cellerator.operation:";
        if (annotation.rfind(prefix, 0) != 0) {
            continue;
        }
        const auto separator = annotation.find(";domain:");
        if (separator == std::string::npos) {
            return {annotation.substr(std::char_traits<char>::length(prefix)), {}};
        }
        return {
            annotation.substr(
                std::char_traits<char>::length(prefix),
                separator - std::char_traits<char>::length(prefix)),
            annotation.substr(separator + std::char_traits<char>::length(";domain:")),
        };
    }
    return {};
}

}  // namespace

overload_semantic_candidate_status_v1 resolve_overloads_and_semantic_candidates_v1(
    std::uint32_t schema_version,
    const upstream_clang_adapter_v1& adapter,
    const std::vector<overload_semantic_request_v1>& requests,
    overload_semantic_result_v1* result) noexcept {
    if (result == nullptr || requests.empty()) {
        return overload_semantic_candidate_status_v1::missing_resolved_call;
    }
    result->candidates.clear();
    result->statuses.clear();
    if (schema_version != overload_semantic_candidate_schema_version_v1) {
        return overload_semantic_candidate_status_v1::schema_mismatch;
    }
    if (validate_upstream_clang_adapter_v1(adapter) !=
        upstream_clang_adapter_status_v1::success) {
        return overload_semantic_candidate_status_v1::invalid_adapter;
    }

    try {
        auto* ast_unit = static_cast<clang::ASTUnit*>(const_cast<void*>(adapter.tooling.address));
        auto& context = ast_unit->getASTContext();
        auto overall = overload_semantic_candidate_status_v1::success;
        for (const auto& request : requests) {
            call_visitor visitor(context, request.source_offset);
            visitor.TraverseDecl(context.getTranslationUnitDecl());
            if (visitor.calls.empty()) {
                result->statuses.push_back(
                    overload_semantic_candidate_status_v1::missing_resolved_call);
                overall = overload_semantic_candidate_status_v1::missing_resolved_call;
                continue;
            }
            const auto narrowest = std::min_element(
                visitor.calls.begin(), visitor.calls.end(),
                [](const resolved_call& lhs, const resolved_call& rhs) {
                    return lhs.end - lhs.begin < rhs.end - rhs.begin;
                });
            const auto width = narrowest->end - narrowest->begin;
            if (std::count_if(
                    visitor.calls.begin(), visitor.calls.end(),
                    [width](const resolved_call& item) { return item.end - item.begin == width; }) != 1) {
                result->statuses.push_back(
                    overload_semantic_candidate_status_v1::ambiguous_source_location);
                overall = overload_semantic_candidate_status_v1::ambiguous_source_location;
                continue;
            }
            const auto* declaration = narrowest->expression->getDirectCallee();
            if (declaration == nullptr) {
                result->statuses.push_back(
                    overload_semantic_candidate_status_v1::missing_resolved_call);
                overall = overload_semantic_candidate_status_v1::missing_resolved_call;
                continue;
            }
            const auto annotation = biological_annotation(*declaration);
            overload_semantic_candidate_v1 candidate;
            candidate.selected_declaration = declaration;
            candidate.qualified_name = declaration->getQualifiedNameAsString();
            candidate.function_type = declaration->getType().getCanonicalType().getAsString();
            candidate.cellerator_aware = !annotation.first.empty();
            candidate.declared_operation = annotation.first;
            candidate.declared_domain_type = annotation.second;
            candidate.mechanism = candidate.cellerator_aware
                ? "cellerator-semantic-operation" : "ordinary-cxx-call";
            auto status = overload_semantic_candidate_status_v1::success;
            if (candidate.cellerator_aware && !request.required_operation.empty() &&
                request.required_operation != candidate.declared_operation) {
                status = overload_semantic_candidate_status_v1::biological_operation_mismatch;
            } else if (candidate.cellerator_aware && !request.required_domain_type.empty() &&
                       request.required_domain_type != candidate.declared_domain_type) {
                status = overload_semantic_candidate_status_v1::biological_domain_mismatch;
            }
            candidate.biologically_compatible = status == overload_semantic_candidate_status_v1::success;
            result->candidates.push_back(std::move(candidate));
            result->statuses.push_back(status);
            if (status != overload_semantic_candidate_status_v1::success) {
                overall = status;
            }
        }
        return overall;
    } catch (...) {
        return overload_semantic_candidate_status_v1::invalid_adapter;
    }
}

}  // namespace Cellerator::compiler::frontend::cxx
