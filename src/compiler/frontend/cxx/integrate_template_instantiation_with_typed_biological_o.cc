#include <Cellerator/compiler/frontend/cxx/integrate_template_instantiation_with_typed_biological_o_v1.hh>

#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Lex/Lexer.h>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {
namespace {

class template_visitor : public clang::RecursiveASTVisitor<template_visitor> {
public:
    explicit template_visitor(std::string name) : name_(std::move(name)) {}

    bool VisitFunctionTemplateDecl(clang::FunctionTemplateDecl* declaration) {
        if (declaration->getQualifiedNameAsString() == name_ ||
            declaration->getNameAsString() == name_) {
            declarations.push_back(declaration);
        }
        return true;
    }

    std::vector<clang::FunctionTemplateDecl*> declarations;

private:
    std::string name_;
};

std::string source_text(
    const clang::Expr* expression,
    const clang::ASTContext& context) {
    if (expression == nullptr) {
        return {};
    }
    return clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(expression->getSourceRange()),
        context.getSourceManager(),
        context.getLangOpts()).str();
}

std::string short_hash(const std::string& value) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash;
    return stream.str();
}

}  // namespace

biological_template_operation_status_v1 instantiate_biological_template_operations_v1(
    std::uint32_t schema_version,
    const upstream_clang_adapter_v1& adapter,
    const std::string& qualified_template_name,
    std::vector<biological_template_operation_v1>* operations) noexcept {
    if (operations == nullptr || qualified_template_name.empty()) {
        return biological_template_operation_status_v1::missing_template;
    }
    operations->clear();
    if (schema_version != biological_template_operation_schema_version_v1) {
        return biological_template_operation_status_v1::schema_mismatch;
    }
    if (validate_upstream_clang_adapter_v1(adapter) !=
        upstream_clang_adapter_status_v1::success) {
        return biological_template_operation_status_v1::invalid_adapter;
    }

    try {
        auto* ast_unit = static_cast<clang::ASTUnit*>(const_cast<void*>(adapter.tooling.address));
        auto& context = ast_unit->getASTContext();
        template_visitor visitor(qualified_template_name);
        visitor.TraverseDecl(context.getTranslationUnitDecl());
        if (visitor.declarations.empty()) {
            return biological_template_operation_status_v1::missing_template;
        }
        if (visitor.declarations.size() != 1) {
            return biological_template_operation_status_v1::ambiguous_template;
        }
        auto* function_template = visitor.declarations.front();
        const std::string constraint = source_text(
            function_template->getTemplatedDecl()->getTrailingRequiresClause(), context);
        clang::PrintingPolicy policy(context.getLangOpts());
        policy.SuppressScope = false;
        for (auto* specialization : function_template->specializations()) {
            if (specialization->getTemplateSpecializationKind() == clang::TSK_Undeclared ||
                specialization->getNumParams() < 2) {
                continue;
            }
            const auto numeric = specialization->getParamDecl(0)->getType();
            const auto domain = specialization->getParamDecl(1)->getType();
            const auto result = specialization->getReturnType();
            if (numeric->isDependentType() || domain->isDependentType() || result->isDependentType()) {
                return biological_template_operation_status_v1::unresolved_substitution;
            }
            biological_template_operation_v1 operation;
            operation.function_declaration = specialization;
            operation.template_name = function_template->getQualifiedNameAsString();
            operation.dependent_constraint = constraint;
            operation.numeric_user_spelling = numeric.getAsString(policy);
            operation.numeric_canonical_spelling = numeric.getCanonicalType().getAsString(policy);
            operation.domain_user_spelling = domain.getAsString(policy);
            operation.domain_canonical_spelling = domain.getCanonicalType().getAsString(policy);
            operation.result_canonical_spelling = result.getCanonicalType().getAsString(policy);
            const std::string signature = operation.template_name + "|" +
                operation.domain_canonical_spelling + "|" +
                operation.numeric_canonical_spelling + "|" +
                operation.result_canonical_spelling;
            operation.operation_identity = "cxx-biological-op-v1:" + short_hash(signature);
            operation.candidates = {
                {operation.operation_identity + ":native", "cellerator-native"},
                {operation.operation_identity + ":clang", "clang-aot-inline"},
            };
            operations->push_back(std::move(operation));
        }
        return operations->empty()
            ? biological_template_operation_status_v1::no_instantiations
            : biological_template_operation_status_v1::success;
    } catch (...) {
        return biological_template_operation_status_v1::invalid_adapter;
    }
}

}  // namespace Cellerator::compiler::frontend::cxx
