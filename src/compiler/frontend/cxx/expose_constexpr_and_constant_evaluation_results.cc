#include <Cellerator/compiler/frontend/cxx/expose_constexpr_and_constant_evaluation_results_v1.hh>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/ASTUnit.h>

#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {
namespace {

class constant_visitor : public clang::RecursiveASTVisitor<constant_visitor> {
public:
    explicit constant_visitor(std::string name) : name_(std::move(name)) {}

    bool VisitVarDecl(clang::VarDecl* declaration) {
        if ((declaration->getQualifiedNameAsString() == name_ ||
             declaration->getNameAsString() == name_) && declaration->hasInit()) {
            declarations.push_back(declaration);
        }
        return true;
    }

    std::vector<clang::VarDecl*> declarations;

private:
    std::string name_;
};

constexpr_import_status_v1 evaluate(
    clang::VarDecl& declaration,
    clang::ASTContext& context,
    constexpr_value_v1* output) {
    output->declaration = &declaration;
    output->qualified_name = declaration.getQualifiedNameAsString();
    output->canonical_type = declaration.getType().getCanonicalType().getAsString();
    const auto* initializer = declaration.getInit()->IgnoreParenImpCasts();
    if (const auto* string = llvm::dyn_cast<clang::StringLiteral>(initializer)) {
        output->kind = constexpr_value_kind_v1::string;
        output->string_value = string->getBytes().str();
        return constexpr_import_status_v1::success;
    }
    clang::Expr::EvalResult result;
    if (!initializer->EvaluateAsRValue(result, context) || result.HasSideEffects) {
        return constexpr_import_status_v1::not_constant;
    }
    const auto& value = result.Val;
    if (value.isInt()) {
        const auto integer = value.getInt();
        if (declaration.getType()->isBooleanType()) {
            output->kind = constexpr_value_kind_v1::boolean;
            output->boolean_value = integer.getBoolValue();
        } else if (declaration.getType()->isUnsignedIntegerType() ||
                   (declaration.getType()->isEnumeralType() &&
                    declaration.getType()->getAs<clang::EnumType>()->getDecl()
                        ->getIntegerType()->isUnsignedIntegerType())) {
            output->kind = constexpr_value_kind_v1::unsigned_integer;
            output->unsigned_value = integer.getZExtValue();
        } else {
            output->kind = constexpr_value_kind_v1::signed_integer;
            output->signed_value = integer.getSExtValue();
        }
        return constexpr_import_status_v1::success;
    }
    if (value.isFloat()) {
        output->kind = constexpr_value_kind_v1::floating_point;
        output->floating_value = value.getFloat().convertToDouble();
        return constexpr_import_status_v1::success;
    }
    return constexpr_import_status_v1::unsupported_value;
}

}  // namespace

constexpr_import_status_v1 import_constexpr_values_v1(
    std::uint32_t schema_version,
    const upstream_clang_adapter_v1& adapter,
    const std::vector<constexpr_import_request_v1>& requests,
    std::vector<constexpr_value_v1>* values) noexcept {
    if (values == nullptr || requests.empty()) {
        return constexpr_import_status_v1::missing_constant;
    }
    values->clear();
    if (schema_version != constexpr_import_schema_version_v1) {
        return constexpr_import_status_v1::schema_mismatch;
    }
    if (validate_upstream_clang_adapter_v1(adapter) !=
        upstream_clang_adapter_status_v1::success) {
        return constexpr_import_status_v1::invalid_adapter;
    }
    try {
        auto* ast_unit = static_cast<clang::ASTUnit*>(const_cast<void*>(adapter.tooling.address));
        auto& context = ast_unit->getASTContext();
        for (const auto& request : requests) {
            constant_visitor visitor(request.qualified_name);
            visitor.TraverseDecl(context.getTranslationUnitDecl());
            if (visitor.declarations.empty()) {
                return constexpr_import_status_v1::missing_constant;
            }
            if (visitor.declarations.size() != 1) {
                return constexpr_import_status_v1::ambiguous_constant;
            }
            constexpr_value_v1 value;
            const auto status = evaluate(*visitor.declarations.front(), context, &value);
            if (status != constexpr_import_status_v1::success) {
                return status;
            }
            values->push_back(std::move(value));
        }
        return constexpr_import_status_v1::success;
    } catch (...) {
        return constexpr_import_status_v1::unsupported_value;
    }
}

}  // namespace Cellerator::compiler::frontend::cxx
