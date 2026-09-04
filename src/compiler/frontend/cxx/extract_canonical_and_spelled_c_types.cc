#include <Cellerator/compiler/frontend/cxx/extract_canonical_and_spelled_c_types_v1.hh>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Type.h>
#include <clang/Frontend/ASTUnit.h>

#include <iomanip>
#include <sstream>

namespace Cellerator::compiler::frontend::cxx {
namespace {

clang::QualType capture_type(
    const bound_source_capture_v1& capture,
    clang::ASTContext& context) {
    if (capture.ast_kind == source_capture_ast_kind_v1::expression) {
        return static_cast<const clang::Expr*>(capture.ast_node)->getType();
    }
    const auto* declaration = static_cast<const clang::Decl*>(capture.ast_node);
    if (const auto* value = llvm::dyn_cast<clang::ValueDecl>(declaration)) {
        return value->getType();
    }
    if (const auto* type = llvm::dyn_cast<clang::TypeDecl>(declaration)) {
        return context.getTypeDeclType(type);
    }
    return {};
}

std::string identity_for(
    const std::string& canonical,
    std::uint64_t size,
    std::uint64_t alignment,
    std::uint32_t address_space) {
    std::uint64_t hash = 1469598103934665603ull;
    const std::string input = canonical + "|" + std::to_string(size) + "|" +
                              std::to_string(alignment) + "|" +
                              std::to_string(address_space);
    for (const unsigned char byte : input) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    std::ostringstream stream;
    stream << "cxx-type-v1:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return stream.str();
}

bool contains_name(const std::string& spelling, const char* name) {
    return spelling.find(name) != std::string::npos;
}

}  // namespace

cxx_type_extraction_status_v1 extract_cxx_types_v1(
    std::uint32_t schema_version,
    const upstream_clang_adapter_v1& adapter,
    const std::vector<bound_source_capture_v1>& captures,
    std::vector<cxx_type_record_v1>* records) noexcept {
    if (records == nullptr || captures.empty()) {
        return cxx_type_extraction_status_v1::invalid_capture;
    }
    records->clear();
    if (schema_version != cxx_type_extraction_schema_version_v1) {
        return cxx_type_extraction_status_v1::schema_mismatch;
    }
    if (validate_upstream_clang_adapter_v1(adapter) !=
        upstream_clang_adapter_status_v1::success) {
        return cxx_type_extraction_status_v1::invalid_adapter;
    }

    try {
        auto* ast_unit = static_cast<clang::ASTUnit*>(const_cast<void*>(adapter.tooling.address));
        auto& context = ast_unit->getASTContext();
        clang::PrintingPolicy policy(context.getLangOpts());
        policy.SuppressScope = false;
        for (const auto& capture : captures) {
            if (capture.ast_node == nullptr) {
                return cxx_type_extraction_status_v1::invalid_capture;
            }
            const auto type = capture_type(capture, context);
            if (type.isNull() || type->isIncompleteType()) {
                return cxx_type_extraction_status_v1::incomplete_type;
            }
            const auto canonical = type.getCanonicalType();
            const auto info = context.getTypeInfo(type);
            cxx_type_record_v1 record;
            record.user_spelling = type.getAsString(policy);
            record.canonical_spelling = canonical.getAsString(policy);
            record.size_bytes = info.Width / context.getCharWidth();
            record.alignment_bytes = info.Align / context.getCharWidth();
            if (type->isBuiltinType()) {
                record.traits |= cxx_type_builtin_v1;
            }
            if (contains_name(record.user_spelling, "__half") ||
                contains_name(record.canonical_spelling, "__half")) {
                record.traits |= cxx_type_half_v1;
            }
            if (contains_name(record.user_spelling, "bfloat16") ||
                contains_name(record.user_spelling, "bf16") ||
                contains_name(record.canonical_spelling, "bfloat16") ||
                contains_name(record.canonical_spelling, "bf16")) {
                record.traits |= cxx_type_bfloat16_v1;
            }
            if (type->isVectorType()) {
                record.traits |= cxx_type_vector_v1;
            }
            if (type->isPointerType()) {
                record.traits |= cxx_type_pointer_v1;
            }
            if (type->isLValueReferenceType()) {
                record.traits |= cxx_type_lvalue_reference_v1;
            }
            if (type->isRValueReferenceType()) {
                record.traits |= cxx_type_rvalue_reference_v1;
            }
            clang::QualType address_space_type = type;
            if (type->isPointerType()) {
                address_space_type = type->getPointeeType();
            }
            if (address_space_type.hasAddressSpace()) {
                record.traits |= cxx_type_address_space_v1;
                record.address_space = static_cast<std::uint32_t>(
                    address_space_type.getAddressSpace());
            }
            if (type->isRecordType() || type->isEnumeralType()) {
                record.traits |= cxx_type_user_defined_v1;
            }
            record.canonical_identity = identity_for(
                record.canonical_spelling,
                record.size_bytes,
                record.alignment_bytes,
                record.address_space);
            records->push_back(std::move(record));
        }
        return cxx_type_extraction_status_v1::success;
    } catch (...) {
        return cxx_type_extraction_status_v1::invalid_capture;
    }
}

}  // namespace Cellerator::compiler::frontend::cxx
