#include <Cellerator/compiler/ast/define_source_level_ast_node_families_v1.hh>

#include <algorithm>
#include <array>
#include <utility>

namespace Cellerator::compiler::ast {
namespace {

void set_error(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
}

struct concept_mapping_v1 {
    std::string_view spelling;
    ast_semantic_family_v1 family;
};

constexpr std::array concept_mappings{
    concept_mapping_v1{"declaration", ast_semantic_family_v1::declaration},
    concept_mapping_v1{"domain", ast_semantic_family_v1::declaration},
    concept_mapping_v1{"axis", ast_semantic_family_v1::declaration},
    concept_mapping_v1{"relation", ast_semantic_family_v1::declaration},
    concept_mapping_v1{"state", ast_semantic_family_v1::declaration},
    concept_mapping_v1{"support", ast_semantic_family_v1::declaration},
    concept_mapping_v1{"segment", ast_semantic_family_v1::declaration},
    concept_mapping_v1{"grouping", ast_semantic_family_v1::declaration},
    concept_mapping_v1{"field", ast_semantic_family_v1::execution_field},
    concept_mapping_v1{"execution_field", ast_semantic_family_v1::execution_field},
    concept_mapping_v1{"operation", ast_semantic_family_v1::operation},
    concept_mapping_v1{"relation_transfer", ast_semantic_family_v1::operation},
    concept_mapping_v1{"contraction", ast_semantic_family_v1::operation},
    concept_mapping_v1{"reduction", ast_semantic_family_v1::operation},
    concept_mapping_v1{"map", ast_semantic_family_v1::operation},
    concept_mapping_v1{"gate", ast_semantic_family_v1::operation},
    concept_mapping_v1{"update", ast_semantic_family_v1::operation},
    concept_mapping_v1{"hierarchy", ast_semantic_family_v1::operation},
    concept_mapping_v1{"moment", ast_semantic_family_v1::operation},
    concept_mapping_v1{"composition", ast_semantic_family_v1::operation},
    concept_mapping_v1{"given", ast_semantic_family_v1::policy_directive},
    concept_mapping_v1{"prefer", ast_semantic_family_v1::policy_directive},
    concept_mapping_v1{"require", ast_semantic_family_v1::policy_directive},
    concept_mapping_v1{"offer", ast_semantic_family_v1::policy_directive},
    concept_mapping_v1{"force", ast_semantic_family_v1::policy_directive},
    concept_mapping_v1{"inspect", ast_semantic_family_v1::policy_directive},
    concept_mapping_v1{"policy_directive", ast_semantic_family_v1::policy_directive},
    concept_mapping_v1{"effects", ast_semantic_family_v1::effect_contract},
    concept_mapping_v1{"effect_contract", ast_semantic_family_v1::effect_contract},
    concept_mapping_v1{"profile", ast_semantic_family_v1::profile_binding},
    concept_mapping_v1{"profile_binding", ast_semantic_family_v1::profile_binding},
    concept_mapping_v1{"inline_ir", ast_semantic_family_v1::inline_ir},
    concept_mapping_v1{"reflection", ast_semantic_family_v1::reflection},
    concept_mapping_v1{"pass", ast_semantic_family_v1::compiler_pass},
    concept_mapping_v1{"compiler_pass", ast_semantic_family_v1::compiler_pass},
    concept_mapping_v1{"native", ast_semantic_family_v1::native_fragment},
    concept_mapping_v1{"native_fragment", ast_semantic_family_v1::native_fragment},
};

} // namespace

ast_arena_id_v1 ast_semantic_table_v1::arena_id() const noexcept {
    return arena_id_;
}

std::size_t ast_semantic_table_v1::size() const noexcept {
    return records_.size();
}

const ast_semantic_node_v1* ast_semantic_table_v1::find(ast_node_handle_v1 node) const noexcept {
    if (node.arena != arena_id_ || node.slot >= records_.size()) {
        return nullptr;
    }
    return &records_[node.slot];
}

const std::vector<ast_semantic_node_v1>& ast_semantic_table_v1::records() const noexcept {
    return records_;
}

const std::vector<ast_family_contract_v1>& ast_family_contracts_v1() {
    static const std::vector<ast_family_contract_v1> contracts{
        {ast_semantic_family_v1::declaration, "declaration", ast_node_class_v1::declaration, false, false},
        {ast_semantic_family_v1::execution_field, "execution_field", ast_node_class_v1::expression, true, false},
        {ast_semantic_family_v1::operation, "operation", ast_node_class_v1::expression, true, false},
        {ast_semantic_family_v1::policy_directive, "policy_directive", ast_node_class_v1::policy, true, false},
        {ast_semantic_family_v1::effect_contract, "effect_contract", ast_node_class_v1::policy, true, true},
        {ast_semantic_family_v1::profile_binding, "profile_binding", ast_node_class_v1::policy, true, false},
        {ast_semantic_family_v1::inline_ir, "inline_ir", ast_node_class_v1::expression, true, false},
        {ast_semantic_family_v1::reflection, "reflection", ast_node_class_v1::expression, false, false},
        {ast_semantic_family_v1::compiler_pass, "compiler_pass", ast_node_class_v1::declaration, true, false},
        {ast_semantic_family_v1::native_fragment, "native_fragment", ast_node_class_v1::native_fragment, false, true},
    };
    return contracts;
}

std::optional<ast_semantic_family_v1>
classify_semantic_concept_v1(std::string_view concept_name) noexcept {
    const auto found = std::find_if(concept_mappings.begin(), concept_mappings.end(),
                                    [concept_name](const auto& item) {
                                        return item.spelling == concept_name;
                                    });
    if (found == concept_mappings.end()) {
        return std::nullopt;
    }
    return found->family;
}

bool validate_ast_family_contracts_v1(std::string* error) {
    const auto& contracts = ast_family_contracts_v1();
    constexpr auto expected = static_cast<std::size_t>(ast_semantic_family_v1::native_fragment);
    if (contracts.size() != expected) {
        set_error(error, "semantic family contract count is incomplete");
        return false;
    }
    std::array<bool, expected + 1> seen{};
    for (const auto& contract : contracts) {
        const auto index = static_cast<std::size_t>(contract.family);
        if (index == 0 || index > expected || seen[index] || contract.stable_name.empty() ||
            contract.storage_class == ast_node_class_v1::unknown) {
            set_error(error, "semantic family contract is invalid or duplicated");
            return false;
        }
        seen[index] = true;
    }
    return true;
}

std::optional<ast_semantic_table_v1>
freeze_semantic_nodes_v1(const ast_snapshot_v1& syntax,
                         std::vector<ast_semantic_node_v1> records,
                         std::string* error) {
    if (syntax.arena_id() == 0 || records.size() != syntax.node_count()) {
        set_error(error, "every syntax node must have exactly one semantic family record");
        return std::nullopt;
    }
    std::sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
        return left.node.slot < right.node.slot;
    });
    for (std::size_t index = 0; index < records.size(); ++index) {
        const auto& record = records[index];
        const auto family_index = static_cast<std::size_t>(record.family);
        if (record.node.arena != syntax.arena_id() || record.node.slot != index ||
            record.family == ast_semantic_family_v1::invalid ||
            family_index > ast_family_contracts_v1().size() ||
            !classify_semantic_concept_v1(
                ast_family_contracts_v1()[family_index - 1].stable_name)) {
            set_error(error, "semantic records contain a foreign, missing, or invalid family binding");
            return std::nullopt;
        }
    }
    ast_semantic_table_v1 table;
    table.arena_id_ = syntax.arena_id();
    table.records_ = std::move(records);
    return table;
}

} // namespace Cellerator::compiler::ast
