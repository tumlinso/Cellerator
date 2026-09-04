#include <Cellerator/compiler/ir/semantic/deliver_source_to_semantic_ir_vertical_slice_v1.hh>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace Cellerator::compiler::ir::semantic {
namespace {

using operation_kind = cellerator::compute::operation::v2::operation_kind;
using stable_id = cellerator::compute::operation::v2::stable_id;

void set_status(semantic_vertical_slice_status_v1* status,
                semantic_vertical_slice_status_v1 value) noexcept {
    if (status != nullptr) *status = value;
}

std::string trim(std::string_view value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string_view::npos) return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(begin, end - begin + 1));
}

bool valid_name(std::string_view value) noexcept {
    if (value.empty() || !(std::isalpha(static_cast<unsigned char>(value.front())) ||
                           value.front() == '_')) return false;
    return std::all_of(value.begin() + 1, value.end(), [](unsigned char character) {
        return std::isalnum(character) || character == '_' || character == ':' ||
            character == '.';
    });
}

std::uint64_t stable_hash(std::string_view value,
                          std::uint64_t seed = 14695981039346656037ULL) noexcept {
    auto result = seed;
    for (const unsigned char character : value)
        result = (result ^ character) * 1099511628211ULL;
    return result;
}

stable_id operation_identity(operation_kind kind, std::string_view result,
                             const std::vector<std::string>& operands,
                             std::string_view destination) {
    std::ostringstream meaning;
    meaning << static_cast<unsigned>(kind) << ':' << result << '=';
    for (const auto& operand : operands) meaning << operand.size() << ':' << operand;
    meaning << '>' << destination;
    const auto text = meaning.str();
    return {stable_hash(text), stable_hash(text, 0xd6e8feb86659fd93ULL)};
}

std::vector<std::string> split_arguments(std::string_view arguments) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    for (std::size_t index = 0; index <= arguments.size(); ++index) {
        if (index == arguments.size() || arguments[index] == ',') {
            result.push_back(trim(arguments.substr(begin, index - begin)));
            begin = index + 1;
        }
    }
    return result;
}

bool parse_assignment(std::string_view line, std::string* result,
                      std::string_view* expression) {
    const auto equal = line.find('=');
    if (equal == std::string_view::npos) return false;
    *result = trim(line.substr(0, equal));
    *expression = line.substr(equal + 1);
    return valid_name(*result);
}

bool parse_call(std::string_view expression, std::string_view call,
                std::vector<std::string>* operands) {
    const auto call_position = expression.find(call);
    if (call_position == std::string_view::npos) return false;
    const auto open = expression.find('(', call_position + call.size());
    const auto close = expression.find(')', open == std::string_view::npos ? 0 : open + 1);
    if (open == std::string_view::npos || close == std::string_view::npos) return false;
    *operands = split_arguments(expression.substr(open + 1, close - open - 1));
    return !operands->empty() &&
        std::all_of(operands->begin(), operands->end(), valid_name);
}

bool validate_module(const source_linked_semantic_module_v1& module) noexcept {
    if (!valid_name(module.field) || !valid_name(module.profile) ||
        module.operations.empty()) return false;
    for (const auto& operation : module.operations) {
        if (!cellerator::compute::operation::v2::valid_stable_id(operation.identity) ||
            !cellerator::compute::operation::v2::valid_operation_kind(operation.kind) ||
            !valid_name(operation.result) || operation.operands.empty() ||
            !std::all_of(operation.operands.begin(), operation.operands.end(), valid_name) ||
            operation.source.line == 0 || operation.source.column == 0) return false;
        if (operation.kind == operation_kind::relation_apply &&
            (!valid_name(operation.destination_domain) || operation.operands.size() != 2))
            return false;
        if (operation.kind != operation_kind::relation_apply &&
            !operation.destination_domain.empty()) return false;
    }
    return true;
}

const char* kind_name(operation_kind kind) noexcept {
    switch (kind) {
        case operation_kind::relation_apply: return "relation_apply";
        case operation_kind::contract_on_support: return "contract_on_support";
        case operation_kind::segment_normalize: return "segment_normalize";
        default: return nullptr;
    }
}

std::optional<operation_kind> parse_kind(std::string_view value) noexcept {
    if (value == "relation_apply") return operation_kind::relation_apply;
    if (value == "contract_on_support") return operation_kind::contract_on_support;
    if (value == "segment_normalize") return operation_kind::segment_normalize;
    return std::nullopt;
}

std::vector<std::string> split_tab(std::string_view line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    for (std::size_t index = 0; index <= line.size(); ++index) {
        if (index == line.size() || line[index] == '\t') {
            fields.emplace_back(line.substr(begin, index - begin));
            begin = index + 1;
        }
    }
    return fields;
}

}  // namespace

std::optional<source_linked_semantic_module_v1> lower_cell_source_to_semantic_ir_v1(
    const std::string& source, semantic_vertical_slice_status_v1* status) noexcept {
    if (source.empty()) {
        set_status(status, semantic_vertical_slice_status_v1::invalid_source);
        return std::nullopt;
    }
    source_linked_semantic_module_v1 module;
    std::istringstream input(source);
    std::string line;
    std::uint32_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (const auto field = line.find("field void "); field != std::string::npos) {
            const auto begin = field + 11;
            const auto end = line.find('(', begin);
            if (end != std::string::npos) module.field = trim(
                std::string_view(line).substr(begin, end - begin));
        }
        if (const auto profile = line.find("given ce::profile(");
            profile != std::string::npos) {
            const auto begin = profile + 18;
            const auto end = line.find(')', begin);
            if (end != std::string::npos) module.profile = trim(
                std::string_view(line).substr(begin, end - begin));
        }

        std::string result;
        std::string_view expression;
        if (!parse_assignment(line, &result, &expression)) continue;
        source_linked_semantic_operation_v1 operation;
        operation.result = result;
        operation.source = {line_number,
            static_cast<std::uint32_t>(line.find(result) + 1)};

        const auto arrow_open = expression.find("-[");
        const auto arrow_close = expression.find("]->", arrow_open);
        if (arrow_open != std::string_view::npos && arrow_close != std::string_view::npos) {
            operation.kind = operation_kind::relation_apply;
            const auto source_value = trim(expression.substr(0, arrow_open));
            const auto relation = trim(expression.substr(
                arrow_open + 2, arrow_close - arrow_open - 2));
            const auto semicolon = expression.find(';', arrow_close + 3);
            operation.destination_domain = trim(expression.substr(
                arrow_close + 3, semicolon - (arrow_close + 3)));
            operation.operands = {source_value, relation};
        } else if (parse_call(expression, "ce::contract_on", &operation.operands)) {
            operation.kind = operation_kind::contract_on_support;
        } else if (parse_call(expression, "ce::segment_normalize", &operation.operands)) {
            operation.kind = operation_kind::segment_normalize;
        } else {
            continue;
        }
        operation.identity = operation_identity(operation.kind, operation.result,
                                                operation.operands,
                                                operation.destination_domain);
        module.operations.push_back(std::move(operation));
    }
    if (module.field.empty()) {
        set_status(status, semantic_vertical_slice_status_v1::missing_field);
        return std::nullopt;
    }
    if (module.profile.empty()) {
        set_status(status, semantic_vertical_slice_status_v1::missing_profile);
        return std::nullopt;
    }
    if (!validate_module(module)) {
        set_status(status, semantic_vertical_slice_status_v1::malformed_operation);
        return std::nullopt;
    }
    set_status(status, semantic_vertical_slice_status_v1::success);
    return module;
}

std::optional<std::string> write_semantic_ir_v1(
    const source_linked_semantic_module_v1& module,
    semantic_vertical_slice_status_v1* status) noexcept {
    if (!validate_module(module)) {
        set_status(status, semantic_vertical_slice_status_v1::invalid_semantic_ir);
        return std::nullopt;
    }
    std::ostringstream output;
    output << "cellerator-semantic-ir-v1\nfield\t" << module.field
           << "\nprofile\t" << module.profile << '\n';
    for (const auto& operation : module.operations) {
        output << "op\t" << kind_name(operation.kind) << '\t' << operation.result << '\t'
               << operation.source.line << '\t' << operation.source.column << '\t'
               << operation.destination_domain << '\t' << operation.operands.size();
        for (const auto& operand : operation.operands) output << '\t' << operand;
        output << '\n';
    }
    set_status(status, semantic_vertical_slice_status_v1::success);
    return output.str();
}

std::optional<source_linked_semantic_module_v1> read_semantic_ir_v1(
    const std::string& text, semantic_vertical_slice_status_v1* status) noexcept {
    std::istringstream input(text);
    std::string line;
    if (!std::getline(input, line) || line != "cellerator-semantic-ir-v1") {
        set_status(status, semantic_vertical_slice_status_v1::invalid_semantic_ir);
        return std::nullopt;
    }
    source_linked_semantic_module_v1 module;
    while (std::getline(input, line)) {
        const auto fields = split_tab(line);
        try {
            if (fields.size() == 2 && fields[0] == "field") module.field = fields[1];
            else if (fields.size() == 2 && fields[0] == "profile") module.profile = fields[1];
            else if (fields.size() >= 8 && fields[0] == "op") {
                const auto kind = parse_kind(fields[1]);
                if (!kind) throw std::invalid_argument("kind");
                source_linked_semantic_operation_v1 operation;
                operation.kind = *kind;
                operation.result = fields[2];
                operation.source = {static_cast<std::uint32_t>(std::stoul(fields[3])),
                                    static_cast<std::uint32_t>(std::stoul(fields[4]))};
                operation.destination_domain = fields[5];
                const auto count = static_cast<std::size_t>(std::stoull(fields[6]));
                if (fields.size() != 7 + count) throw std::invalid_argument("arity");
                operation.operands.assign(fields.begin() + 7, fields.end());
                operation.identity = operation_identity(
                    operation.kind, operation.result, operation.operands,
                    operation.destination_domain);
                module.operations.push_back(std::move(operation));
            } else throw std::invalid_argument("record");
        } catch (...) {
            set_status(status, semantic_vertical_slice_status_v1::invalid_semantic_ir);
            return std::nullopt;
        }
    }
    if (!validate_module(module)) {
        set_status(status, semantic_vertical_slice_status_v1::invalid_semantic_ir);
        return std::nullopt;
    }
    set_status(status, semantic_vertical_slice_status_v1::success);
    return module;
}

bool operation_core_compatible_v1(const source_linked_semantic_module_v1& module) noexcept {
    return validate_module(module) &&
        std::all_of(module.operations.begin(), module.operations.end(), [](const auto& operation) {
            return cellerator::compute::operation::v2::valid_operation_kind(operation.kind) &&
                cellerator::compute::operation::v2::valid_stable_id(operation.identity);
        });
}

std::optional<std::unordered_map<std::string, double>> execute_semantic_referee_v1(
    const source_linked_semantic_module_v1& module,
    std::unordered_map<std::string, double> bindings,
    semantic_vertical_slice_status_v1* status) noexcept {
    if (!validate_module(module)) {
        set_status(status, semantic_vertical_slice_status_v1::invalid_semantic_ir);
        return std::nullopt;
    }
    for (const auto& operation : module.operations) {
        std::vector<double> values;
        for (const auto& operand : operation.operands) {
            const auto found = bindings.find(operand);
            if (found == bindings.end()) {
                set_status(status, semantic_vertical_slice_status_v1::missing_referee_binding);
                return std::nullopt;
            }
            values.push_back(found->second);
        }
        double result = 0.0;
        switch (operation.kind) {
            case operation_kind::relation_apply:
                result = values[0] * values[1];
                break;
            case operation_kind::contract_on_support:
                result = 1.0;
                for (const auto value : values) result *= value;
                break;
            case operation_kind::segment_normalize:
                if (values.size() != 2) {
                    set_status(status, semantic_vertical_slice_status_v1::invalid_referee_operation);
                    return std::nullopt;
                }
                result = values[0] / std::max(std::abs(values[1]), 1.0);
                break;
            default:
                set_status(status, semantic_vertical_slice_status_v1::invalid_referee_operation);
                return std::nullopt;
        }
        bindings[operation.result] = result;
    }
    set_status(status, semantic_vertical_slice_status_v1::success);
    return bindings;
}

std::optional<semantic_source_receipt_v1> make_source_linked_receipt_v1(
    const std::string& source, const source_linked_semantic_module_v1& module,
    semantic_vertical_slice_status_v1* status) noexcept {
    const auto text = write_semantic_ir_v1(module, status);
    if (!text) return std::nullopt;
    semantic_source_receipt_v1 receipt;
    receipt.source_hash = stable_hash(source);
    receipt.semantic_hash = stable_hash(*text);
    receipt.operation_count = static_cast<std::uint32_t>(module.operations.size());
    receipt.exact_source_mapping = std::all_of(
        module.operations.begin(), module.operations.end(), [](const auto& operation) {
            return operation.source.line != 0 && operation.source.column != 0;
        });
    receipt.operation_core_compatible = operation_core_compatible_v1(module);
    set_status(status, semantic_vertical_slice_status_v1::success);
    return receipt;
}

}  // namespace Cellerator::compiler::ir::semantic
