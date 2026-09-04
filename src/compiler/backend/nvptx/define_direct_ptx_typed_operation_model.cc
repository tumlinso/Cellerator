#include <Cellerator/compiler/backend/nvptx/define_direct_ptx_typed_operation_model_v1.hh>

#include <cctype>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace Cellerator::compiler::backend::nvptx {
namespace {

constexpr std::uint16_t known_memory_effects = direct_ptx_memory_read_v1 |
    direct_ptx_memory_write_v1 | direct_ptx_memory_atomic_v1 |
    direct_ptx_memory_volatile_v1;

bool valid_name(const std::string& value) {
    if (value.empty() || !(std::isalpha(static_cast<unsigned char>(value.front())) ||
                           value.front() == '_')) return false;
    for (const char character : value) {
        if (!(std::isalnum(static_cast<unsigned char>(character)) || character == '_')) {
            return false;
        }
    }
    return true;
}

template <class Enum>
bool read_enum(std::istream& input, Enum* result) {
    unsigned value = 0u;
    if (!(input >> value) || value > 255u) return false;
    *result = static_cast<Enum>(value);
    return true;
}

void fail(std::string* diagnostic, const char* message) {
    if (diagnostic != nullptr) *diagnostic = message;
}

}  // namespace

direct_ptx_model_status_v1 validate_direct_ptx_kernel_v1(
    const direct_ptx_kernel_v1& kernel,
    const std::uint16_t target_sm_major,
    const std::uint16_t target_sm_minor,
    std::string* diagnostic) noexcept {
    if (kernel.realization_kernel_identity == 0u || !valid_name(kernel.symbol) ||
        kernel.operations.empty() || target_sm_major == 0u) {
        fail(diagnostic, "kernel identity, symbol, operations, or target is invalid");
        return direct_ptx_model_status_v1::invalid_kernel;
    }

    std::unordered_set<std::string> parameter_names;
    for (const auto& parameter : kernel.parameters) {
        if (!valid_name(parameter.name) || parameter.address_space == direct_ptx_address_space_v1::none ||
            parameter.alignment == 0u || (parameter.alignment & (parameter.alignment - 1u)) != 0u ||
            !parameter_names.insert(parameter.name).second) {
            fail(diagnostic, "parameter name, address space, alignment, or uniqueness is invalid");
            return direct_ptx_model_status_v1::invalid_parameter;
        }
    }

    std::unordered_set<std::uint32_t> registers;
    std::unordered_set<std::uint32_t> predicates;
    for (const auto& value : kernel.registers) {
        if (value.identity == 0u || !registers.insert(value.identity).second) {
            fail(diagnostic, "register identity is zero or duplicated");
            return direct_ptx_model_status_v1::invalid_register;
        }
        if (value.type == direct_ptx_type_v1::predicate) predicates.insert(value.identity);
    }

    std::unordered_set<std::uint64_t> node_identities;
    std::unordered_set<std::string> labels;
    for (const auto& operation : kernel.operations) {
        if (operation.realization_node_identity == 0u ||
            !node_identities.insert(operation.realization_node_identity).second ||
            (operation.memory_effects & ~known_memory_effects) != 0u) {
            fail(diagnostic, "operation identity or memory effects are invalid");
            return direct_ptx_model_status_v1::invalid_operation;
        }
        if (operation.result_register != 0u && registers.count(operation.result_register) == 0u) {
            fail(diagnostic, "operation result references an unknown register");
            return direct_ptx_model_status_v1::invalid_reference;
        }
        for (const auto operand : operation.operand_registers) {
            if (registers.count(operand) == 0u) {
                fail(diagnostic, "operation operand references an unknown register");
                return direct_ptx_model_status_v1::invalid_reference;
            }
        }
        if (operation.predicate_register != 0u && predicates.count(operation.predicate_register) == 0u) {
            fail(diagnostic, "operation predicate references a non-predicate register");
            return direct_ptx_model_status_v1::invalid_reference;
        }
        const auto& requirement = operation.requirement;
        if (requirement.minimum_sm_major > target_sm_major ||
            (requirement.minimum_sm_major == target_sm_major &&
             requirement.minimum_sm_minor > target_sm_minor)) {
            fail(diagnostic, "instruction requirement exceeds the selected target");
            return direct_ptx_model_status_v1::unsupported_requirement;
        }

        switch (operation.kind) {
        case direct_ptx_node_kind_v1::label:
            if (!valid_name(operation.label) || !labels.insert(operation.label).second ||
                !operation.opcode.empty() || operation.result_register != 0u ||
                !operation.operand_registers.empty()) {
                fail(diagnostic, "label node is malformed or duplicated");
                return direct_ptx_model_status_v1::invalid_operation;
            }
            break;
        case direct_ptx_node_kind_v1::instruction:
            if (!valid_name(operation.opcode) || operation.barrier != direct_ptx_barrier_kind_v1::none ||
                operation.collective != direct_ptx_collective_kind_v1::none) {
                fail(diagnostic, "instruction opcode or specialized effect is invalid");
                return direct_ptx_model_status_v1::invalid_operation;
            }
            break;
        case direct_ptx_node_kind_v1::barrier:
            if (operation.barrier == direct_ptx_barrier_kind_v1::none ||
                operation.collective_scope == direct_ptx_collective_scope_v1::none ||
                operation.collective_threads == 0u) {
                fail(diagnostic, "barrier kind, scope, or participant count is invalid");
                return direct_ptx_model_status_v1::invalid_operation;
            }
            break;
        case direct_ptx_node_kind_v1::collective:
            if (operation.collective == direct_ptx_collective_kind_v1::none ||
                operation.collective_scope == direct_ptx_collective_scope_v1::none ||
                operation.collective_threads == 0u) {
                fail(diagnostic, "collective kind, scope, or participant count is invalid");
                return direct_ptx_model_status_v1::invalid_operation;
            }
            break;
        default:
            fail(diagnostic, "operation kind is unknown");
            return direct_ptx_model_status_v1::invalid_operation;
        }
    }
    if (diagnostic != nullptr) diagnostic->clear();
    return direct_ptx_model_status_v1::success;
}

std::string print_direct_ptx_kernel_model_v1(const direct_ptx_kernel_v1& kernel) {
    std::ostringstream output;
    output << "kernel " << kernel.realization_kernel_identity << ' ' << std::quoted(kernel.symbol) << '\n';
    for (const auto& parameter : kernel.parameters) {
        output << "param " << std::quoted(parameter.name) << ' '
               << static_cast<unsigned>(parameter.type) << ' '
               << static_cast<unsigned>(parameter.address_space) << ' '
               << parameter.alignment << '\n';
    }
    for (const auto& value : kernel.registers) {
        output << "reg " << value.identity << ' ' << static_cast<unsigned>(value.type) << '\n';
    }
    for (const auto& operation : kernel.operations) {
        output << "node " << operation.realization_node_identity << ' '
               << static_cast<unsigned>(operation.kind) << ' ' << std::quoted(operation.opcode) << ' '
               << operation.result_register << ' ' << operation.predicate_register << ' '
               << operation.predicate_negated << ' ' << std::quoted(operation.label) << ' '
               << static_cast<unsigned>(operation.address_space) << ' ' << operation.memory_effects << ' '
               << static_cast<unsigned>(operation.barrier) << ' '
               << static_cast<unsigned>(operation.collective) << ' '
               << static_cast<unsigned>(operation.collective_scope) << ' '
               << operation.collective_threads << ' ' << operation.requirement.minimum_sm_major << ' '
               << operation.requirement.minimum_sm_minor << ' '
               << std::quoted(operation.requirement.feature) << ' '
               << operation.operand_registers.size();
        for (const auto operand : operation.operand_registers) output << ' ' << operand;
        output << '\n';
    }
    output << "end\n";
    return output.str();
}

direct_ptx_model_status_v1 parse_direct_ptx_kernel_model_v1(
    const std::string_view text,
    direct_ptx_kernel_v1* kernel,
    std::string* diagnostic) {
    if (kernel == nullptr) {
        fail(diagnostic, "output kernel is null");
        return direct_ptx_model_status_v1::parse_error;
    }
    direct_ptx_kernel_v1 parsed;
    std::istringstream input{std::string(text)};
    std::string record;
    if (!(input >> record) || record != "kernel" ||
        !(input >> parsed.realization_kernel_identity >> std::quoted(parsed.symbol))) {
        fail(diagnostic, "kernel header is malformed");
        return direct_ptx_model_status_v1::parse_error;
    }
    while (input >> record) {
        if (record == "end") {
            input >> std::ws;
            if (!input.eof()) break;
            *kernel = std::move(parsed);
            if (diagnostic != nullptr) diagnostic->clear();
            return direct_ptx_model_status_v1::success;
        }
        if (record == "param") {
            direct_ptx_parameter_v1 parameter;
            if (!(input >> std::quoted(parameter.name)) || !read_enum(input, &parameter.type) ||
                !read_enum(input, &parameter.address_space) || !(input >> parameter.alignment)) break;
            parsed.parameters.push_back(std::move(parameter));
        } else if (record == "reg") {
            direct_ptx_register_v1 value;
            if (!(input >> value.identity) || !read_enum(input, &value.type)) break;
            parsed.registers.push_back(value);
        } else if (record == "node") {
            direct_ptx_operation_v1 operation;
            std::size_t operand_count = 0u;
            if (!(input >> operation.realization_node_identity) || !read_enum(input, &operation.kind) ||
                !(input >> std::quoted(operation.opcode) >> operation.result_register >>
                  operation.predicate_register >> operation.predicate_negated >> std::quoted(operation.label)) ||
                !read_enum(input, &operation.address_space) || !(input >> operation.memory_effects) ||
                !read_enum(input, &operation.barrier) || !read_enum(input, &operation.collective) ||
                !read_enum(input, &operation.collective_scope) ||
                !(input >> operation.collective_threads >> operation.requirement.minimum_sm_major >>
                  operation.requirement.minimum_sm_minor >> std::quoted(operation.requirement.feature) >>
                  operand_count)) break;
            operation.operand_registers.resize(operand_count);
            bool operands_valid = true;
            for (auto& operand : operation.operand_registers) operands_valid = operands_valid && bool(input >> operand);
            if (!operands_valid) break;
            parsed.operations.push_back(std::move(operation));
        } else {
            break;
        }
    }
    fail(diagnostic, "typed PTX model text is malformed or lacks an end record");
    return direct_ptx_model_status_v1::parse_error;
}

}  // namespace Cellerator::compiler::backend::nvptx
