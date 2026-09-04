#include <Cellerator/compiler/backend/nvptx/implement_ptx_emission_and_ptxas_assembly_v1.hh>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sys/wait.h>

namespace Cellerator::compiler::backend::nvptx {
namespace {

const char* type_name(const direct_ptx_type_v1 type) {
    switch (type) {
    case direct_ptx_type_v1::predicate: return ".pred";
    case direct_ptx_type_v1::b32: return ".b32";
    case direct_ptx_type_v1::b64: return ".b64";
    case direct_ptx_type_v1::u32: return ".u32";
    case direct_ptx_type_v1::u64: return ".u64";
    case direct_ptx_type_v1::s32: return ".s32";
    case direct_ptx_type_v1::s64: return ".s64";
    case direct_ptx_type_v1::f32: return ".f32";
    case direct_ptx_type_v1::f64: return ".f64";
    }
    return nullptr;
}

bool safe_path(const std::string& value) {
    return !value.empty() && value.find_first_of("'\n\r") == std::string::npos;
}

std::string quote_path(const std::string& value) {
    return "'" + value + "'";
}

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::uint32_t number_before(const std::string& text, const std::string& marker) {
    const auto marker_at = text.find(marker);
    if (marker_at == std::string::npos) return 0u;
    auto begin = marker_at;
    while (begin != 0u && text[begin - 1u] == ' ') --begin;
    auto number_begin = begin;
    while (number_begin != 0u && text[number_begin - 1u] >= '0' && text[number_begin - 1u] <= '9') {
        --number_begin;
    }
    if (number_begin == begin) return 0u;
    return static_cast<std::uint32_t>(std::stoul(text.substr(number_begin, begin - number_begin)));
}

}  // namespace

ptx_emission_result_v1 emit_deterministic_ptx_v1(
    const direct_ptx_kernel_v1& kernel,
    const std::uint16_t target_sm_major,
    const std::uint16_t target_sm_minor,
    const std::uint16_t ptx_version_major,
    const std::uint16_t ptx_version_minor) {
    std::string diagnostic;
    if (ptx_version_major == 0u ||
        validate_direct_ptx_kernel_v1(kernel, target_sm_major, target_sm_minor,
                                      &diagnostic) != direct_ptx_model_status_v1::success) {
        return {ptx_emission_status_v1::invalid_model, {}, std::move(diagnostic)};
    }

    std::ostringstream ptx;
    ptx << ".version " << ptx_version_major << '.' << ptx_version_minor << '\n'
        << ".target sm_" << target_sm_major << target_sm_minor << '\n'
        << ".address_size 64\n\n"
        << ".visible .entry " << kernel.symbol << '(';
    if (!kernel.parameters.empty()) ptx << '\n';
    for (std::size_t index = 0; index < kernel.parameters.size(); ++index) {
        const auto& parameter = kernel.parameters[index];
        const auto* type = type_name(parameter.type);
        if (type == nullptr) return {ptx_emission_status_v1::invalid_model, {}, "unknown parameter type"};
        ptx << "    .param .align " << parameter.alignment << ' ' << type << ' ' << parameter.name;
        ptx << (index + 1u == kernel.parameters.size() ? "\n" : ",\n");
    }
    ptx << ")\n{\n";
    for (const auto& value : kernel.registers) {
        const auto* type = type_name(value.type);
        if (type == nullptr) return {ptx_emission_status_v1::invalid_model, {}, "unknown register type"};
        ptx << "    .reg " << type << " %r" << value.identity << ";\n";
    }
    if (!kernel.registers.empty()) ptx << '\n';

    for (const auto& operation : kernel.operations) {
        if (operation.predicate_register != 0u) {
            ptx << "    @" << (operation.predicate_negated ? "!" : "") << "%r"
                << operation.predicate_register << ' ';
        } else if (operation.kind != direct_ptx_node_kind_v1::label) {
            ptx << "    ";
        }
        if (operation.kind == direct_ptx_node_kind_v1::label) {
            ptx << operation.label << ":\n";
        } else if (operation.kind == direct_ptx_node_kind_v1::barrier) {
            ptx << "bar.sync 0, " << operation.collective_threads << ";\n";
        } else if (operation.kind == direct_ptx_node_kind_v1::collective &&
                   operation.collective == direct_ptx_collective_kind_v1::vote_all &&
                   operation.result_register != 0u && operation.operand_registers.size() == 1u) {
            ptx << "vote.sync.all.pred %r" << operation.result_register << ", %r"
                << operation.operand_registers[0] << ", 0xffffffff;\n";
        } else if (operation.kind == direct_ptx_node_kind_v1::instruction) {
            if (operation.opcode == "ret" && operation.result_register == 0u &&
                operation.operand_registers.empty()) {
                ptx << "ret;\n";
            } else if (operation.opcode == "add_f32" && operation.result_register != 0u &&
                       operation.operand_registers.size() == 2u) {
                ptx << "add.f32 %r" << operation.result_register << ", %r"
                    << operation.operand_registers[0] << ", %r"
                    << operation.operand_registers[1] << ";\n";
            } else if (operation.opcode == "mov_u32" && operation.result_register != 0u &&
                       operation.operand_registers.size() == 1u) {
                ptx << "mov.u32 %r" << operation.result_register << ", %r"
                    << operation.operand_registers[0] << ";\n";
            } else {
                return {ptx_emission_status_v1::unsupported_operation, {},
                        "typed operation has no deterministic PTX spelling"};
            }
        } else {
            return {ptx_emission_status_v1::unsupported_operation, {},
                    "typed barrier or collective has no deterministic PTX spelling"};
        }
    }
    ptx << "}\n";
    return {ptx_emission_status_v1::success, ptx.str(), {}};
}

ptxas_resource_diagnostics_v1 parse_ptxas_resource_diagnostics_v1(
    const std::string& diagnostics) noexcept {
    ptxas_resource_diagnostics_v1 result;
    try {
        result.registers = number_before(diagnostics, " registers");
        result.stack_bytes = number_before(diagnostics, " bytes stack frame");
        result.spill_store_bytes = number_before(diagnostics, " bytes spill stores");
        result.spill_load_bytes = number_before(diagnostics, " bytes spill loads");
        result.shared_bytes = number_before(diagnostics, " bytes smem");
    } catch (...) {
        return {};
    }
    return result;
}

ptxas_assembly_result_v1 assemble_ptx_with_ptxas_v1(
    const ptxas_assembly_request_v1& request) {
    ptxas_assembly_result_v1 result;
    if (!safe_path(request.ptxas_executable) || !safe_path(request.ptx_path) ||
        !safe_path(request.cubin_path) || !safe_path(request.diagnostic_path) ||
        request.ptx.empty() || request.target_sm_major == 0u) {
        result.status = ptxas_assembly_status_v1::invalid_request;
        return result;
    }
    if (!std::filesystem::exists(request.ptxas_executable)) {
        result.status = ptxas_assembly_status_v1::ptxas_unavailable;
        return result;
    }
    {
        std::ofstream output(request.ptx_path, std::ios::binary | std::ios::trunc);
        output << request.ptx;
        if (!output) {
            result.status = ptxas_assembly_status_v1::write_failed;
            return result;
        }
    }
    const auto target = "sm_" + std::to_string(request.target_sm_major) +
        std::to_string(request.target_sm_minor);
    const auto command = quote_path(request.ptxas_executable) + " -v -arch=" + target + " " +
        quote_path(request.ptx_path) + " -o " + quote_path(request.cubin_path) + " >" +
        quote_path(request.diagnostic_path) + " 2>&1";
    const int raw_exit = std::system(command.c_str());
    result.exit_code = raw_exit == -1 ? -1 : WEXITSTATUS(raw_exit);
    result.diagnostics = read_file(request.diagnostic_path);
    result.resources = parse_ptxas_resource_diagnostics_v1(result.diagnostics);
    if (request.retain_ptx) result.retained_ptx_path = request.ptx_path;
    else std::filesystem::remove(request.ptx_path);
    if (result.exit_code != 0 || !std::filesystem::exists(request.cubin_path)) {
        result.status = ptxas_assembly_status_v1::assembly_failed;
        return result;
    }
    result.status = ptxas_assembly_status_v1::success;
    result.cubin_path = request.cubin_path;
    return result;
}

}  // namespace Cellerator::compiler::backend::nvptx
