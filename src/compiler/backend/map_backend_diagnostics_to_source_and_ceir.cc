#include <Cellerator/compiler/backend/map_backend_diagnostics_to_source_and_ceir_v1.hh>

#include <sstream>

namespace cellerator::compiler::backend::v1 {

bool map_backend_diagnostic_v1(
    const backend_diagnostic_input_v1& input,
    const std::vector<generated_source_map_entry_v1>& source_map,
    bool include_generated_note,
    mapped_backend_diagnostic_v1* output) noexcept {
    if (output == nullptr || input.generated_file.empty()
        || input.generated_line == 0 || input.message.empty())
        return false;
    const generated_source_map_entry_v1* match = nullptr;
    for (const auto& entry : source_map) {
        if (entry.generated_file == input.generated_file
            && input.generated_line >= entry.generated_line_begin
            && input.generated_line <= entry.generated_line_end) {
            if (match != nullptr) return false;  // Ambiguous provenance fails closed.
            match = &entry;
        }
    }
    if (match == nullptr || match->source_file.empty() || match->source_line == 0
        || match->semantic_operation == 0 || match->realization_operation == 0)
        return false;
    *output = {input.severity, match->source_file, match->source_line,
        match->source_column, match->semantic_operation,
        match->realization_operation, input.message, {}};
    if (include_generated_note) {
        std::ostringstream note;
        note << "generated at " << input.generated_file << ':'
             << input.generated_line << ':' << input.generated_column;
        output->generated_code_note = note.str();
    }
    return true;
}

}  // namespace cellerator::compiler::backend::v1
