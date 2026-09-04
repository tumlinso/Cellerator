#include <Cellerator/compiler/backend/deliver_the_first_cpu_object_milestone_v1.hh>

#include <cctype>
#include <filesystem>
#include <fstream>

namespace cellerator::compiler::backend::v1 {
namespace {

bool valid_symbol(const std::string& symbol) {
    if (symbol.empty()
        || !(std::isalpha(static_cast<unsigned char>(symbol[0]))
            || symbol[0] == '_')) return false;
    for (char character : symbol)
        if (!(std::isalnum(static_cast<unsigned char>(character))
                || character == '_')) return false;
    return true;
}

bool complete_pipeline(const cpu_object_pipeline_receipt_v1& pipeline) {
    return pipeline.source_hash != 0 && pipeline.syntax_ir_hash != 0
        && pipeline.semantic_ir_hash != 0 && pipeline.planning_ir_hash != 0
        && pipeline.realization_ir_hash != 0;
}

}  // namespace

compile_object_status_v1 deliver_first_cpu_object_v1(
    const first_cpu_object_request_v1& request,
    first_cpu_object_receipt_v1* receipt) noexcept {
    if (receipt == nullptr || request.cell_source_path.empty()
        || request.profile_name.empty() || !valid_symbol(request.symbol)
        || request.compiler.empty() || request.output_directory.empty()
        || !complete_pipeline(request.pipeline)
        || request.destination_offsets.size() < 2
        || request.source_indices.size() != request.relation_values.size()
        || request.destination_offsets.back() != request.source_indices.size())
        return compile_object_status_v1::invalid_argument;
    std::ifstream cell_source(request.cell_source_path);
    const std::string cell_text((std::istreambuf_iterator<char>(cell_source)), {});
    if (cell_text.find("profile") == std::string::npos
        || cell_text.find("relation") == std::string::npos)
        return compile_object_status_v1::invalid_argument;

    std::filesystem::create_directories(request.output_directory);
    const auto generated = std::filesystem::path(request.output_directory)
        / (request.symbol + ".cc");
    const auto object = std::filesystem::path(request.output_directory)
        / (request.symbol + ".o");
    const auto depfile = std::filesystem::path(request.output_directory)
        / (request.symbol + ".d");
    std::ofstream output(generated);
    output << "// profile: " << request.profile_name << "\n"
           << "extern \"C\" void " << request.symbol
           << "(const float* input, float* result) {\n";
    for (std::size_t destination = 0;
         destination + 1 < request.destination_offsets.size(); ++destination) {
        output << "  result[" << destination << "] = 0.0f";
        for (std::uint64_t edge = request.destination_offsets[destination];
             edge < request.destination_offsets[destination + 1]; ++edge) {
            output << " + static_cast<float>(" << request.relation_values[edge]
                   << ") * input[" << request.source_indices[edge] << ']';
        }
        output << ";\n";
    }
    output << "}\n";
    output.close();

    *receipt = {};
    receipt->generated_source_path = generated.string();
    receipt->object_path = object.string();
    receipt->pipeline = request.pipeline;
    receipt->profile_bound = true;
    const auto status = compile_generated_cpp_object_v1({request.compiler,
        generated.string(), object.string(), depfile.string(),
        request.output_directory, {"-fPIC"}, {}, {"Cellerator"}},
        &receipt->compilation);
    receipt->source_to_native_provenance = status == compile_object_status_v1::success;
    return status;
}

}  // namespace cellerator::compiler::backend::v1
