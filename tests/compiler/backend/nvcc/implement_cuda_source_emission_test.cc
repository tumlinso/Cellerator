#include <Cellerator/compiler/backend/nvcc/implement_cuda_source_emission_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::backend::nvcc::v1;
    realized_cuda_module module{"field.generated.cu", "field.cell", {"cstdint"}, {
        {cuda_entity_kind::kernel, "z_kernel", "extern \"C\" void z_kernel() {}", 30u},
        {cuda_entity_kind::constant, "width", "constexpr std::uint32_t width = 32u;", 10u},
        {cuda_entity_kind::host_stub, "launch", "void launch() { z_kernel(); }", 40u},
        {cuda_entity_kind::projection_view, "view", "struct view { const float* values; };", 20u},
    }};
    emission_status status{};
    const auto first = emit_cuda_source(module, &status);
    const auto second = emit_cuda_source(module, &status);
    assert(first && second && first->text == second->text);
    assert(first->source_map.size() == 4u);
    assert(first->text.find("width") < first->text.find("struct view"));
    assert(first->text.find("z_kernel") < first->text.find("void launch"));
    module.entities.push_back(module.entities.front());
    assert(!emit_cuda_source(module, &status));
    assert(status == emission_status::duplicate_entity);
}
