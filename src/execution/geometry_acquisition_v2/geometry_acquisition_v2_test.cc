#include <Cellerator/execution/geometry_acquisition_v2.hh>

#include <cstdlib>

using namespace cellerator::execution;
using namespace cellerator::execution::acquisition_v2;

namespace {

void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

status query(const acquisition_request &request, acquisition_requirements *out) noexcept {
    out->projection_count = request.projection_requirement_count;
    out->projection_chunk_count = request.projection_requirement_count;
    out->semantic_geometry = {64, 8};
    out->projections = {64, 8};
    out->catalog = {0, 1};
    out->planner = {0, 1};
    out->program = {64, 8};
    out->transient = {0, 1};
    out->diagnostics = {64, 8};
    return {};
}

status execute(const acquisition_request &, const acquisition_requirements &requirements,
    const acquisition_buffers &buffers, acquired_geometry *out) noexcept {
    out->semantic_geometry = {9, 10};
    out->semantic_geometry_image = {buffers.semantic_geometry.data, 64};
    out->projection_records = {buffers.projections.data, 64};
    out->projection_count = requirements.projection_count;
    out->prepared_program = {buffers.program.data, 64};
    out->diagnostics = {buffers.diagnostics.data, 64};
    return {};
}

void test_two_pass_facade() {
    projection_requirement projection{};
    projection.candidate = {1, 2};
    projection.logical_work_items = (std::uint64_t{1} << 32) + 3;
    acquisition_request request{};
    request.structure.low = 3;
    request.structure.high = 4;
    request.epoch.value = 1;
    std::uint64_t opaque_source = 0;
    request.source = {&opaque_source, sizeof(opaque_source)};
    request.projection_requirements = &projection;
    request.projection_requirement_count = 1;
    acquisition_facade facade{query, execute};
    acquisition_requirements requirements{};
    require(static_cast<bool>(query_requirements(facade, request, &requirements)));

    alignas(8) std::uint8_t semantic[64]{};
    alignas(8) std::uint8_t projections[64]{};
    alignas(8) std::uint8_t program[64]{};
    alignas(8) std::uint8_t diagnostics[64]{};
    acquisition_buffers buffers{};
    buffers.semantic_geometry = {semantic, sizeof(semantic)};
    buffers.projections = {projections, sizeof(projections)};
    buffers.program = {program, sizeof(program)};
    buffers.diagnostics = {diagnostics, sizeof(diagnostics)};
    acquired_geometry result{};
    require(static_cast<bool>(acquire(facade, request, requirements, buffers, &result)));
    buffers.program.bytes = 63;
    require(acquire(facade, request, requirements, buffers, &result).code
        == status_code::insufficient_capacity);
}

void test_default_assembly() {
    compiled_provider provider{};
    provider.identity = {21, 22};
    provider.provider_kind = 7;
    provider.architecture_major = 7;
    provider.primary = true;
    catalog_candidate candidate{};
    candidate.identity = {23, 24};
    candidate.provider_kind = 7;
    candidate.projection_kind = 11;
    default_assembly assembly{};
    assembly.registry = {{25, 26}, &provider, 1};
    assembly.catalog = {{27, 28}, &candidate, 1};
    assembly.planner = {{29, 30}, {query, execute}};
    require(static_cast<bool>(validate_default_assembly(assembly)));

    projection_requirement projection{};
    projection.candidate = candidate.identity;
    projection.provider_kind = provider.provider_kind;
    projection.projection_kind = candidate.projection_kind;
    projection.logical_work_items = 10;
    acquisition_request request{};
    request.structure.low = 31;
    request.epoch.value = 1;
    std::uint64_t source = 0;
    request.source = {&source, sizeof(source)};
    request.projection_requirements = &projection;
    request.projection_requirement_count = 1;
    acquisition_requirements requirements{};
    require(static_cast<bool>(query_default_assembly(assembly, request, &requirements)));
    candidate.experimental = true;
    require(query_default_assembly(assembly, request, &requirements).code
        == status_code::invalid_argument);
}

}  // namespace

int main() {
    test_two_pass_facade();
    test_default_assembly();
    return 0;
}
