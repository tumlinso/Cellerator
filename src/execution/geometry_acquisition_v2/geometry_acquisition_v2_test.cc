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

struct external_fixture {
    std::uint8_t bytes[16]{};
    bool return_outside = false;
};

status describe_external(void *, const external_payload_query &query,
    external_payload_descriptor *descriptor) noexcept {
    descriptor->payload_identity = query.payload_identity;
    descriptor->encoding = query.encoding;
    descriptor->payload_bytes = 16;
    descriptor->content_hash[0] = 1;
    return {};
}

status read_external(void *context, const external_payload_descriptor &descriptor,
    byte_span destination, immutable_byte_span *payload) noexcept {
    auto *fixture = static_cast<external_fixture *>(context);
    *payload = fixture->return_outside
        ? immutable_byte_span{fixture->bytes, descriptor.payload_bytes}
        : immutable_byte_span{destination.data, descriptor.payload_bytes};
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

void test_chunked_projection_and_explicit_fallback() {
    projection_requirement requirements[2]{};
    requirements[0].candidate = {41, 42};
    requirements[0].logical_work_items = (std::uint64_t{1} << 32) + 5;
    requirements[1].candidate = {43, 44};
    requirements[1].logical_work_items = 7;
    acquisition_request request{};
    request.structure.low = 45;
    request.epoch.value = 1;
    std::uint64_t source = 0;
    request.source = {&source, sizeof(source)};
    request.projection_requirements = requirements;
    request.projection_requirement_count = 2;

    projection_record projections[2]{};
    projections[0] = {{41, 42}, {51, 52}, requirements[0].logical_work_items,
        requirements[0].logical_work_items + 3, 0, 2,
        logical_primary_values, true, {}};
    projections[1] = {{43, 44}, {53, 54}, 7, 8, 2, 1,
        logical_primary_values, true, {}};
    projection_chunk chunks[3]{};
    chunks[0] = {0, 0, 0, 0xffffffffu, 32, {}, 0, 16};
    chunks[1] = {0, 1, 0xffffffffu, 6, 16, {}, 16, 16};
    chunks[2] = {1, 0, 0, 7, 16, {}, 32, 16};
    projection_set set{projections, 2, chunks, 3, 48};
    require(static_cast<bool>(validate_projection_set(request, set)));
    chunks[1].logical_begin = 0;
    require(validate_projection_set(request, set).code == status_code::invalid_result);
    chunks[1].logical_begin = 0xffffffffu;

    request.preferred_route = route::load_cpe2;
    request.cpe2 = cpe2_disposition::incompatible;
    route_resolution resolution{};
    require(resolve_route(request, &resolution).code
        == status_code::incompatible_cpe2_rejected);
    acquisition_requirements queried{};
    require(query_requirements({query, execute}, request, &queried).code
        == status_code::incompatible_cpe2_rejected);
    request.fallback = fallback_policy::rebuild_from_embedded_csg1;
    require(static_cast<bool>(resolve_route(request, &resolution)));
    require(resolution.selected == route::load_csg1
        && resolution.rebuilt_from_embedded_csg1);
}

void test_generic_external_payload_boundary() {
    external_fixture fixture{};
    external_payload_source source{&fixture, describe_external, read_external};
    external_payload_query query_request{{61, 62}, external_payload_encoding::csg1};
    external_payload_descriptor descriptor{};
    require(static_cast<bool>(describe_external_payload(source, query_request, &descriptor)));
    alignas(8) std::uint8_t destination[16]{};
    external_payload_consumption consumption{};
    require(static_cast<bool>(consume_external_payload(
        source, descriptor, {destination, sizeof(destination)}, &consumption)));

    projection_requirement projection{};
    projection.candidate = {63, 64};
    projection.logical_work_items = 1;
    acquisition_request prototype{};
    prototype.structure.low = 65;
    prototype.epoch.value = 1;
    prototype.projection_requirements = &projection;
    prototype.projection_requirement_count = 1;
    acquisition_request bound{};
    require(static_cast<bool>(bind_external_payload_request(
        consumption, prototype, &bound)));
    require(bound.preferred_route == route::load_csg1
        && bound.source.data == destination);

    fixture.return_outside = true;
    require(consume_external_payload(source, descriptor,
        {destination, sizeof(destination)}, &consumption).code
        == status_code::invalid_source);
    require(consume_external_payload(source, descriptor,
        {destination, sizeof(destination) - 1}, &consumption).code
        == status_code::insufficient_capacity);
}

}  // namespace

int main() {
    test_two_pass_facade();
    test_default_assembly();
    test_chunked_projection_and_explicit_fallback();
    test_generic_external_payload_boundary();
    return 0;
}
