#include <bench/ce_live/runtime_fixture/quantitative_fixture.hh>

#include <cstring>

namespace cellerator::ce_live {
namespace {

int hex_digit(char value) noexcept {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

template<typename Identity, typename Handle>
bool intern(execution::identity_registry *registry,
    Identity identity, Handle *handle) noexcept {
    return execution::intern_identity(registry, identity, handle)
        == execution::identity_registry_status::ok;
}

bool valid_support(const destination_row_csr_view &support) noexcept {
    if (support.destination_offsets == nullptr
        || (support.logical_edge_count != 0u
            && support.source_indices == nullptr)
        || support.source_count == 0u || support.destination_count == 0u
        || support.destination_offsets[0] != 0u
        || support.destination_offsets[support.destination_count]
            != support.logical_edge_count)
        return false;
    for (std::uint32_t row = 0u; row < support.destination_count; ++row) {
        const std::uint64_t begin = support.destination_offsets[row];
        const std::uint64_t end = support.destination_offsets[row + 1u];
        if (begin > end || end > support.logical_edge_count) return false;
        for (std::uint64_t edge = begin; edge < end; ++edge)
            if (support.source_indices[edge] >= support.source_count)
                return false;
    }
    return true;
}

execution::value_plane make_value_plane(
    const execution::relation_structure &structure, float *values,
    std::uint64_t generation) noexcept {
    execution::value_plane plane{};
    plane.structure = structure.identity;
    plane.structure_epoch_value = structure.epoch;
    plane.values = values;
    plane.location = {execution::residency_kind::host, {}, -1, 0u};
    plane.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::logical_edge_order;
    plane.generation = {generation};
    plane.element_count = structure.logical_edge_count;
    plane.value_bytes = structure.logical_edge_count * sizeof(float);
    return plane;
}

} // namespace

quantitative_fixture_status identity_from_sha256(
    const char *hex, std::uint64_t *low, std::uint64_t *high) noexcept {
    if (hex == nullptr || low == nullptr || high == nullptr)
        return quantitative_fixture_status::invalid_argument;
    if (std::strlen(hex) != 64u)
        return quantitative_fixture_status::invalid_identity;
    std::uint64_t words[2]{};
    for (std::size_t byte = 0u; byte < 16u; ++byte) {
        const int upper = hex_digit(hex[2u * byte]);
        const int lower = hex_digit(hex[2u * byte + 1u]);
        if (upper < 0 || lower < 0)
            return quantitative_fixture_status::invalid_identity;
        words[byte / 8u] = (words[byte / 8u] << 8u)
            | static_cast<std::uint64_t>((upper << 4) | lower);
    }
    if (words[0] == 0u && words[1] == 0u)
        return quantitative_fixture_status::invalid_identity;
    *high = words[0];
    *low = words[1];
    return quantitative_fixture_status::ok;
}

quantitative_fixture_identities pbmc3k_quantitative_v1_identities() noexcept {
    quantitative_fixture_identities ids{};
    identity_from_sha256(
        "46c0b8e197efcc3099e90064f068b973261c39b25708879910b9395aa19903fd",
        &ids.feature_domain.low, &ids.feature_domain.high);
    identity_from_sha256(
        "d1a036e8f5daeef3fc4bd332d3fc3c3ed36982a4d141d7e296041e56da97320a",
        &ids.observation_domain.low, &ids.observation_domain.high);
    identity_from_sha256(
        "8063dd906ee17e0081879c42e6617bb125aa12145cbd9278711d36f73cd2b77a",
        &ids.feature_order.low, &ids.feature_order.high);
    identity_from_sha256(
        "f28d0adacfb46ff6d29015ee049a16b82ce8e90c0e7eff45856e9c6f7dd96cde",
        &ids.observation_order.low, &ids.observation_order.high);
    identity_from_sha256(
        "ebf75567ed40a8c7de9b7b386503fc2b82541697345364044cefd2205e250f83",
        &ids.geometry.low, &ids.geometry.high);
    identity_from_sha256(
        "4dccc668b51302cf954734a0f4ea55edb3e8d0d9cd145dbe2c5c1c04d008024b",
        &ids.partition.low, &ids.partition.high);
    identity_from_sha256(
        "5ec566e0bd56b468e9025ffe7c75fc54a4cf0eae2bc93107ae570fae188a7ccb",
        &ids.structure.low, &ids.structure.high);
    identity_from_sha256(
        "f5813da4defcdeb2f40efac3d74791e415e4302d573428d960b280a1c0bf926f",
        &ids.destination_row_csr_projection.low,
        &ids.destination_row_csr_projection.high);
    return ids;
}

quantitative_fixture_status bind_quantitative_fixture(
    const quantitative_fixture_arrays &arrays,
    const quantitative_fixture_identities &identities,
    execution::identity_registry *registry,
    execution::projection_catalog_handle projection_catalog,
    native_quantitative_relation *relation) noexcept {
    if (registry == nullptr || relation == nullptr
        || arrays.generation_1_values == nullptr
        || arrays.generation_2_values == nullptr
        || !execution::valid_projection_catalog(projection_catalog))
        return quantitative_fixture_status::invalid_argument;
    if (!valid_support(arrays.support))
        return quantitative_fixture_status::invalid_support;

    execution::domain_handle feature_domain{}, observation_domain{};
    execution::order_handle feature_order{}, observation_order{};
    execution::geometry_handle geometry{};
    execution::partition_handle partition{};
    execution::structure_handle structure{};
    execution::projection_handle projection{};
    if (!intern(registry, identities.feature_domain, &feature_domain)
        || !intern(registry, identities.observation_domain, &observation_domain)
        || !intern(registry, identities.feature_order, &feature_order)
        || !intern(registry, identities.observation_order, &observation_order)
        || !intern(registry, identities.geometry, &geometry)
        || !intern(registry, identities.partition, &partition)
        || !intern(registry, identities.structure, &structure)
        || !intern(registry, identities.destination_row_csr_projection,
            &projection))
        return quantitative_fixture_status::registry_failure;

    const execution::axis_identity source_axis{
        feature_domain, feature_order, geometry, partition};
    const execution::axis_identity destination_axis{
        observation_domain, observation_order, geometry, partition};

    native_quantitative_relation result{};
    result.projection = arrays.support;
    result.structure = {structure, {1u}, source_axis, destination_axis,
        projection_catalog, arrays.support.logical_edge_count};
    result.operand.source_axis = source_axis;
    result.operand.destination_axis = destination_axis;
    result.operand.structure = structure;
    result.operand.projection = projection;
    result.operand.epoch = result.structure.epoch;
    result.operand.projection_data = &result.projection;
    result.operand.projection_bytes = sizeof(result.projection);
    result.operand.logical_edge_count = arrays.support.logical_edge_count;
    result.operand.location = {
        execution::residency_kind::host, {}, -1, 0u};
    result.generations[0] = make_value_plane(
        result.structure, arrays.generation_1_values, 1u);
    result.generations[1] = make_value_plane(
        result.structure, arrays.generation_2_values, 2u);
    *relation = result;
    // Repair the self-reference after copying the aggregate to its owner.
    relation->operand.projection_data = &relation->projection;
    return quantitative_fixture_status::ok;
}

float deterministic_dense_operand(
    std::uint32_t source, std::uint32_t lane) noexcept {
    const std::uint32_t mixed = ((source + 1u) * 17u + (lane + 3u) * 29u)
        % 37u;
    return static_cast<float>(static_cast<int>(mixed) - 18) / 11.0f;
}

void fill_deterministic_dense_operand(float *values,
    std::uint32_t source_count, std::uint32_t dense_width) noexcept {
    if (values == nullptr) return;
    for (std::uint32_t source = 0u; source < source_count; ++source)
        for (std::uint32_t lane = 0u; lane < dense_width; ++lane)
            values[static_cast<std::size_t>(source) * dense_width + lane]
                = deterministic_dense_operand(source, lane);
}

} // namespace cellerator::ce_live
