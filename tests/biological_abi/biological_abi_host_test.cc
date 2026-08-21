#include <Baseplane/seq/dna2_views.hh>
#include <Cellerator/execution/biological_abi.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace ce = cellerator::execution;
namespace bp = baseplane::seq;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "celleratorBiologicalAbiHostTest: " << message << '\n';
        std::exit(1);
    }
}

ce::axis_identity axis(
    ce::u32 domain, ce::u32 order, ce::u32 geometry, ce::u32 partition) {
    return ce::axis_identity{
        {domain, 1u}, {order, 1u}, {geometry, 1u}, {partition, 1u}};
}

ce::device_location host_location() {
    return ce::device_location{ce::residency_kind::host, {}, -1, 0u};
}

ce::dense_tensor_view dense_view(
    void *data, ce::axis_identity rows, ce::axis_identity columns) {
    ce::dense_tensor_view result{};
    result.data = data;
    result.location = host_location();
    result.value_type = ce::numeric_type::f32;
    result.rank = 2u;
    result.axes[0] = rows;
    result.axes[1] = columns;
    result.shape[0] = 3u;
    result.shape[1] = 5u;
    result.stride[0] = 5;
    result.stride[1] = 1;
    return result;
}

ce::bit_plane_view adapt_baseplane_planes(
    bp::dna2_planes32_stream_view source,
    const std::uint32_t *validity,
    std::uint32_t base_count,
    ce::axis_identity coordinate_axis) {
    return ce::bit_plane_view{
        coordinate_axis,
        source.lo_words,
        source.hi_words,
        validity,
        host_location(),
        static_cast<ce::u32>(source.n_words),
        base_count};
}

void test_identity_and_relocation() {
    const ce::axis_identity cells = axis(1u, 10u, 20u, 30u);
    const ce::axis_identity genes = axis(2u, 11u, 21u, 31u);
    const ce::axis_identity other_genes = axis(3u, 11u, 21u, 31u);
    const ce::axis_identity packed_genes = axis(2u, 12u, 21u, 31u);
    const ce::axis_identity other_geometry = axis(2u, 11u, 121u, 31u);
    const ce::axis_identity other_partition = axis(2u, 11u, 21u, 32u);

    std::array<float, 15> first{}, relocated{};
    const ce::dense_tensor_view original = dense_view(
        first.data(), cells, genes);
    const ce::dense_tensor_view moved = dense_view(
        relocated.data(), cells, genes);
    require(ce::same_dense_contract(original, moved),
        "pointer relocation changed semantic identity");
    require(!ce::same_dense_contract(
        original, dense_view(relocated.data(), cells, other_genes)),
        "equal shapes with different domains were interchangeable");
    require(!ce::same_dense_contract(
        original, dense_view(relocated.data(), cells, packed_genes)),
        "equal domains with different orders were interchangeable");
    require(!ce::same_dense_contract(
        original, dense_view(relocated.data(), cells, other_geometry)),
        "equal orders with different geometry were interchangeable");
    require(!ce::same_dense_contract(
        original, dense_view(relocated.data(), cells, other_partition)),
        "equal domains with different partitions were interchangeable");

    ce::persistent_axis_identity persistent{
        {ce::biological_abi_version,
            ce::serialized_record_kind::persistent_axis_identity,
            sizeof(ce::persistent_axis_identity)},
        {1u, 101u}, {2u, 102u}, {3u, 103u}, {4u, 104u}};
    alignas(ce::persistent_axis_identity)
        std::array<std::byte, sizeof(ce::persistent_axis_identity)> bytes{};
    std::memcpy(bytes.data(), &persistent, sizeof(persistent));
    const auto *copy = reinterpret_cast<const ce::persistent_axis_identity *>(
        bytes.data());
    require(ce::validate_persistent_axis_identity(*copy)
            == ce::biological_validation_code::ok,
        "relocated persistent identity did not validate");
    persistent.header.schema_version = ce::biological_abi_version + 1u;
    require(ce::validate_persistent_axis_identity(persistent)
            == ce::biological_validation_code::unsupported_version,
        "version mismatch was accepted");
}

void test_sequence_views() {
    const ce::axis_identity coordinates = axis(50u, 51u, 52u, 53u);
    const ce::sequence_domain chunk{
        {50u, 1u}, 7u, 9u, (1ull << 40u), 33u, 4u, 29u, 4u, 4u};
    require(ce::validate_sequence_domain(chunk)
            == ce::biological_validation_code::ok,
        "large global coordinate with bounded local positions failed");

    std::uint32_t low[2]{}, high[2]{}, validity[2]{~0u, 1u};
    const bp::dna2_planes32_stream_view baseplane{low, high, 2u};
    const ce::bit_plane_view bits = adapt_baseplane_planes(
        baseplane, validity, 33u, coordinates);
    require(ce::validate_bit_plane(bits) == ce::biological_validation_code::ok,
        "Baseplane plane adapter did not preserve explicit tail validity");
    ce::bit_plane_view missing_validity = bits;
    missing_validity.validity = nullptr;
    require(ce::validate_bit_plane(missing_validity)
            == ce::biological_validation_code::missing_pointer,
        "nonempty plane without validity was accepted");

    ce::bit_plane_view empty{coordinates, nullptr, nullptr, nullptr,
        host_location(), 0u, 0u};
    require(ce::validate_bit_plane(empty) == ce::biological_validation_code::ok,
        "empty bit plane required storage");

    std::uint32_t positions[2]{3u, 31u};
    std::uint16_t rules[2]{1u, 2u};
    std::uint8_t attributes[2]{}, strands[2]{};
    ce::event_stream_view events{
        axis(60u, 61u, 62u, 63u), chunk,
        positions, rules, attributes, strands, host_location(),
        3u, 2u, 1u, ce::event_ordering::coordinate_stable, {}};
    require(ce::validate_event_stream(events)
            == ce::biological_validation_code::ok,
        "valid bounded event stream failed");
    events.dropped_records = 0u;
    require(ce::validate_event_stream(events)
            == ce::biological_validation_code::invalid_count,
        "inconsistent total/stored/dropped event counts were accepted");

    ce::sequence_domain overflow = chunk;
    overflow.global_base_begin = ~ce::u64{0} - 16u;
    require(ce::validate_sequence_domain(overflow)
            == ce::biological_validation_code::invalid_sequence_domain,
        "overflowing global sequence coordinate was accepted");
}

void test_operand_discrimination() {
    std::array<float, 15> values{};
    ce::biological_operand_view operand{};
    operand.kind = ce::operand_kind::dense_tensor;
    operand.storage.dense = dense_view(
        values.data(), axis(1u, 2u, 3u, 4u), axis(5u, 6u, 7u, 8u));
    require(ce::validate_operand(operand) == ce::biological_validation_code::ok,
        "dense biological operand failed validation");
    operand.kind = static_cast<ce::operand_kind>(255u);
    require(ce::validate_operand(operand)
            == ce::biological_validation_code::invalid_operand_kind,
        "unknown operand kind was accepted");
}

} // namespace

int main() {
    static_assert(std::is_trivially_copyable<ce::bit_plane_view>::value, "POD");
    static_assert(std::is_trivially_copyable<ce::event_stream_view>::value, "POD");
    static_assert(std::is_trivially_copyable<ce::segment_stream_view>::value, "POD");
    static_assert(std::is_trivially_copyable<ce::sparse_relation_view>::value, "POD");
    test_identity_and_relocation();
    test_sequence_views();
    test_operand_discrimination();
    std::cout << "celleratorBiologicalAbiHostTest passed"
              << " axis_bytes=" << sizeof(ce::axis_identity)
              << " operand_bytes=" << sizeof(ce::biological_operand_view)
              << " sequence_domain_bytes=" << sizeof(ce::sequence_domain)
              << '\n';
    return 0;
}
