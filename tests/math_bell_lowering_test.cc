#include <Cellerator/compat/cp_math_v1/physical_bell.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace math = cellerator::compute::math;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "math_bell_lowering_test: " << message << '\n';
    std::abort();
}

void require(math::bell_lowering_status status, const char *message) {
    require(static_cast<bool>(status), message);
}

struct fixture {
    static constexpr std::uint32_t rows = 9u;
    static constexpr std::uint32_t features = 13u;

    std::vector<std::uint32_t> blocks{0u, 3u, 8u, 13u};
    std::vector<std::uint32_t> fperm{
        4u, 0u, 7u, 2u, 10u, 1u, 5u, 8u, 3u, 6u, 9u, 11u, 12u};
    std::vector<std::uint32_t> finv;
    std::vector<std::uint32_t> rperm{
        2u, 0u, 4u, 1u, 6u, 3u, 7u, 5u, 8u};
    std::vector<std::uint32_t> rinv;
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> feature_ids;
    std::vector<float> values;
    math::bell_semantic_plan_view plan{};
    cellpack::local_cell_order_view order{};
    math::bell_csr_source_view source{};

    fixture() {
        finv.resize(features);
        for (std::uint32_t execution = 0u; execution < features; ++execution)
            finv[fperm[execution]] = execution;
        rinv.resize(rows);
        for (std::uint32_t execution = 0u; execution < rows; ++execution)
            rinv[rperm[execution]] = execution;

        const std::uint32_t desired[rows][2] = {
            {0u, 3u}, {1u, 4u}, {2u, 5u}, {0u, 6u}, {1u, 7u},
            {2u, 3u}, {0u, 4u}, {1u, 5u}, {8u, 12u}};
        std::vector<std::vector<std::uint32_t>> canonical(rows);
        for (std::uint32_t er = 0u; er < rows;
             ++er) {
            auto &row = canonical[rperm[er]];
            row = {fperm[desired[er][0]],
                fperm[desired[er][1]]};
            std::sort(row.begin(), row.end());
        }
        row_offsets.push_back(0u);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t feature : canonical[row]) {
                feature_ids.push_back(feature);
                values.push_back(static_cast<float>(row * 100u + feature + 1u));
            }
            row_offsets.push_back(static_cast<std::uint32_t>(feature_ids.size()));
        }

        plan.semantic_schema_version = cellpack::packing_plan_semantic_schema_version;
        plan.full_row_count = rows;
        plan.feature_count = features;
        plan.feature_block_count = 3u;
        plan.feature_block_geometry_identity = 0x12345678ull;
        plan.row_domain_identity = 0xabcddcbaull;
        plan.feature_block_offsets = blocks.data();
        plan.feature_permutation = fperm.data();
        plan.inverse_feature_permutation = finv.data();

        order.order_schema_version = cellpack::local_cell_order_schema_version;
        order.signature_algorithm_version
            = cellpack::local_cell_signature_algorithm_version;
        order.kind = cellpack::local_cell_order_kind::inferred_minhash;
        order.window_size = 1024u;
        order.group_width = 32u;
        order.ordering_identity = 0xfedcba98ull;
        order.full_row_count = rows;
        order.row_count = rows;
        order.feature_block_count = 3u;
        order.feature_block_geometry_identity
            = plan.feature_block_geometry_identity;
        order.row_domain_identity = plan.row_domain_identity;
        order.row_permutation = rperm.data();
        order.inverse_row_permutation = rinv.data();

        source.row_count = rows;
        source.feature_count = features;
        source.nnz_count = static_cast<std::uint32_t>(feature_ids.size());
        source.value_size_bytes = sizeof(float);
        source.row_offsets = row_offsets.data();
        source.feature_ids = feature_ids.data();
        source.values = values.data();
    }
};

struct workspace {
    std::vector<std::uint32_t> markers;
    std::vector<std::uint32_t> blocks;

    explicit workspace(const fixture &data) {
        math::bell_lowering_workspace_requirements required;
        require(math::query_bell_lowering_workspace_requirements(
            data.source, data.plan, data.order, &required),
            "workspace query failed");
        markers.resize(required.marker_count);
        blocks.resize(required.feature_block_offset_count);
    }

    math::bell_lowering_workspace view() {
        return {markers.size(), markers.data(), blocks.size(),
            blocks.data()};
    }
};

std::vector<float> dense(const fixture &data) {
    std::vector<float> dense(fixture::rows * fixture::features, 0.0f);
    for (std::uint32_t row = 0u; row < fixture::rows; ++row) {
        for (std::uint32_t entry = data.row_offsets[row];
             entry < data.row_offsets[row + 1u]; ++entry) {
            dense[row * fixture::features + data.feature_ids[entry]]
                = data.values[entry];
        }
    }
    return dense;
}

void decode(
    const fixture &data,
    const math::physical_bell_view &view) {
    const auto expected = dense(data);
    std::vector<float> decoded(expected.size(), 0.0f);
    const auto *values = static_cast<const float *>(view.values);
    const std::uint32_t block_size = view.block_size;
    const std::uint32_t block_rows = view.padded_row_count / block_size;
    const std::uint32_t ell_blocks = view.ell_columns / block_size;

    for (std::uint32_t block_row = 0u; block_row < block_rows; ++block_row) {
        for (std::uint32_t slot = 0u; slot < ell_blocks; ++slot) {
            const std::size_t bi
                = static_cast<std::size_t>(block_row) * ell_blocks + slot;
            const std::int32_t pb = view.column_indices[bi];
            for (std::uint32_t lane = 0u; lane < block_size; ++lane) {
                for (std::uint32_t local = 0u; local < block_size; ++local) {
                    const float value = values[
                        bi * block_size * block_size
                        + lane * block_size + local];
                    if (pb < 0) {
                        require(value == 0.0f, "empty ELL block contains a value");
                        continue;
                    }
                    const std::uint32_t er
                        = block_row * block_size + lane;
                    const std::uint32_t pf
                        = static_cast<std::uint32_t>(pb) * block_size
                        + local;
                    if (er >= view.row_count) {
                        require(value == 0.0f, "padded BELL row contains a value");
                        continue;
                    }
                    std::uint32_t semantic = 0u;
                    while (semantic < data.plan.feature_block_count
                        && pf
                            >= view.padded_feature_block_offsets[semantic + 1u])
                        ++semantic;
                    require(semantic < data.plan.feature_block_count,
                        "physical feature lies outside padded semantic blocks");
                    const std::uint32_t within = pf
                        - view.padded_feature_block_offsets[semantic];
                    const std::uint32_t width
                        = data.blocks[semantic + 1u]
                        - data.blocks[semantic];
                    if (within >= width) {
                        require(value == 0.0f,
                            "semantic feature padding contains a value");
                        continue;
                    }
                    const std::uint32_t ef
                        = data.blocks[semantic] + within;
                    const std::uint32_t cr
                        = data.rperm[er];
                    const std::uint32_t cf
                        = data.fperm[ef];
                    decoded[cr * fixture::features + cf]
                        = value;
                }
            }
        }
    }
    require(decoded == expected, "independent BELL decode changed canonical math");
}

void test_candidates() {
    fixture data;
    workspace owner(data);
    math::bell_lowering_policy relaxed;
    relaxed.maximum_value_slot_expansion = 1000.0;
    relaxed.maximum_storage_expansion = 1000.0;
    math::bell_candidate_set choices;
    require(math::query_bell_candidates_host(
        data.source, data.plan, data.order, relaxed, owner.view(), &choices),
        "cand query failed");

    const std::uint32_t expected_b[3] = {8u, 16u, 32u};
    const std::uint32_t expected_f[3] = {24u, 48u, 96u};
    const std::uint64_t expected_s[3] = {256u, 768u, 3072u};
    for (std::uint32_t index = 0u; index < 3u; ++index) {
        const auto &cand = choices.candidates[index];
        require(cand.state == math::bell_candidate_state::legal,
            "relaxed cand was rejected");
        require(cand.block_size == expected_b[index]
                && cand.padded_feature_count == expected_f[index]
                && cand.metrics.dense_value_slots == expected_s[index],
            "cand geometry or expansion is wrong");

        std::vector<std::uint32_t> padded_offsets(
            cand.feature_block_offset_count);
        std::vector<std::int32_t> columns(cand.column_index_count);
        std::vector<unsigned char> values(cand.value_bytes);
        math::bell_candidate_buffers buffers{
            padded_offsets.size(), padded_offsets.data(), columns.size(),
            columns.data(), values.size(), values.data()};
        math::physical_bell_view view;
        require(math::materialize_bell_candidate_host(
            data.source, data.plan, data.order, relaxed, cand,
            owner.view(), buffers, &view),
            "cand materialization failed");
        require(view.candidate_identity == cand.candidate_identity,
            "cand identity changed during materialization");
        decode(data, view);
    }

    math::bell_candidate_set filtered;
    require(math::query_bell_candidates_host(
        data.source, data.plan, data.order, {}, owner.view(), &filtered),
        "default cand filtering failed");
    require(filtered.candidates[0].state == math::bell_candidate_state::legal
            && filtered.candidates[1].state == math::bell_candidate_state::legal
            && filtered.candidates[2].state
                == math::bell_candidate_state::value_expansion_exceeded,
        "default policy did not reject only the absurd BELL32 expansion");
    require(std::abs(filtered.candidates[0].metrics.ell_slot_utilization - 0.75)
            < 1.0e-12,
        "BELL8 ELL-slot utilization is wrong");
}

void test_failures() {
    fixture data;
    workspace owner(data);
    math::bell_lowering_policy policy;
    policy.maximum_value_slot_expansion = 1000.0;
    policy.maximum_storage_expansion = 1000.0;
    math::bell_candidate_set choices;
    require(math::query_bell_candidates_host(
        data.source, data.plan, data.order, policy, owner.view(), &choices),
        "cand query for failure checks failed");

    const auto &cand = choices.candidates[0];
    std::vector<std::uint32_t> padded(cand.feature_block_offset_count);
    std::vector<std::int32_t> columns(cand.column_index_count);
    std::vector<unsigned char> values(cand.value_bytes);
    math::bell_candidate_buffers buffers{padded.size(), padded.data(),
        columns.size(), columns.data(), values.size(), values.data()};
    data.order.ordering_identity ^= 1u;
    math::physical_bell_view view;
    const auto stale = math::materialize_bell_candidate_host(
        data.source, data.plan, data.order, policy, cand,
        owner.view(), buffers, &view);
    require(stale.code == math::bell_lowering_status_code::candidate_mismatch,
        "stale cand identity was accepted");

    data.order.ordering_identity ^= 1u;
    auto narrow = cand;
    --narrow.ell_blocks_per_row;
    const auto changed = math::materialize_bell_candidate_host(
        data.source, data.plan, data.order, policy, narrow,
        owner.view(), buffers, &view);
    require(changed.code == math::bell_lowering_status_code::candidate_mismatch,
        "modified cand geometry was accepted");

    const std::uint32_t saved = data.feature_ids[1];
    data.feature_ids[1] = data.feature_ids[0];
    const auto invalid = math::query_bell_candidates_host(
        data.source, data.plan, data.order, policy, owner.view(), &choices);
    data.feature_ids[1] = saved;
    require(invalid.code == math::bell_lowering_status_code::invalid_source,
        "duplicate CSR feature was accepted");
}

} // namespace

int main() {
    test_candidates();
    test_failures();
    return 0;
}
