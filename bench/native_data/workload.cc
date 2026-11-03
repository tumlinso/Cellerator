#include "workload.hh"

#include <c10/util/Half.h>

#include <cstring>
#include <stdexcept>

namespace cellerator::bench::native_data {
namespace {

struct Hash64 {
    std::uint64_t value = 14695981039346656037ULL;

    void add_byte(std::uint8_t byte) noexcept {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    void add_u64(std::uint64_t input) noexcept {
        for (unsigned shift = 0; shift < 64; shift += 8)
            add_byte(static_cast<std::uint8_t>(input >> shift));
    }
    void add_float(float input) noexcept {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &input, sizeof(bits));
        for (unsigned shift = 0; shift < 32; shift += 8)
            add_byte(static_cast<std::uint8_t>(bits >> shift));
    }
    template<class T, class Add>
    void add_vector(const std::vector<T>& input, Add add) noexcept {
        add_u64(input.size());
        for (const auto& value : input) add(value);
    }
};

std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

float bounded_value(std::uint64_t& state) noexcept {
    // Use the high 24 bits directly so the fixture is stable across standard
    // library implementations; values are exactly reproducible in [0.98,1.02).
    const auto mantissa = static_cast<std::uint32_t>(splitmix64(state) >> 40);
    const float unit = static_cast<float>(mantissa) * (1.0f / 16777216.0f);
    return 0.98f + 0.04f * unit;
}

Workload make_block_workload(std::string_view name, std::size_t groups,
                             std::size_t stride, std::size_t active_conditions,
                             std::size_t active_group_target,
                             bool paired_outputs) {
    constexpr std::size_t block_width = 32;
    constexpr std::size_t conditions_and_parameters = 64;
    Workload result;
    result.name = name;
    result.coordinates = (groups - 1) * stride + block_width;
    result.conditions = conditions_and_parameters;
    result.parameters = conditions_and_parameters;
    result.baseline_scale = -0.25f;
    result.x.resize(result.coordinates);
    result.z.resize(result.conditions * 2);
    result.theta.resize(result.parameters);
    result.phi.assign(result.parameters, 0.5f);
    result.tau.assign(result.parameters, 0.6f);
    result.upstream.resize(result.coordinates);

    std::uint64_t random_state = 0x6e61746976656461ULL;
    for (float& value : result.x) value = bounded_value(random_state);
    for (std::size_t condition = 0; condition < result.conditions; ++condition) {
        result.z[condition * 2] = condition < active_conditions ? 0.5f : 0.0f;
        result.z[condition * 2 + 1] = 0.5f;
    }
    for (std::size_t parameter = 0; parameter < result.parameters; ++parameter)
        result.theta[parameter] = 0.05f + 0.001f * static_cast<float>(parameter % 7);
    for (std::size_t coordinate = 0; coordinate < result.coordinates; ++coordinate)
        result.upstream[coordinate] = ((coordinate * 17 + 3) % 7 < 3) ? -0.5f : 0.5f;

    if (active_group_target > groups || active_conditions > result.conditions ||
        (active_conditions == 0 && active_group_target != 0) ||
        (active_conditions == result.conditions && active_group_target != groups))
        throw std::invalid_argument("inconsistent native-data activity target");
    std::vector<std::size_t> condition_frequency(result.conditions, 0);
    const auto distribute = [&](std::size_t first, std::size_t count, std::size_t total) {
        if (!count) return;
        const auto quotient = total / count;
        const auto remainder = total % count;
        for (std::size_t index = 0; index < count; ++index)
            condition_frequency[first + index] = quotient + (index < remainder ? 1 : 0);
    };
    distribute(0, active_conditions, active_group_target);
    distribute(active_conditions, result.conditions - active_conditions, groups - active_group_target);
    std::vector<std::uint64_t> condition_for_group;
    condition_for_group.reserve(groups);
    for (std::size_t condition = 0; condition < condition_frequency.size(); ++condition)
        for (std::size_t use = 0; use < condition_frequency[condition]; ++use)
            condition_for_group.push_back(condition);
    if (condition_for_group.size() != groups)
        throw std::logic_error("condition frequency did not cover all groups");

    result.argument_begin.reserve(groups);
    result.argument_count.assign(groups, block_width);
    result.output_begin.reserve(groups);
    result.output_count.assign(groups, 1);
    result.parameter_index.reserve(groups);
    result.condition_index.reserve(groups);
    result.group_owner.reserve(groups);
    result.argument_coordinate.reserve(groups * block_width);
    result.output_coordinate.reserve(groups);
    result.output_scale.reserve(groups);
    for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t begin = group * stride;
        result.argument_begin.push_back(result.argument_coordinate.size());
        for (std::size_t slot = 0; slot < block_width; ++slot)
            result.argument_coordinate.push_back(begin + slot);
        result.output_begin.push_back(result.output_coordinate.size());
        const std::size_t output = paired_outputs ? (group / 2) * block_width : begin;
        result.output_coordinate.push_back(output);
        result.output_scale.push_back(paired_outputs && group % 2 ? 0.5f : 1.0f);
        result.parameter_index.push_back(group % result.parameters);
        result.condition_index.push_back(condition_for_group[group]);
        result.group_owner.push_back(static_cast<int>(group % 2));
    }
    return result;
}

void validate(const Workload& value) {
    const auto group_count = value.groups();
    if (value.batch != 1 || !value.coordinates || !value.conditions || !value.parameters)
        throw std::invalid_argument("native-data benchmark currently requires one non-empty logical sample");
    if (value.x.size() != value.batch * value.coordinates ||
        value.upstream.size() != value.batch * value.coordinates ||
        value.z.size() != value.batch * value.conditions * 2 ||
        value.theta.size() != value.parameters || value.phi.size() != value.parameters ||
        value.tau.size() != value.parameters || value.argument_count.size() != group_count ||
        value.output_begin.size() != group_count || value.output_count.size() != group_count ||
        value.parameter_index.size() != group_count || value.condition_index.size() != group_count ||
        value.group_owner.size() != group_count ||
        value.output_coordinate.size() != value.output_scale.size())
        throw std::invalid_argument("inconsistent native-data benchmark workload extents");
    if (value.argument_begin.empty() || value.argument_begin.front() != 0 ||
        value.argument_begin.back() + value.argument_count.back() != value.argument_coordinate.size() ||
        value.output_begin.empty() || value.output_begin.front() != 0 ||
        value.output_begin.back() + value.output_count.back() != value.output_coordinate.size())
        throw std::invalid_argument("native-data benchmark incidence ranges are not contiguous");
    for (std::size_t group = 0; group < group_count; ++group) {
        if (value.argument_begin[group] + value.argument_count[group] > value.argument_coordinate.size() ||
            value.output_begin[group] + value.output_count[group] > value.output_coordinate.size() ||
            value.parameter_index[group] >= value.parameters ||
            value.condition_index[group] >= value.conditions || value.group_owner[group] < 0 ||
            value.group_owner[group] > 1)
            throw std::invalid_argument("invalid native-data benchmark group metadata");
    }
    for (const auto coordinate : value.argument_coordinate)
        if (coordinate >= value.coordinates) throw std::invalid_argument("argument coordinate out of range");
    for (const auto coordinate : value.output_coordinate)
        if (coordinate >= value.coordinates) throw std::invalid_argument("output coordinate out of range");
}

} // namespace

std::uint64_t Workload::identity_hash() const noexcept {
    Hash64 hash;
    for (const auto value : name) hash.add_byte(static_cast<std::uint8_t>(value));
    hash.add_u64(batch);
    hash.add_u64(coordinates);
    hash.add_u64(conditions);
    hash.add_u64(parameters);
    hash.add_float(baseline_scale);
    hash.add_vector(x, [&](float value) { hash.add_float(value); });
    hash.add_vector(z, [&](float value) { hash.add_float(value); });
    hash.add_vector(theta, [&](float value) { hash.add_float(value); });
    hash.add_vector(phi, [&](float value) { hash.add_float(value); });
    hash.add_vector(tau, [&](float value) { hash.add_float(value); });
    hash.add_vector(upstream, [&](float value) { hash.add_float(value); });
    hash.add_vector(argument_begin, [&](std::uint64_t value) { hash.add_u64(value); });
    hash.add_vector(argument_count, [&](std::uint64_t value) { hash.add_u64(value); });
    hash.add_vector(output_begin, [&](std::uint64_t value) { hash.add_u64(value); });
    hash.add_vector(output_count, [&](std::uint64_t value) { hash.add_u64(value); });
    hash.add_vector(parameter_index, [&](std::uint64_t value) { hash.add_u64(value); });
    hash.add_vector(condition_index, [&](std::uint64_t value) { hash.add_u64(value); });
    hash.add_vector(argument_coordinate, [&](std::uint64_t value) { hash.add_u64(value); });
    hash.add_vector(output_coordinate, [&](std::uint64_t value) { hash.add_u64(value); });
    hash.add_vector(output_scale, [&](float value) { hash.add_float(value); });
    hash.add_vector(group_owner, [&](int value) { hash.add_u64(static_cast<std::uint64_t>(value)); });
    return hash.value;
}

std::uint64_t Workload::quantized_x_hash() const noexcept {
    Hash64 hash;
    hash.add_u64(x.size());
    for (const float value : x) {
        const c10::Half quantized(value);
        std::uint16_t bits = 0;
        std::memcpy(&bits, &quantized, sizeof(bits));
        hash.add_byte(static_cast<std::uint8_t>(bits));
        hash.add_byte(static_cast<std::uint8_t>(bits >> 8));
    }
    return hash.value;
}

Workload make_low_activity_disjoint() {
    auto result = make_block_workload("low_activity_disjoint_32", 512, 32, 6, 51, false);
    validate(result);
    return result;
}

Workload make_high_activity_overlap() {
    auto result = make_block_workload("high_activity_overlap_32", 1024, 16, 58, 922, true);
    validate(result);
    return result;
}

} // namespace cellerator::bench::native_data
