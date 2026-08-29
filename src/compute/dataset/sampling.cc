#include <Cellerator/compute/sampling.hh>

#include <algorithm>
#include <numeric>
#include <queue>
#include <string_view>
#include <utility>

namespace cellerator::compute::sampling {

namespace {

constexpr std::uint64_t row_identity_domain = 0x726f775f69645f31ull;
constexpr std::uint64_t cell_id_domain = 0x63656c6c5f696431ull;
constexpr std::uint64_t fnv1a_offset = 14695981039346656037ull;
constexpr std::uint64_t fnv1a_prime = 1099511628211ull;
constexpr std::uint64_t max_quantile_denominator = 0xffffffffull;

struct candidate {
    std::uint64_t row = 0u;
    std::uint64_t hash = 0u;
    std::string_view stable_id;
    std::uint32_t stratum = 0u;
};

void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
}

bool valid_boundary(const quantile_boundary &value) {
    return value.denominator != 0u
        && value.denominator <= max_quantile_denominator
        && value.numerator <= value.denominator;
}

int compare_boundary(const quantile_boundary &left, const quantile_boundary &right) {
    const __uint128_t lhs = (__uint128_t) left.numerator * right.denominator;
    const __uint128_t rhs = (__uint128_t) right.numerator * left.denominator;
    return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

bool hash_at_or_above(std::uint64_t hash, const quantile_boundary &boundary) {
    const __uint128_t lhs = (__uint128_t) hash * boundary.denominator;
    const __uint128_t rhs = (__uint128_t) boundary.numerator << 64u;
    return lhs >= rhs;
}

bool hash_below(std::uint64_t hash, const quantile_boundary &boundary) {
    if (boundary.numerator == boundary.denominator) return true;
    const __uint128_t lhs = (__uint128_t) hash * boundary.denominator;
    const __uint128_t rhs = (__uint128_t) boundary.numerator << 64u;
    return lhs < rhs;
}

bool validate_spec(const sample_spec &spec, std::string *error) {
    if (spec.split_name.empty()) {
        set_error(error, "sample split_name must be non-empty");
        return false;
    }
    if (spec.mode == selection_mode::hash_quantile_range) {
        if (!valid_boundary(spec.quantile.begin) || !valid_boundary(spec.quantile.end)) {
            set_error(error, "quantile boundaries require numerator <= denominator <= 2^32-1");
            return false;
        }
        if (compare_boundary(spec.quantile.begin, spec.quantile.end) >= 0) {
            set_error(error, "quantile range must be non-empty and increasing");
            return false;
        }
        return true;
    }
    if (spec.mode == selection_mode::exact_lowest_hash) {
        if (spec.requested_row_count > maximum_exact_sample_rows) {
            set_error(error, "requested exact row count exceeds maximum_exact_sample_rows");
            return false;
        }
        return true;
    }
    set_error(error, "unknown sample selection mode");
    return false;
}

bool validate_identities(std::uint64_t total_rows,
                         const cell_identity_view &identities,
                         std::string *error) {
    if (identities.kind == cell_identity_kind::global_row_index) return true;
    if (identities.kind != cell_identity_kind::stable_item_id) {
        set_error(error, "unknown cell identity kind");
        return false;
    }
    if (identities.count != total_rows || (total_rows != 0u && identities.stable_cell_ids == nullptr)) {
        set_error(error, "stable item ID count does not match total rows");
        return false;
    }
    std::vector<std::string_view> sorted_ids;
    sorted_ids.reserve((std::size_t) total_rows);
    for (std::uint64_t row = 0u; row < total_rows; ++row) {
        const char *id = identities.stable_cell_ids[(std::size_t) row];
        if (id == nullptr || *id == '\0') {
            set_error(error, "stable item IDs must be non-empty");
            return false;
        }
        sorted_ids.emplace_back(id);
    }
    std::sort(sorted_ids.begin(), sorted_ids.end());
    if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) != sorted_ids.end()) {
        set_error(error, "stable item IDs must be unique");
        return false;
    }
    return true;
}

std::uint64_t identity_hash(std::uint64_t row,
                            const cell_identity_view &identities,
                            std::string_view *stable_id,
                            std::uint64_t seed) {
    if (identities.kind == cell_identity_kind::global_row_index) {
        *stable_id = std::string_view{};
        return hash_global_row_index(row, seed);
    }
    *stable_id = identities.stable_cell_ids[(std::size_t) row];
    return hash_stable_cell_id(stable_id->data(), stable_id->size(), seed);
}

bool candidate_hash_order(const candidate &left, const candidate &right) {
    if (left.hash != right.hash) return left.hash < right.hash;
    if (!left.stable_id.empty() || !right.stable_id.empty()) {
        const std::size_t shared = std::min(left.stable_id.size(), right.stable_id.size());
        for (std::size_t i = 0u; i < shared; ++i) {
            const std::uint8_t lhs = (std::uint8_t) left.stable_id[i];
            const std::uint8_t rhs = (std::uint8_t) right.stable_id[i];
            if (lhs != rhs) return lhs < rhs;
        }
        if (left.stable_id.size() != right.stable_id.size()) {
            return left.stable_id.size() < right.stable_id.size();
        }
    }
    return left.row < right.row;
}

sample_spec spec_from_provenance(const sample_provenance &provenance) {
    sample_spec spec;
    spec.mode = provenance.mode;
    spec.seed = provenance.seed;
    spec.split_name = provenance.split_name;
    spec.quantile = provenance.quantile;
    spec.requested_row_count = provenance.requested_row_count;
    return spec;
}

bool validate_density_spec(std::uint64_t total_rows,
                           const row_nnz_view &row_nnz,
                           const density_sample_spec &spec,
                           std::string *error) {
    if (total_rows == 0u) {
        set_error(error, "density sampling requires at least one row");
        return false;
    }
    if (row_nnz.count != total_rows || row_nnz.values == nullptr) {
        set_error(error, "row nnz count does not match total rows");
        return false;
    }
    if (spec.split_name.empty()) {
        set_error(error, "density sample split_name must be non-empty");
        return false;
    }
    if (spec.requested_strata == 0u) {
        set_error(error, "density sampling requires at least one requested stratum");
        return false;
    }
    if (spec.requested_strata > total_rows) {
        set_error(error, "requested density strata exceeds total rows");
        return false;
    }
    if (spec.requested_row_count == 0u || spec.requested_row_count > total_rows) {
        set_error(error, "density requested row count must be in [1, total_rows]");
        return false;
    }
    return true;
}

std::vector<std::uint64_t> build_density_boundaries(const row_nnz_view &row_nnz,
                                                    std::uint32_t requested_strata) {
    std::vector<std::uint64_t> sorted(row_nnz.values, row_nnz.values + row_nnz.count);
    std::vector<std::uint64_t> unique_values, cumulative_counts;
    std::vector<std::uint64_t> boundaries;
    std::sort(sorted.begin(), sorted.end());
    for (std::uint64_t value : sorted) {
        if (unique_values.empty() || unique_values.back() != value) {
            unique_values.push_back(value);
            cumulative_counts.push_back(1u);
        } else {
            ++cumulative_counts.back();
        }
    }
    for (std::size_t i = 1u; i < cumulative_counts.size(); ++i) {
        cumulative_counts[i] += cumulative_counts[i - 1u];
    }
    const std::uint32_t actual_strata = (std::uint32_t) std::min<std::size_t>(requested_strata, unique_values.size());
    boundaries.reserve(actual_strata);
    std::size_t previous = (std::size_t) -1;
    for (std::uint32_t cut = 1u; cut < actual_strata; ++cut) {
        const std::size_t min_index = previous + 1u;
        const std::size_t max_index = unique_values.size() - (actual_strata - cut) - 1u;
        const __uint128_t target = (__uint128_t) cut * sorted.size();
        auto begin = cumulative_counts.begin() + (std::ptrdiff_t) min_index;
        auto end = cumulative_counts.begin() + (std::ptrdiff_t) max_index + 1;
        auto at_or_above = std::lower_bound(begin, end, target, [actual_strata](std::uint64_t count, __uint128_t value) {
            return (__uint128_t) count * actual_strata < value;
        });
        std::size_t chosen = at_or_above == end ? max_index : (std::size_t) (at_or_above - cumulative_counts.begin());
        if (chosen > min_index) {
            const std::size_t lower = chosen - 1u;
            const __uint128_t lower_scaled = (__uint128_t) cumulative_counts[lower] * actual_strata;
            const __uint128_t upper_scaled = (__uint128_t) cumulative_counts[chosen] * actual_strata;
            const __uint128_t lower_distance = target > lower_scaled ? target - lower_scaled : lower_scaled - target;
            const __uint128_t upper_distance = target > upper_scaled ? target - upper_scaled : upper_scaled - target;
            if (lower_distance <= upper_distance) chosen = lower;
        }
        boundaries.push_back(unique_values[chosen]);
        previous = chosen;
    }
    boundaries.push_back(unique_values.back());
    return boundaries;
}

std::vector<std::uint64_t> allocate_density_samples(const std::vector<std::uint64_t> &stratum_counts,
                                                    std::uint64_t requested_rows) {
    const std::uint64_t strata = (std::uint64_t) stratum_counts.size();
    std::vector<std::uint64_t> sampled(stratum_counts.size(), 1u);
    const std::uint64_t remaining = requested_rows - strata;
    const std::uint64_t capacity_total = std::accumulate(
        stratum_counts.begin(), stratum_counts.end(), 0ull) - strata;
    if (remaining == 0u || capacity_total == 0u) return sampled;

    struct remainder_item {
        std::uint64_t remainder = 0u;
        std::uint32_t stratum = 0u;
    };
    std::vector<remainder_item> remainders;
    std::uint64_t assigned = 0u;
    remainders.reserve(stratum_counts.size());
    for (std::uint32_t stratum = 0u; stratum < stratum_counts.size(); ++stratum) {
        const std::uint64_t capacity = stratum_counts[stratum] - 1u;
        const __uint128_t scaled = (__uint128_t) remaining * capacity;
        const std::uint64_t add = (std::uint64_t) (scaled / capacity_total);
        sampled[stratum] += add;
        assigned += add;
        remainders.push_back({(std::uint64_t) (scaled % capacity_total), stratum});
    }
    std::sort(remainders.begin(), remainders.end(), [](const remainder_item &left, const remainder_item &right) {
        return left.remainder != right.remainder
            ? left.remainder > right.remainder
            : left.stratum < right.stratum;
    });
    std::uint64_t leftover = remaining - assigned;
    for (const remainder_item &item : remainders) {
        if (leftover == 0u) break;
        if (sampled[item.stratum] < stratum_counts[item.stratum]) {
            ++sampled[item.stratum];
            --leftover;
        }
    }
    return sampled;
}

} // namespace

std::uint64_t splitmix64_hash(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

std::uint64_t hash_global_row_index(std::uint64_t global_row_index, std::uint64_t seed) noexcept {
    return splitmix64_hash(global_row_index ^ splitmix64_hash(seed ^ row_identity_domain));
}

std::uint64_t hash_stable_cell_id(const char *cell_id, std::size_t length, std::uint64_t seed) noexcept {
    std::uint64_t folded = fnv1a_offset;
    for (std::size_t i = 0; i < length; ++i) {
        folded ^= (std::uint8_t) cell_id[i];
        folded *= fnv1a_prime;
    }
    folded ^= splitmix64_hash((std::uint64_t) length ^ cell_id_domain);
    return splitmix64_hash(folded ^ splitmix64_hash(seed ^ cell_id_domain));
}

bool build_sample_plan(std::uint64_t total_rows,
                       const sample_spec &spec,
                       const cell_identity_view &identities,
                       sample_plan *out,
                       std::string *error) {
    std::vector<candidate> candidates;
    const std::uint64_t exact_row_count = spec.mode == selection_mode::exact_lowest_hash
        ? std::min(total_rows, spec.requested_row_count)
        : 0u;
    if (out == nullptr) {
        set_error(error, "output sample plan is null");
        return false;
    }
    *out = sample_plan{};
    if (!validate_spec(spec, error) || !validate_identities(total_rows, identities, error)) return false;

    if (spec.mode == selection_mode::hash_quantile_range) {
        candidates.reserve((std::size_t) std::min<std::uint64_t>(total_rows, 4096u));
    }
    std::vector<candidate> heap_storage;
    if (spec.mode == selection_mode::exact_lowest_hash) {
        heap_storage.reserve((std::size_t) exact_row_count);
    }
    std::priority_queue<candidate, std::vector<candidate>, decltype(&candidate_hash_order)> lowest_hashes(
        &candidate_hash_order,
        std::move(heap_storage));
    for (std::uint64_t row = 0u; row < total_rows; ++row) {
        std::string_view stable_id;
        const std::uint64_t hash = identity_hash(row, identities, &stable_id, spec.seed);
        if (spec.mode == selection_mode::exact_lowest_hash) {
            const candidate item{row, hash, stable_id};
            if (lowest_hashes.size() < (std::size_t) exact_row_count) {
                lowest_hashes.push(item);
            } else if (!lowest_hashes.empty() && candidate_hash_order(item, lowest_hashes.top())) {
                lowest_hashes.pop();
                lowest_hashes.push(item);
            }
            continue;
        }
        if (!hash_at_or_above(hash, spec.quantile.begin) || !hash_below(hash, spec.quantile.end)) {
            continue;
        }
        candidates.push_back(candidate{row, hash, stable_id});
    }
    if (spec.mode == selection_mode::exact_lowest_hash) {
        candidates.reserve(lowest_hashes.size());
        while (!lowest_hashes.empty()) {
            candidates.push_back(lowest_hashes.top());
            lowest_hashes.pop();
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const candidate &left, const candidate &right) {
        return left.row < right.row;
    });

    out->global_row_indices.reserve(candidates.size());
    out->identity_hashes.reserve(candidates.size());
    for (const candidate &item : candidates) {
        out->global_row_indices.push_back(item.row);
        out->identity_hashes.push_back(item.hash);
    }
    out->provenance.seed = spec.seed;
    out->provenance.hash_algorithm = splitmix64_algorithm_name;
    out->provenance.hash_version = splitmix64_algorithm_version;
    out->provenance.total_rows = total_rows;
    out->provenance.selected_rows = (std::uint64_t) candidates.size();
    out->provenance.mode = spec.mode;
    out->provenance.split_name = spec.split_name;
    out->provenance.cell_identity = identities.kind;
    out->provenance.quantile = spec.quantile;
    out->provenance.requested_row_count = spec.requested_row_count;
    return true;
}

bool reproduce_sample_plan(const sample_provenance &provenance,
                           const cell_identity_view &identities,
                           sample_plan *out,
                           std::string *error) {
    if (provenance.hash_algorithm != splitmix64_algorithm_name
        || provenance.hash_version != splitmix64_algorithm_version) {
        set_error(error, "unsupported sample provenance hash algorithm or version");
        return false;
    }
    if (provenance.cell_identity != identities.kind) {
        set_error(error, "sample provenance cell identity does not match supplied identities");
        return false;
    }
    if (!build_sample_plan(provenance.total_rows, spec_from_provenance(provenance), identities, out, error)) {
        return false;
    }
    if (out->provenance.selected_rows != provenance.selected_rows) {
        set_error(error, "reproduced sample row count differs from provenance");
        *out = sample_plan{};
        return false;
    }
    return true;
}

bool build_density_sample_plan(std::uint64_t total_rows,
                               const row_nnz_view &row_nnz,
                               const density_sample_spec &spec,
                               const cell_identity_view &identities,
                               sample_plan *out,
                               std::string *error) {
    using candidate_heap = std::priority_queue<
        candidate,
        std::vector<candidate>,
        decltype(&candidate_hash_order)>;
    std::vector<std::uint64_t> boundaries, stratum_counts, sampled_counts;
    std::vector<candidate_heap> heaps;
    std::vector<candidate> selected;
    if (out == nullptr) {
        set_error(error, "output density sample plan is null");
        return false;
    }
    *out = sample_plan{};
    if (!validate_density_spec(total_rows, row_nnz, spec, error)
        || !validate_identities(total_rows, identities, error)) {
        return false;
    }

    boundaries = build_density_boundaries(row_nnz, spec.requested_strata);
    stratum_counts.assign(boundaries.size(), 0u);
    for (std::uint64_t row = 0u; row < total_rows; ++row) {
        const auto boundary = std::lower_bound(boundaries.begin(), boundaries.end(), row_nnz.values[row]);
        if (boundary == boundaries.end()) {
            set_error(error, "row nnz did not resolve to a density stratum");
            return false;
        }
        ++stratum_counts[(std::size_t) (boundary - boundaries.begin())];
    }
    if (spec.requested_row_count < stratum_counts.size()) {
        set_error(error, "requested row count must sample every non-empty density stratum");
        return false;
    }
    sampled_counts = allocate_density_samples(stratum_counts, spec.requested_row_count);
    heaps.reserve(stratum_counts.size());
    for (std::size_t stratum = 0u; stratum < stratum_counts.size(); ++stratum) {
        std::vector<candidate> storage;
        storage.reserve((std::size_t) sampled_counts[stratum]);
        heaps.emplace_back(&candidate_hash_order, std::move(storage));
    }

    for (std::uint64_t row = 0u; row < total_rows; ++row) {
        const std::uint32_t stratum = (std::uint32_t) (
            std::lower_bound(boundaries.begin(), boundaries.end(), row_nnz.values[row]) - boundaries.begin());
        std::string_view stable_id;
        const candidate item{row, identity_hash(row, identities, &stable_id, spec.seed), stable_id, stratum};
        candidate_heap &heap = heaps[stratum];
        if (heap.size() < (std::size_t) sampled_counts[stratum]) {
            heap.push(item);
        } else if (candidate_hash_order(item, heap.top())) {
            heap.pop();
            heap.push(item);
        }
    }

    selected.reserve((std::size_t) spec.requested_row_count);
    for (candidate_heap &heap : heaps) {
        while (!heap.empty()) {
            selected.push_back(heap.top());
            heap.pop();
        }
    }
    std::sort(selected.begin(), selected.end(), [](const candidate &left, const candidate &right) {
        return left.row < right.row;
    });
    if (selected.size() != (std::size_t) spec.requested_row_count) {
        set_error(error, "density allocation did not produce the exact requested row count");
        return false;
    }

    out->global_row_indices.reserve(selected.size());
    out->identity_hashes.reserve(selected.size());
    out->row_strata.reserve(selected.size());
    out->sampling_weights.reserve(selected.size());
    for (const candidate &item : selected) {
        out->global_row_indices.push_back(item.row);
        out->identity_hashes.push_back(item.hash);
        out->row_strata.push_back(item.stratum);
        out->sampling_weights.push_back(
            (double) stratum_counts[item.stratum] / (double) sampled_counts[item.stratum]);
    }
    out->provenance.seed = spec.seed;
    out->provenance.hash_algorithm = splitmix64_algorithm_name;
    out->provenance.hash_version = splitmix64_algorithm_version;
    out->provenance.total_rows = total_rows;
    out->provenance.selected_rows = (std::uint64_t) selected.size();
    out->provenance.mode = selection_mode::density_quantile_exact;
    out->provenance.split_name = spec.split_name;
    out->provenance.cell_identity = identities.kind;
    out->provenance.requested_row_count = spec.requested_row_count;
    out->provenance.requested_density_strata = spec.requested_strata;
    out->provenance.density_strata = (std::uint32_t) boundaries.size();
    out->provenance.density_bin_upper_bounds_inclusive = std::move(boundaries);
    out->provenance.stratum_total_rows = std::move(stratum_counts);
    out->provenance.stratum_sampled_rows = std::move(sampled_counts);
    out->provenance.weighting_rule = inverse_stratum_weighting_rule;
    return true;
}

bool build_csr_density_sample_plan(const ::cellerator::matrix::compressed *source,
                                   const density_sample_spec &spec,
                                   const cell_identity_view &identities,
                                   sample_plan *out,
                                   std::string *error) {
    std::vector<std::uint64_t> row_nnz;
    if (source == nullptr || out == nullptr) {
        set_error(error, "CSR density sampling requires source and output");
        return false;
    }
    if (source->axis != ::cellerator::matrix::compressed_by_row) {
        set_error(error, "density sampling requires row-compressed CSR input");
        return false;
    }
    if (source->majorPtr == nullptr) {
        set_error(error, "CSR density sampling requires row pointers");
        return false;
    }
    if (source->majorPtr[0] != 0u || source->majorPtr[source->rows] != source->nnz) {
        set_error(error, "CSR row pointers do not span the declared nnz");
        return false;
    }
    row_nnz.reserve(source->rows);
    for (std::uint64_t row = 0u; row < source->rows; ++row) {
        const std::uint64_t begin = source->majorPtr[row], end = source->majorPtr[row + 1u];
        if (end < begin) {
            set_error(error, "CSR row pointers are not monotonic");
            return false;
        }
        row_nnz.push_back(end - begin);
    }
    return build_density_sample_plan(
        source->rows,
        {row_nnz.data(), (std::uint64_t) row_nnz.size()},
        spec,
        identities,
        out,
        error);
}

bool reproduce_density_sample_plan(const sample_provenance &provenance,
                                   const row_nnz_view &row_nnz,
                                   const cell_identity_view &identities,
                                   sample_plan *out,
                                   std::string *error) {
    density_sample_spec spec;
    if (provenance.mode != selection_mode::density_quantile_exact
        || provenance.hash_algorithm != splitmix64_algorithm_name
        || provenance.hash_version != splitmix64_algorithm_version
        || provenance.weighting_rule != inverse_stratum_weighting_rule) {
        set_error(error, "unsupported density sample provenance contract");
        return false;
    }
    if (provenance.cell_identity != identities.kind) {
        set_error(error, "density provenance cell identity does not match supplied identities");
        return false;
    }
    spec.seed = provenance.seed;
    spec.split_name = provenance.split_name;
    spec.requested_strata = provenance.requested_density_strata;
    spec.requested_row_count = provenance.requested_row_count;
    if (!build_density_sample_plan(provenance.total_rows, row_nnz, spec, identities, out, error)) return false;
    if (out->provenance.selected_rows != provenance.selected_rows
        || out->provenance.density_strata != provenance.density_strata
        || out->provenance.density_bin_upper_bounds_inclusive != provenance.density_bin_upper_bounds_inclusive
        || out->provenance.stratum_total_rows != provenance.stratum_total_rows
        || out->provenance.stratum_sampled_rows != provenance.stratum_sampled_rows) {
        set_error(error, "density sample cannot be reproduced from provenance and row nnz");
        *out = sample_plan{};
        return false;
    }
    return true;
}

} // namespace cellerator::compute::sampling
