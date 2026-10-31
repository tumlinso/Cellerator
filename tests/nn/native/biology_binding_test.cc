#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Cellerator/biology/feature_binding.hh>

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

struct FeatureKey {
    int value;
    friend bool operator==(FeatureKey lhs, FeatureKey rhs) noexcept {
        return lhs.value == rhs.value;
    }
};

int main() {
    using cellerator::biology::FeatureBinding;
    using cellerator::biology::FeatureBindingOptions;
    using cellerator::biology::OccurrenceCoordinates;

    FeatureBindingOptions options;
    options.domain = "gene";
    options.canonical_indices = {2, 8};
    options.model_capacity = 12;
    auto genes = FeatureBinding<std::string>::from_ids(
        {"gene-A", "gene-B"}, {9, 3}, options);

    // Feature identity, canonical vocabulary index, model row, event order,
    // and physical position are independent coordinates.
    auto events = OccurrenceCoordinates<std::string>::from_ids(
        {"gene-A", "gene-B", "gene-A"}, {0, 3}, {}, 5, {100, 104, 110});
    auto mapped = genes.map_occurrences(events);
    assert((mapped.canonical_ids() == std::vector<std::string>{
        "gene-A", "gene-B", "gene-A"}));
    assert((mapped.source_positions() == std::vector<std::int64_t>{0, 1, 2}));
    assert((mapped.canonical_indices()
        == std::vector<std::optional<std::int64_t>>{2, 8, 2}));
    assert((mapped.model_indices() == std::vector<std::int64_t>{9, 3, 9}));
    assert((mapped.physical_positions() == std::vector<std::int64_t>{100, 104, 110}));
    assert(mapped.has_physical_positions());
    genes.validate_alignment(mapped);

    // Target columns are source positions, never the token/embedding IDs.
    auto output_axis = OccurrenceCoordinates<std::string>::from_ids(
        {"gene-B", "gene-A"}, {0, 2}, {}, 5, {20, 10});
    auto target = genes.prepare(output_axis, {"gene-A", "gene-B"});
    assert((target.canonical_ids() == std::vector<std::string>{"gene-A", "gene-B"}));
    assert((target.source_positions() == std::vector<std::int64_t>{1, 0}));
    assert(target.output_columns() == target.source_positions());
    assert((target.model_indices() == std::vector<std::int64_t>{9, 3}));
    assert((target.physical_positions() == std::vector<std::int64_t>{10, 20}));

    // Gather indices are explicit; no values or sparse representation are
    // retained by the prepared alignment.
    const std::vector<int> source_values{5, 7};
    std::vector<int> gathered;
    for (const auto position : target.gather_positions()) {
        gathered.push_back(source_values.at(static_cast<std::size_t>(position)));
    }
    assert((gathered == std::vector<int>{7, 5}));

    auto drop_options = FeatureBindingOptions{};
    drop_options.unknown_policy = cellerator::biology::UnknownIdentityPolicy::drop;
    auto drop_binding = FeatureBinding<int>::from_ids({11, 19}, {4, 8}, drop_options);
    const auto dropped = drop_binding.map_occurrences(std::vector<int>{11, 42, 19});
    assert((dropped.canonical_ids() == std::vector<int>{11, 19}));
    assert((dropped.dropped_ids() == std::vector<int>{42}));
    assert((dropped.source_positions() == std::vector<std::int64_t>{0, 2}));
    assert((dropped.model_indices() == std::vector<std::int64_t>{4, 8}));

    FeatureBindingOptions map_options;
    map_options.unknown_policy = cellerator::biology::UnknownIdentityPolicy::map_to_token;
    map_options.unknown_model_index = 1;
    auto typed = FeatureBinding<FeatureKey>::from_ids(
        {{1}, {2}}, {6, 7}, map_options);
    const auto mapped_unknown = typed.map_occurrences(std::vector<FeatureKey>{{3}});
    assert(mapped_unknown.canonical_ids().size() == 1);
    assert(mapped_unknown.canonical_indices().front() == std::nullopt);
    assert(mapped_unknown.model_indices().front() == 1);

    auto incompatible = FeatureBinding<std::string>::from_ids(
        {"gene-A", "gene-B"}, {9, 3}, options);
    bool rejected = false;
    try {
        incompatible.validate_alignment(target);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    auto descriptor = genes.to_descriptor(
        "identity-test", 1, [](const std::string& id) { return id; });
    assert(descriptor.domain == "gene");
    assert((descriptor.model_indices == std::vector<std::int64_t>{9, 3}));
    assert((descriptor.canonical_indices
        == std::vector<std::optional<std::int64_t>>{2, 8}));
}
