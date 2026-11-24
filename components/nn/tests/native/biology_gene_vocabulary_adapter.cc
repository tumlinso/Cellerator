#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Cellerator/biology/gene_feature_binding.hh>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using cellerator::biology::FeatureBinding;
    using cellerator::biology::from_gene_vocabulary;
    using cellerator::tokenization::GeneRecord;
    using cellerator::tokenization::GeneVocabulary;
    using cellerator::tokenization::GeneVocabularyMetadata;
    using cellerator::tokenization::unknown_identity_policy;

    auto vocabulary = GeneVocabulary::from_records(
        {GeneRecord{"ENSG-Z", "ZSYM", {"z-alias"}, 9},
         GeneRecord{"ENSG-A", "ASYM", {"a-alias", "ENSG-Z"}, 3}},
        GeneVocabularyMetadata{"fixture", "test-organism", "assembly-1", "release-2"},
        unknown_identity_policy::map_to_token, 1);
    auto binding = from_gene_vocabulary(vocabulary, {}, 12);

    // Canonical IDs follow GeneVocabulary order; model indices remain its
    // actual, noncontiguous token rows.
    assert((binding.canonical_ids() == std::vector<std::string>{"ENSG-A", "ENSG-Z"}));
    assert((binding.canonical_indices()
        == std::vector<std::optional<std::int64_t>>{0, 1}));
    assert((binding.model_indices() == std::vector<std::int64_t>{3, 9}));
    assert(binding.metadata().at("organism") == "test-organism");

    const auto alias = binding.resolve("z-alias");
    assert(alias);
    assert(alias->canonical_id == "ENSG-Z");
    assert(alias->canonical_index == 1);
    assert(alias->model_index == 9);
    const auto symbol = binding.resolve("ASYM");
    assert(symbol && symbol->canonical_id == "ENSG-A");
    const auto canonical_precedence = binding.resolve("ENSG-Z");
    assert(canonical_precedence && canonical_precedence->canonical_index == 1);
    assert(canonical_precedence->model_index == 9);
    const auto unknown = binding.resolve("not-in-vocabulary");
    assert(unknown && unknown->canonical_id == "not-in-vocabulary");
    assert(!unknown->canonical_index && unknown->model_index == 1);

    const auto descriptor = binding.to_descriptor(
        "test-identity-wire", 1,
        [](const std::string& value) { return value; });
    assert((descriptor.identities == std::vector<std::string>{"ENSG-A", "ENSG-Z"}));
    assert(descriptor.canonical_indices.has_value());
    assert((*descriptor.canonical_indices == std::vector<std::int64_t>{0, 1}));
    assert(descriptor.aliases.size() == 2);
    assert(descriptor.aliases[0].identity == "ENSG-A");
    assert((descriptor.aliases[0].aliases
        == std::vector<std::string>{"ENSG-Z", "a-alias", "ASYM"}));
    assert(descriptor.unknown_policy == cellerator::biology::UnknownIdentityPolicy::map_to_token);
    assert(descriptor.unknown_model_index == 1);

    const auto default_indices = FeatureBinding<std::string>::from_ids(
        {"x", "y"}).to_descriptor(
            "test-identity-wire", 1,
            [](const std::string& value) { return value; });
    assert(!default_indices.canonical_indices);

    auto ambiguous_vocabulary = GeneVocabulary::from_records(
        {GeneRecord{"ENSG-1", "SYM-1", {"shared"}, 0},
         GeneRecord{"ENSG-2", "SYM-2", {"shared"}, 2}});
    const auto ambiguous_binding = from_gene_vocabulary(ambiguous_vocabulary);
    bool rejected = false;
    try {
        (void)ambiguous_binding.resolve("shared");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}
