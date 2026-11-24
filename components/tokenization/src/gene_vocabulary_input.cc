#include <Cellerator/compute/tokenization/gene_vocabulary.hh>

namespace cellerator::tokenization {

GeneVocabulary GeneVocabulary::from_catalog(
    const biology::GeneAnnotation& annotation, GeneVocabularyMetadata metadata,
    unknown_identity_policy unknown_policy, std::optional<std::int64_t> unknown_token_id) {
    return from_catalog(annotation.catalog(), std::move(metadata), unknown_policy, unknown_token_id);
}

GeneVocabulary GeneVocabulary::from_catalog(
    biology::GeneCatalog catalog, GeneVocabularyMetadata metadata,
    unknown_identity_policy unknown_policy, std::optional<std::int64_t> unknown_token_id) {
    std::vector<GeneVocabularyEntry> records;
    records.reserve(catalog.count);
    for (size_t i = 0; i < catalog.count; ++i) {
        const auto& row = catalog.records[i];
        auto& gene = records.emplace_back();
        gene.gene_id.assign(row.id.data, row.id.size);
        if (row.symbol.size) gene.symbol.assign(row.symbol.data, row.symbol.size);
        gene.aliases.reserve(row.alias_count);
        for (size_t j = 0; j < row.alias_count; ++j)
            gene.aliases.emplace_back(row.aliases[j].data, row.aliases[j].size);
    }
    return from_records(std::move(records), std::move(metadata), unknown_policy, unknown_token_id);
}

} // namespace cellerator::tokenization
