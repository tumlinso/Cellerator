#include <Cellerator/compute/tokenization/gene_tokenizer.hh>

#include <ATen/ATen.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using cellerator::tokenization::EventStreamGeneTokenizer;
using cellerator::tokenization::GeneAnnotationFormat;
using cellerator::tokenization::GeneAnnotationOptions;
using cellerator::tokenization::GeneTokenizer;
using cellerator::tokenization::GeneVocabulary;
using cellerator::tokenization::RankedFeatureOptions;
using cellerator::tokenization::RankedGeneTokenizer;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class T>
std::vector<T> values(const at::Tensor& tensor) {
    auto host = tensor.to(at::kCPU).contiguous();
    std::vector<T> result;
    result.reserve(host.numel());
    for (std::int64_t i = 0; i < host.numel(); ++i)
        result.push_back(host.data_ptr<T>()[i]);
    return result;
}

std::filesystem::path temp_path(const std::string& name) {
    return std::filesystem::temp_directory_path()
        / ("cellerator-gene-vocabulary-" + name);
}

void gtf_builds_stable_vocabulary_and_roundtrips() {
    const auto gtf = temp_path("fixture.gtf");
    const auto saved = temp_path("fixture.vocab");
    {
        std::ofstream out(gtf);
        // Realistic GENCODE details: unquoted scalar attributes and repeated
        // unrelated `tag` fields are valid and do not alter gene identity.
        out << "chr1\ttest\tgene\t4\t8\t.\t+\t.\tgene_id \"ENSG2\"; gene_name \"BETA\"; gene_synonym \"Bee\"; level 2; tag \"basic\"; tag \"appris_principal_1\";\n";
        out << "chr1\ttest\ttranscript\t4\t8\t.\t+\t.\tgene_id \"ENSG2\"; transcript_id \"tx2\";\n";
        out << "chr1\ttest\tgene\t1\t3\t.\t+\t.\tgene_id \"ENSG1\"; gene_name \"ALPHA\"; gene_synonym \"A\";\n";
        out << "chr1\ttest\tgene\t5\t9\t.\t+\t.\tgene_id \"ENSG2\"; gene_name \"BETA\"; gene_alias \"BeeTwo\";\n";
    }
    GeneAnnotationOptions options;
    options.metadata.organism = "human";
    options.metadata.assembly = "GRCh38";
    options.metadata.release = "test";
    auto vocabulary = GeneVocabulary::from_annotation(
        gtf.string(), GeneAnnotationFormat::gtf, options);
    check(vocabulary.genes().size() == 2,
          "selected GTF gene rows should merge repeated IDs and ignore transcripts");
    check(vocabulary.resolve_identity("ENSG1") == 0
              && vocabulary.resolve_identity("ALPHA") == 0
              && vocabulary.resolve_identity("BeeTwo") == 1,
          "stable identities should sort independently of annotation row order");
    check(vocabulary.metadata().organism == "human"
              && vocabulary.metadata().assembly == "GRCh38"
              && vocabulary.metadata().source == gtf.string(),
          "gene annotation provenance should be recorded");

    auto mutable_copy = vocabulary.integer_vocabulary();
    mutable_copy.token_ids.fill_(73);
    check(values<std::int64_t>(vocabulary.integer_vocabulary().token_ids)
              == std::vector<std::int64_t>({0, 1}),
          "returned tensor storage must not mutate the immutable vocabulary");
    vocabulary.save(saved.string());
    auto loaded = GeneVocabulary::load(saved.string());
    check(loaded.genes().size() == 2
              && loaded.resolve_identity("A") == 0
              && loaded.resolve_identity("ENSG2") == 1
              && loaded.metadata().release == "test",
          "versioned round-trip should preserve IDs, aliases, tokens and metadata");
    std::filesystem::remove(gtf);
    std::filesystem::remove(saved);
}

void gff3_parsing_and_explicit_filters() {
    const auto gff = temp_path("fixture.gff3");
    {
        std::ofstream out(gff);
        out << "##gff-version 3\n";
        out << "chr1\ttest\tgene\t1\t9\t.\t+\t.\tID=gene%3A1;Name=ONE;Alias=alpha%2Cbeta,gamma\n";
        out << "chr1\ttest\texon\t1\t9\t.\t+\t.\tParent=gene%3A1\n";
        out << "##FASTA\n>chr1\nACGT\n";
    }
    auto vocabulary = GeneVocabulary::from_annotation(gff.string(), GeneAnnotationFormat::gff3);
    check(vocabulary.genes().size() == 1
              && vocabulary.resolve_identity("gene:1") == 0
              && vocabulary.resolve_identity("alpha,beta") == 0
              && vocabulary.resolve_identity("gamma") == 0,
          "GFF3 IDs and percent-encoded alias separators should be preserved");

    GeneAnnotationOptions selected;
    selected.feature_type = "exon";
    bool missing_id_rejected = false;
    try {
        (void)GeneVocabulary::from_annotation(gff.string(),
            GeneAnnotationFormat::gff3, selected);
    } catch (const std::invalid_argument&) { missing_id_rejected = true; }
    check(missing_id_rejected,
          "feature_type filters must report selected rows without gene identity");

    const auto gzip_name = temp_path("fixture.gtf.gz");
    bool gzip_rejected = false;
    try {
        (void)GeneVocabulary::from_annotation(gzip_name.string(),
                                               GeneAnnotationFormat::gtf);
    } catch (const std::invalid_argument&) { gzip_rejected = true; }
    check(gzip_rejected, "compressed annotation support must fail with a clear error");
    std::filesystem::remove(gff);
}

void canonical_ids_alias_conflicts_and_custom_tokens() {
    auto vocabulary = GeneVocabulary::from_records({
        {"ID_B", "BETA", {"ID_A", "COMMON"}, 29},
        {"ID_A", "ALPHA", {"COMMON"}, 11}});
    check(vocabulary.resolve_identity("ID_A") == 0
              && vocabulary.resolve_identity("ID_B") == 1,
          "canonical identity order should be stable-ID sorted");
    check(vocabulary.resolve_identity("ID_B") == 1,
          "stable gene IDs should take precedence over colliding aliases");
    bool ambiguous = false;
    try { (void)vocabulary.resolve_identity("COMMON"); }
    catch (const std::invalid_argument&) { ambiguous = true; }
    check(ambiguous, "ambiguous aliases must be rejected on lookup");
    check(values<std::int64_t>(vocabulary.integer_vocabulary().token_ids)
              == std::vector<std::int64_t>({11, 29}),
          "caller model token assignments should survive sorting");

    bool duplicate_tokens = false;
    try {
        (void)GeneVocabulary::from_records({
            {"ID_C", "GAMMA", {}, 7}, {"ID_D", "DELTA", {}, 7}});
    } catch (const std::invalid_argument&) { duplicate_tokens = true; }
    check(duplicate_tokens, "duplicate model token assignments must be rejected");

    bool conflicting_symbol = false;
    try {
        (void)GeneVocabulary::from_records({
            {"ID_E", "EPSILON", {}, {}}, {"ID_E", "OTHER", {}, {}}});
    } catch (const std::invalid_argument&) { conflicting_symbol = true; }
    check(conflicting_symbol, "conflicting symbols for one ID must be rejected");
}

void malformed_persistence_is_rejected() {
    for (const auto& [name, unknown_row] : std::vector<std::pair<std::string, std::string>>{
             {"trailing-meta.vocab", "UNKNOWN error -"},
             {"trailing-policy.vocab", "UNKNOWN error - trailing"}}) {
        const auto malformed = temp_path(name);
        {
            std::ofstream out(malformed);
            out << "CELLERATOR_GENE_VOCABULARY 1\n";
            const bool bad_metadata = name == "trailing-meta.vocab";
            out << (bad_metadata
                ? "META \"src\" \"human\" \"GRCh38\" \"test\" trailing\n"
                : "META \"src\" \"human\" \"GRCh38\" \"test\"\n");
            out << unknown_row << '\n';
            out << "GENE \"ID_A\" \"ALPHA\" 0 0\n";
        }
        bool rejected = false;
        try { (void)GeneVocabulary::load(malformed.string()); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "versioned vocabulary loader must reject trailing fields");
        std::filesystem::remove(malformed);
    }
}

void concrete_mechanisms_reuse_existing_contracts() {
    static_assert(std::is_base_of_v<GeneTokenizer, EventStreamGeneTokenizer>);
    static_assert(std::is_base_of_v<GeneTokenizer, RankedGeneTokenizer>);

    auto vocabulary = std::make_shared<const GeneVocabulary>(
        GeneVocabulary::from_records({
            {"ID_A", "ALPHA", {}, 50}, {"ID_B", "BETA", {}, 60},
            {"ID_C", "GAMMA", {}, 70}}));
    EventStreamGeneTokenizer events(vocabulary);
    auto offsets = at::tensor({0, 3, 3}, at::kLong);
    auto time = at::tensor({1.f, 2.f, 3.f});
    auto encoded = events.tokenize_names({"BETA", "ID_A", "BETA"}, offsets,
                                         {{"time", time}});
    check(values<std::int64_t>(encoded.token_ids)
              == std::vector<std::int64_t>({60, 50, 60})
              && values<std::int64_t>(encoded.offsets)
              == std::vector<std::int64_t>({0, 3, 3})
              && at::equal(encoded.attributes.at("time"), time),
          "event tokenizer should retain repeats, empty rows and attributes");

    RankedFeatureOptions options;
    options.max_per_row = 2;
    options.retain_measurements = true;
    RankedGeneTokenizer ranked(vocabulary, options);
    auto features = at::tensor({1.f, 4.f, 4.f, 2.f, 0.f, 1.f}).reshape({2, 3});
    auto feature_ids = vocabulary->identity_tensor();
    auto result = ranked.tokenize(features, feature_ids);
    auto reference = ranked_feature(features, feature_ids,
        vocabulary->integer_vocabulary(), options);
    check(at::equal(result.token_ids, reference.token_ids)
              && at::equal(result.offsets, reference.offsets)
              && at::equal(result.attributes.at("measurement"),
                           reference.attributes.at("measurement")),
          "ranked tokenizer should preserve deterministic ranking semantics");

    auto totals = at::tensor({9.f, 3.f});
    auto medians = at::tensor({1.f, 4.f, 2.f});
    auto geneformer = ranked.geneformer_style(features, totals, medians, feature_ids);
    auto geneformer_reference = cellerator::tokenization::geneformer_style(
        features, totals, medians, feature_ids, vocabulary->integer_vocabulary(), 2);
    check(at::equal(geneformer.token_ids, geneformer_reference.token_ids)
              && at::equal(geneformer.offsets, geneformer_reference.offsets),
          "ranked gene profile should use the established Geneformer scoring function");

    auto crow = at::tensor({0, 2, 3}, at::kLong);
    auto columns = at::tensor({0, 2, 1}, at::kLong);
    auto stored = at::tensor({1.f, 4.f, 2.f});
    auto sparse = at::sparse_csr_tensor(crow, columns, stored, {2, 3},
        at::TensorOptions().layout(at::kSparseCsr).dtype(at::kFloat));
    auto sparse_result = ranked.tokenize(sparse, feature_ids);
    auto sparse_reference = ranked_feature(sparse, feature_ids,
        vocabulary->integer_vocabulary(), options);
    check(at::equal(sparse_result.token_ids, sparse_reference.token_ids)
              && at::equal(sparse_result.offsets, sparse_reference.offsets),
          "ranked tokenizer should preserve the generic CSR execution path");

    std::unique_ptr<GeneTokenizer> polymorphic =
        std::make_unique<EventStreamGeneTokenizer>(vocabulary);
    check(&polymorphic->gene_vocabulary() == vocabulary.get(),
          "base ownership should retain the shared immutable vocabulary");
}

} // namespace

int main() {
    try {
        gtf_builds_stable_vocabulary_and_roundtrips();
        gff3_parsing_and_explicit_filters();
        canonical_ids_alias_conflicts_and_custom_tokens();
        malformed_persistence_is_rejected();
        concrete_mechanisms_reuse_existing_contracts();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
