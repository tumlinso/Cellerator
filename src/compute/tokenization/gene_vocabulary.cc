#include <Cellerator/compute/tokenization/gene_vocabulary.hh>

#include <ATen/ATen.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace cellerator::tokenization {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

std::string trim(std::string value) {
    auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

bool contains_line_break(const std::string& value) {
    return value.find('\n') != std::string::npos
        || value.find('\r') != std::string::npos;
}

const char* policy_name(unknown_identity_policy policy) {
    switch (policy) {
    case unknown_identity_policy::error: return "error";
    case unknown_identity_policy::drop: return "drop";
    case unknown_identity_policy::map_to_token: return "map_to_token";
    }
    throw std::invalid_argument("invalid unknown identity policy");
}

unknown_identity_policy parse_policy(const std::string& value) {
    if (value == "error") return unknown_identity_policy::error;
    if (value == "drop") return unknown_identity_policy::drop;
    if (value == "map_to_token") return unknown_identity_policy::map_to_token;
    throw std::invalid_argument("unknown policy in gene vocabulary file");
}

} // namespace

GeneVocabulary::GeneVocabulary(std::vector<GeneRecord> genes,
                               GeneVocabularyMetadata metadata,
                               unknown_identity_policy unknown_policy,
                               std::optional<std::int64_t> unknown_token_id)
    : genes_(std::move(genes)), metadata_(std::move(metadata)),
      unknown_policy_(unknown_policy), unknown_token_id_(unknown_token_id) {
    for (std::size_t i = 0; i < genes_.size(); ++i) {
        const auto identity = static_cast<std::int64_t>(i);
        identities_by_id_.emplace(genes_[i].gene_id, identity);
        std::unordered_set<std::string> names(genes_[i].aliases.begin(),
                                              genes_[i].aliases.end());
        if (!genes_[i].symbol.empty()) names.insert(genes_[i].symbol);
        for (const auto& alias : names)
            identities_by_alias_[alias].push_back(identity);
    }
}

GeneVocabulary GeneVocabulary::from_records(
    std::vector<GeneRecord> records, GeneVocabularyMetadata metadata,
    unknown_identity_policy unknown_policy,
    std::optional<std::int64_t> unknown_token_id) {
    require((unknown_policy == unknown_identity_policy::map_to_token)
                == unknown_token_id.has_value(),
            "unknown token ID must be supplied only for map_to_token policy");
    require(!unknown_token_id || *unknown_token_id >= 0,
            "unknown token ID must be nonnegative");
    for (auto& record : records) {
        record.gene_id = trim(std::move(record.gene_id));
        record.symbol = trim(std::move(record.symbol));
        require(!record.gene_id.empty(), "gene IDs must not be empty");
        require(!contains_line_break(record.gene_id)
                    && !contains_line_break(record.symbol),
                "gene IDs and symbols must not contain line breaks");
        for (auto& alias : record.aliases) {
            alias = trim(std::move(alias));
            require(!contains_line_break(alias), "gene aliases must not contain line breaks");
        }
    }
    for (const auto* value : {&metadata.source, &metadata.organism,
                              &metadata.assembly, &metadata.release})
        require(!contains_line_break(*value), "gene metadata must not contain line breaks");

    std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
        return a.gene_id < b.gene_id;
    });
    std::vector<GeneRecord> genes;
    for (auto& record : records) {
        record.aliases.erase(std::remove(record.aliases.begin(), record.aliases.end(), ""),
                             record.aliases.end());
        std::sort(record.aliases.begin(), record.aliases.end());
        record.aliases.erase(std::unique(record.aliases.begin(), record.aliases.end()),
                             record.aliases.end());
        if (genes.empty() || genes.back().gene_id != record.gene_id) {
            genes.push_back(std::move(record));
            continue;
        }
        auto& previous = genes.back();
        require(previous.symbol.empty() || record.symbol.empty()
                    || previous.symbol == record.symbol,
                "conflicting symbols for one stable gene ID");
        if (previous.symbol.empty()) previous.symbol = std::move(record.symbol);
        require(!previous.token_id || !record.token_id
                    || previous.token_id == record.token_id,
                "conflicting token IDs for one stable gene ID");
        if (!previous.token_id) previous.token_id = record.token_id;
        previous.aliases.insert(previous.aliases.end(),
                                record.aliases.begin(), record.aliases.end());
        std::sort(previous.aliases.begin(), previous.aliases.end());
        previous.aliases.erase(std::unique(previous.aliases.begin(),
                                           previous.aliases.end()),
                               previous.aliases.end());
    }

    bool has_model_ids = false;
    bool has_missing_ids = false;
    std::unordered_set<std::int64_t> seen_tokens;
    for (const auto& gene : genes) {
        has_model_ids |= gene.token_id.has_value();
        has_missing_ids |= !gene.token_id.has_value();
        if (gene.token_id)
            require(*gene.token_id >= 0 && seen_tokens.insert(*gene.token_id).second,
                    "gene token IDs must be unique and nonnegative");
    }
    require(!(has_model_ids && has_missing_ids),
            "either every gene or no gene must specify a model token ID");
    if (!has_model_ids) {
        for (std::size_t i = 0; i < genes.size(); ++i)
            genes[i].token_id = static_cast<std::int64_t>(i);
    }
    if (unknown_token_id) {
        for (const auto& gene : genes)
            require(*gene.token_id != *unknown_token_id,
                    "unknown token ID must not collide with a gene token ID");
    }

    return GeneVocabulary(std::move(genes), std::move(metadata), unknown_policy,
                          unknown_token_id);
}



std::int64_t GeneVocabulary::resolve_identity(std::string_view name) const {
    require(!name.empty(), "gene identity or alias must not be empty");
    auto stable = identities_by_id_.find(std::string(name));
    if (stable != identities_by_id_.end()) return stable->second;
    auto alias = identities_by_alias_.find(std::string(name));
    if (alias == identities_by_alias_.end())
        throw std::out_of_range("unknown gene identity or alias: " + std::string(name));
    if (alias->second.size() != 1)
        throw std::invalid_argument("ambiguous gene alias: " + std::string(name));
    return alias->second.front();
}

std::vector<std::int64_t> GeneVocabulary::resolve_identities(
    const std::vector<std::string>& names) const {
    std::vector<std::int64_t> result;
    result.reserve(names.size());
    for (const auto& name : names) result.push_back(resolve_identity(name));
    return result;
}

at::Tensor GeneVocabulary::identity_tensor(at::Device device) const {
    std::vector<std::int64_t> identities(genes_.size());
    for (std::size_t i = 0; i < identities.size(); ++i)
        identities[i] = static_cast<std::int64_t>(i);
    return at::tensor(identities, at::TensorOptions().dtype(at::kLong)).to(device);
}

Vocabulary GeneVocabulary::integer_vocabulary(at::Device device) const {
    std::vector<std::int64_t> tokens;
    tokens.reserve(genes_.size());
    for (const auto& gene : genes_) tokens.push_back(*gene.token_id);
    auto token_tensor = at::tensor(tokens, at::TensorOptions().dtype(at::kLong)).to(device);
    return make_vocabulary(identity_tensor(device), token_tensor,
                           unknown_policy_, unknown_token_id_);
}

void GeneVocabulary::save(const std::string& path) const {
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("unable to write gene vocabulary: " + path);
    output << "CELLERATOR_GENE_VOCABULARY 1\n";
    output << "META " << std::quoted(metadata_.source) << ' '
           << std::quoted(metadata_.organism) << ' '
           << std::quoted(metadata_.assembly) << ' '
           << std::quoted(metadata_.release) << '\n';
    output << "UNKNOWN " << policy_name(unknown_policy_) << ' '
           << (unknown_token_id_ ? std::to_string(*unknown_token_id_) : "-") << '\n';
    for (const auto& gene : genes_) {
        output << "GENE " << std::quoted(gene.gene_id) << ' '
               << std::quoted(gene.symbol) << ' ' << *gene.token_id << ' '
               << gene.aliases.size();
        for (const auto& alias : gene.aliases) output << ' ' << std::quoted(alias);
        output << '\n';
    }
    if (!output) throw std::runtime_error("failed while writing gene vocabulary: " + path);
}

GeneVocabulary GeneVocabulary::load(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("unable to open gene vocabulary: " + path);
    std::string line;
    require(static_cast<bool>(std::getline(input, line))
                && line == "CELLERATOR_GENE_VOCABULARY 1",
            "unsupported gene vocabulary format or version");
    GeneVocabularyMetadata metadata;
    require(static_cast<bool>(std::getline(input, line)),
            "gene vocabulary is missing metadata");
    std::istringstream meta_line(line);
    std::string tag;
    require(static_cast<bool>(meta_line >> tag >> std::quoted(metadata.source)
                >> std::quoted(metadata.organism) >> std::quoted(metadata.assembly)
                >> std::quoted(metadata.release)) && tag == "META",
            "malformed gene vocabulary metadata");
    meta_line >> std::ws;
    require(meta_line.eof(), "unexpected trailing data in gene vocabulary metadata");
    require(static_cast<bool>(std::getline(input, line)),
            "gene vocabulary is missing its unknown-identity policy");
    std::istringstream unknown_line(line);
    std::string policy_text, token_text;
    require(static_cast<bool>(unknown_line >> tag >> policy_text >> token_text)
                && tag == "UNKNOWN",
            "malformed gene vocabulary unknown-identity policy");
    unknown_line >> std::ws;
    require(unknown_line.eof(), "unexpected trailing data in unknown-identity policy");
    auto policy = parse_policy(policy_text);
    std::optional<std::int64_t> unknown_token;
    if (token_text != "-") {
        std::size_t consumed = 0;
        const auto parsed = std::stoll(token_text, &consumed);
        require(consumed == token_text.size(), "invalid unknown token ID in vocabulary");
        unknown_token = parsed;
    }
    std::vector<GeneRecord> genes;
    std::string previous_gene_id;
    while (std::getline(input, line)) {
        std::istringstream gene_line(line);
        GeneRecord gene;
        std::int64_t token = -1;
        std::size_t alias_count = 0;
        require(static_cast<bool>(gene_line >> tag >> std::quoted(gene.gene_id)
                    >> std::quoted(gene.symbol) >> token >> alias_count),
                "malformed gene vocabulary record");
        require(tag == "GENE", "unexpected row in gene vocabulary");
        require(token >= 0, "gene vocabulary token ID must be nonnegative");
        require(previous_gene_id.empty() || previous_gene_id < gene.gene_id,
                "gene vocabulary records must be sorted and unique");
        previous_gene_id = gene.gene_id;
        gene.token_id = token;
        for (std::size_t i = 0; i < alias_count; ++i) {
            std::string alias;
            require(static_cast<bool>(gene_line >> std::quoted(alias)),
                    "truncated gene vocabulary aliases");
            gene.aliases.push_back(std::move(alias));
        }
        gene_line >> std::ws;
        require(gene_line.eof(), "unexpected trailing data in gene vocabulary record");
        genes.push_back(std::move(gene));
    }
    if (input.bad()) throw std::runtime_error("failed while reading gene vocabulary: " + path);
    return from_records(std::move(genes), std::move(metadata), policy, unknown_token);
}

} // namespace cellerator::tokenization
