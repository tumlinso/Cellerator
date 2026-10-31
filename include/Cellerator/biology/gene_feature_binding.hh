#pragma once

// Opt-in adapter for the optional annotation-backed GeneVocabulary.  Generic
// FeatureBinding and coordinates do not depend on ATen or tokenization.
#include <Cellerator/biology/feature_binding.hh>
#include <Cellerator/compute/tokenization/gene_vocabulary.hh>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cellerator::biology::binding_detail {

template<class Identity>
struct FeatureBindingAdapterAccess {
    template<class Resolver>
    static void attach(
        FeatureBinding<Identity>& binding,
        Resolver&& resolver,
        std::vector<FeatureAlias<Identity>> aliases) {
        binding.resolver_ = std::forward<Resolver>(resolver);
        binding.aliases_ = std::move(aliases);
    }
};

} // namespace cellerator::biology::binding_detail

namespace cellerator::biology {

// Build a model-local gene binding while reusing GeneVocabulary's canonical-ID
// precedence, alias ambiguity checks, and unknown-identity resolver policy.
// The copied vocabulary snapshot keeps the resolver alive without adding a
// global alias registry or tying the result to the caller's vocabulary object.
inline FeatureBinding<std::string> from_gene_vocabulary(
    const tokenization::GeneVocabulary& vocabulary,
    std::optional<AxisDescriptor> axis = {},
    std::optional<std::int64_t> model_capacity = {}) {
    const auto& genes = vocabulary.genes();
    if (genes.empty()) {
        throw std::invalid_argument("GeneVocabulary must contain at least one gene");
    }

    std::vector<std::string> canonical_ids;
    std::vector<std::int64_t> model_indices;
    FeatureBindingOptions options;
    options.domain = "gene";
    options.axis = std::move(axis);
    options.model_capacity = model_capacity;
    canonical_ids.reserve(genes.size());
    model_indices.reserve(genes.size());
    options.canonical_indices.reserve(genes.size());
    std::vector<FeatureAlias<std::string>> aliases;
    aliases.reserve(genes.size());

    for (std::size_t i = 0; i < genes.size(); ++i) {
        const auto& gene = genes[i];
        if (!gene.token_id || *gene.token_id < 0) {
            throw std::invalid_argument("GeneVocabulary records need nonnegative token IDs");
        }
        canonical_ids.push_back(gene.gene_id);
        model_indices.push_back(*gene.token_id);
        options.canonical_indices.emplace_back(static_cast<std::int64_t>(i));

        std::vector<std::string> names = gene.aliases;
        if (!gene.symbol.empty()) names.push_back(gene.symbol);
        std::vector<std::string> unique_names;
        for (const auto& name : names) {
            if (!name.empty()
                && std::find(unique_names.begin(), unique_names.end(), name)
                    == unique_names.end()) {
                unique_names.push_back(name);
            }
        }
        if (!unique_names.empty()) {
            aliases.push_back({gene.gene_id, std::move(unique_names)});
        }
    }

    options.metadata = {
        {"source", vocabulary.metadata().source},
        {"organism", vocabulary.metadata().organism},
        {"assembly", vocabulary.metadata().assembly},
        {"release", vocabulary.metadata().release},
    };
    const auto native = vocabulary.integer_vocabulary();
    switch (native.unknown_policy) {
    case tokenization::unknown_identity_policy::error:
        options.unknown_policy = UnknownIdentityPolicy::error;
        break;
    case tokenization::unknown_identity_policy::drop:
        options.unknown_policy = UnknownIdentityPolicy::drop;
        break;
    case tokenization::unknown_identity_policy::map_to_token:
        options.unknown_policy = UnknownIdentityPolicy::map_to_token;
        options.unknown_model_index = native.unknown_token_id;
        break;
    }

    auto binding = FeatureBinding<std::string>::from_ids(
        canonical_ids, model_indices, std::move(options));
    auto snapshot = std::make_shared<const tokenization::GeneVocabulary>(vocabulary);
    using Binding = FeatureBinding<std::string>;
    using Resolved = typename Binding::ResolvedFeature;
    auto resolver = [snapshot, policy = binding.unknown_policy(),
                     unknown_model_index = binding.unknown_model_index(),
                     canonical_ids = std::move(canonical_ids),
                     model_indices = std::move(model_indices)](const std::string& name)
        -> std::optional<Resolved> {
        std::int64_t canonical = -1;
        try {
            canonical = snapshot->resolve_identity(name);
        } catch (const std::out_of_range&) {
            if (policy == UnknownIdentityPolicy::drop) return std::nullopt;
            if (policy == UnknownIdentityPolicy::map_to_token) {
                return Resolved{name, std::nullopt, *unknown_model_index};
            }
            throw;
        }
        if (canonical < 0 || static_cast<std::size_t>(canonical) >= canonical_ids.size()) {
            throw std::logic_error(
                "GeneVocabulary resolver returned an invalid canonical index");
        }
        const auto index = static_cast<std::size_t>(canonical);
        return Resolved{canonical_ids[index], canonical, model_indices[index]};
    };
    binding_detail::FeatureBindingAdapterAccess<std::string>::attach(
        binding, std::move(resolver), std::move(aliases));
    return binding;
}

} // namespace cellerator::biology
