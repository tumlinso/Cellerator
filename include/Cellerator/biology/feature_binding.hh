#pragma once

#include <Cellerator/biology/coordinates.hh>
#include <Cellerator/compute/tokenization/gene_vocabulary.hh>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cellerator::biology {

enum class UnknownIdentityPolicy : std::uint8_t {
    error,
    drop,
    map_to_token
};

inline const char* unknown_policy_name(UnknownIdentityPolicy policy) noexcept {
    switch (policy) {
    case UnknownIdentityPolicy::error: return "error";
    case UnknownIdentityPolicy::drop: return "drop";
    case UnknownIdentityPolicy::map_to_token: return "map_to_token";
    }
    return "error";
}

struct FeatureBindingOptions {
    std::string domain = "feature";
    std::vector<std::optional<std::int64_t>> canonical_indices;
    std::optional<AxisDescriptor> axis;
    std::optional<std::int64_t> model_capacity;
    UnknownIdentityPolicy unknown_policy = UnknownIdentityPolicy::error;
    std::optional<std::int64_t> unknown_model_index;
    std::map<std::string, std::string> metadata;
};

template<class Identity>
struct FeatureAlias {
    Identity identity;
    std::vector<Identity> aliases;
};

template<class EncodedIdentity>
struct EncodedFeatureAlias {
    EncodedIdentity identity;
    std::vector<EncodedIdentity> aliases;
};

// A typed, serialization-neutral descriptor.  The caller supplies an
// identity encoder; IO code owns JSON representation and validation.
template<class EncodedIdentity>
struct FeatureBindingDescriptor {
    std::int32_t schema_version = 1;
    std::string identity_codec_name;
    std::int32_t identity_codec_version = 1;
    std::string domain;
    std::vector<EncodedIdentity> identities;
    std::vector<std::optional<std::int64_t>> canonical_indices;
    std::vector<std::int64_t> model_indices;
    std::vector<EncodedFeatureAlias<EncodedIdentity>> aliases;
    std::optional<AxisDescriptor> axis;
    std::map<std::string, std::string> metadata;
    UnknownIdentityPolicy unknown_policy = UnknownIdentityPolicy::error;
    std::optional<std::int64_t> unknown_model_index;
    std::optional<std::int64_t> model_capacity;
};

template<class Identity>
class PreparedAlignment {
public:
    const std::vector<Identity>& canonical_ids() const noexcept {
        return canonical_ids_;
    }
    const std::vector<Identity>& source_ids() const noexcept { return source_ids_; }
    const std::vector<std::int64_t>& source_positions() const noexcept {
        return source_positions_;
    }
    const std::vector<std::int64_t>& gather_positions() const noexcept {
        return source_positions_;
    }
    const std::vector<std::int64_t>& output_columns() const noexcept {
        return source_positions_;
    }
    const std::vector<std::optional<std::int64_t>>& canonical_indices() const noexcept {
        return canonical_indices_;
    }
    const std::vector<std::int64_t>& model_indices() const noexcept {
        return model_indices_;
    }
    const std::vector<Identity>& dropped_ids() const noexcept { return dropped_ids_; }
    const std::vector<std::int64_t>& physical_positions() const noexcept {
        return physical_positions_;
    }
    bool has_physical_positions() const noexcept { return has_physical_positions_; }

private:
    std::vector<Identity> canonical_ids_;
    std::vector<Identity> source_ids_;
    std::vector<std::int64_t> source_positions_;
    std::vector<std::optional<std::int64_t>> canonical_indices_;
    std::vector<std::int64_t> model_indices_;
    std::vector<Identity> dropped_ids_;
    std::vector<std::int64_t> physical_positions_;
    bool has_physical_positions_ = false;
    std::shared_ptr<const std::uint8_t> binding_identity_;

    template<class Id>
    friend class FeatureBinding;
};

// A model-local mapping from canonical biological identities to model rows.
// Preparation returns integer gather coordinates only; numerical values stay
// with the caller, so dense, sparse, and externally managed inputs are not
// copied or coerced by this declaration.
template<class Identity>
class FeatureBinding {
public:
    struct ResolvedFeature {
        Identity canonical_id;
        std::optional<std::int64_t> canonical_index;
        std::int64_t model_index;
    };

    static FeatureBinding from_ids(
        std::vector<Identity> canonical_ids,
        std::vector<std::int64_t> model_indices = {},
        FeatureBindingOptions options = {}) {
        if (canonical_ids.empty()) {
            throw std::invalid_argument("feature binding requires at least one identity");
        }
        if (canonical_ids.size() > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("feature count exceeds int64 range");
        }
        for (std::size_t i = 0; i < canonical_ids.size(); ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                if (canonical_ids[i] == canonical_ids[j]) {
                    throw std::invalid_argument("canonical feature identities must be unique");
                }
            }
        }

        if (model_indices.empty()) {
            model_indices.resize(canonical_ids.size());
            for (std::size_t i = 0; i < model_indices.size(); ++i) {
                model_indices[i] = static_cast<std::int64_t>(i);
            }
        }
        if (model_indices.size() != canonical_ids.size()) {
            throw std::invalid_argument("model indices must align with canonical identities");
        }
        for (const auto index : model_indices) {
            if (index < 0) throw std::invalid_argument("model indices must be nonnegative");
        }

        if (options.canonical_indices.empty()) {
            options.canonical_indices.resize(canonical_ids.size());
        }
        if (options.canonical_indices.size() != canonical_ids.size()) {
            throw std::invalid_argument(
                "canonical indices must align with canonical identities");
        }
        for (std::size_t i = 0; i < options.canonical_indices.size(); ++i) {
            if (options.canonical_indices[i] && *options.canonical_indices[i] < 0) {
                throw std::invalid_argument("canonical indices must be nonnegative");
            }
            if (options.canonical_indices[i]) {
                for (std::size_t j = 0; j < i; ++j) {
                    if (options.canonical_indices[j]
                        && options.canonical_indices[i] == options.canonical_indices[j]) {
                        throw std::invalid_argument("canonical indices must be unique");
                    }
                }
            }
        }
        if (options.domain.empty()) {
            throw std::invalid_argument("feature domain must not be empty");
        }
        if (options.axis) {
            options.axis->validate();
            if (options.axis->extent
                != static_cast<std::int64_t>(canonical_ids.size())) {
                throw std::invalid_argument(
                    "axis extent must match the ordered feature identities");
            }
        }
        if (options.model_capacity) {
            if (*options.model_capacity <= 0) {
                throw std::invalid_argument("model capacity must be positive");
            }
            for (const auto index : model_indices) {
                if (index >= *options.model_capacity) {
                    throw std::invalid_argument("model index exceeds model capacity");
                }
            }
        }
        if ((options.unknown_policy == UnknownIdentityPolicy::map_to_token)
            != options.unknown_model_index.has_value()) {
            throw std::invalid_argument(
                "map_to_token requires exactly one unknown model index");
        }
        if (options.unknown_model_index && *options.unknown_model_index < 0) {
            throw std::invalid_argument("unknown model index must be nonnegative");
        }
        if (options.model_capacity && options.unknown_model_index
            && *options.unknown_model_index >= *options.model_capacity) {
            throw std::invalid_argument("unknown model index exceeds model capacity");
        }

        FeatureBinding result;
        result.canonical_ids_ = std::move(canonical_ids);
        result.model_indices_ = std::move(model_indices);
        result.options_ = std::move(options);
        result.aliases_.clear();
        result.binding_identity_ = std::make_shared<const std::uint8_t>(0);
        return result;
    }

    static FeatureBinding from_gene_vocabulary(
        const tokenization::GeneVocabulary& vocabulary,
        std::optional<AxisDescriptor> axis = {},
        std::optional<std::int64_t> model_capacity = {}) {
        static_assert(std::is_same<Identity, std::string>::value,
                      "GeneVocabulary bindings use std::string identities");
        const auto& genes = vocabulary.genes();
        if (genes.empty()) {
            throw std::invalid_argument("GeneVocabulary must contain at least one gene");
        }
        std::vector<std::string> ids;
        std::vector<std::int64_t> model_indices;
        FeatureBindingOptions options;
        options.domain = "gene";
        options.axis = std::move(axis);
        options.model_capacity = model_capacity;
        options.canonical_indices.reserve(genes.size());
        ids.reserve(genes.size());
        model_indices.reserve(genes.size());
        for (std::size_t i = 0; i < genes.size(); ++i) {
            if (!genes[i].token_id || *genes[i].token_id < 0) {
                throw std::invalid_argument("GeneVocabulary records need nonnegative token IDs");
            }
            ids.push_back(genes[i].gene_id);
            model_indices.push_back(*genes[i].token_id);
            options.canonical_indices.emplace_back(static_cast<std::int64_t>(i));

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

        auto result = from_ids(std::move(ids), std::move(model_indices), std::move(options));
        auto snapshot = std::make_shared<const tokenization::GeneVocabulary>(vocabulary);
        result.gene_vocabulary_ = snapshot;
        for (const auto& gene : genes) {
            std::vector<std::string> names = gene.aliases;
            if (!gene.symbol.empty()) names.push_back(gene.symbol);
            std::vector<std::string> unique_names;
            for (const auto& name : names) {
                if (name.empty()) continue;
                if (std::find(unique_names.begin(), unique_names.end(), name)
                    == unique_names.end()) unique_names.push_back(name);
            }
            if (!unique_names.empty()) {
                result.aliases_.push_back(FeatureAlias<Identity>{gene.gene_id,
                                                                  std::move(unique_names)});
            }
        }
        result.resolver_ = [snapshot, policy = result.options_.unknown_policy,
                            unknown_model_index = result.options_.unknown_model_index,
                            canonical_ids = result.canonical_ids_,
                            model_indices = result.model_indices_](const Identity& name)
            -> std::optional<ResolvedFeature> {
            std::int64_t canonical = -1;
            try {
                canonical = snapshot->resolve_identity(name);
            } catch (const std::out_of_range&) {
                if (policy == UnknownIdentityPolicy::drop) return std::nullopt;
                if (policy == UnknownIdentityPolicy::map_to_token) {
                    return ResolvedFeature{name, std::nullopt, *unknown_model_index};
                }
                throw;
            }
            if (canonical < 0 || static_cast<std::size_t>(canonical) >= canonical_ids.size()) {
                throw std::logic_error("GeneVocabulary resolver returned an invalid canonical index");
            }
            const auto index = static_cast<std::size_t>(canonical);
            return ResolvedFeature{canonical_ids[index], canonical, model_indices[index]};
        };
        return result;
    }

    const std::vector<Identity>& canonical_ids() const noexcept { return canonical_ids_; }
    const std::vector<Identity>& identities() const noexcept { return canonical_ids_; }
    const std::vector<std::int64_t>& model_indices() const noexcept {
        return model_indices_;
    }
    const std::vector<std::optional<std::int64_t>>& canonical_indices() const noexcept {
        return options_.canonical_indices;
    }
    const std::string& domain() const noexcept { return options_.domain; }
    const std::optional<AxisDescriptor>& axis() const noexcept { return options_.axis; }
    const std::optional<std::int64_t>& model_capacity() const noexcept {
        return options_.model_capacity;
    }
    UnknownIdentityPolicy unknown_policy() const noexcept {
        return options_.unknown_policy;
    }
    const std::optional<std::int64_t>& unknown_model_index() const noexcept {
        return options_.unknown_model_index;
    }
    const std::map<std::string, std::string>& metadata() const noexcept {
        return options_.metadata;
    }
    const std::vector<FeatureAlias<Identity>>& aliases() const noexcept { return aliases_; }
    const tokenization::GeneVocabulary* gene_vocabulary_snapshot() const noexcept {
        return gene_vocabulary_.get();
    }

    void validate_model_capacity(std::int64_t capacity) const {
        if (capacity <= 0) throw std::invalid_argument("model capacity must be positive");
        for (const auto index : model_indices_) {
            if (index >= capacity) {
                throw std::invalid_argument("feature model index exceeds encoder vocabulary size");
            }
        }
        if (options_.unknown_model_index && *options_.unknown_model_index >= capacity) {
            throw std::invalid_argument("unknown model index exceeds encoder vocabulary size");
        }
    }

    std::optional<ResolvedFeature> resolve(const Identity& identity) const {
        if (resolver_) return resolver_(identity);
        for (std::size_t i = 0; i < canonical_ids_.size(); ++i) {
            if (canonical_ids_[i] == identity) {
                return ResolvedFeature{
                    canonical_ids_[i], options_.canonical_indices[i], model_indices_[i]};
            }
        }
        if (options_.unknown_policy == UnknownIdentityPolicy::drop) return std::nullopt;
        if (options_.unknown_policy == UnknownIdentityPolicy::map_to_token) {
            return ResolvedFeature{identity, std::nullopt, *options_.unknown_model_index};
        }
        throw std::out_of_range("unknown feature identity");
    }

    std::int64_t model_index(const Identity& identity) const {
        const auto resolved = resolve(identity);
        if (!resolved) throw std::out_of_range("feature identity was dropped");
        return resolved->model_index;
    }

    PreparedAlignment<Identity> prepare(
        const std::vector<Identity>& source_order) const {
        return prepare_impl(source_order, source_order, {});
    }

    PreparedAlignment<Identity> prepare(
        const std::vector<Identity>& source_order,
        const std::vector<Identity>& requested_ids) const {
        return prepare_impl(source_order, requested_ids, {});
    }

    PreparedAlignment<Identity> prepare(
        const OccurrenceCoordinates<Identity>& source,
        const std::vector<Identity>& requested_ids) const {
        return prepare_impl(source.identities(), requested_ids,
                            source.has_physical_positions()
                                ? &source.physical_positions() : nullptr);
    }

    PreparedAlignment<Identity> prepare(
        const OccurrenceCoordinates<Identity>& source) const {
        return prepare(source, source.identities());
    }

    // Map every occurrence independently; repeated canonical IDs are retained.
    PreparedAlignment<Identity> map_occurrences(
        const std::vector<Identity>& source_order) const {
        return map_occurrences_impl(source_order, nullptr);
    }

    PreparedAlignment<Identity> map_occurrences(
        const OccurrenceCoordinates<Identity>& source) const {
        return map_occurrences_impl(source.identities(),
            source.has_physical_positions() ? &source.physical_positions() : nullptr);
    }

    void validate_alignment(const PreparedAlignment<Identity>& alignment) const {
        if (alignment.binding_identity_ != binding_identity_) {
            throw std::invalid_argument(
                "prepared alignment belongs to a different feature binding");
        }
    }

    template<class Encoder>
    auto to_descriptor(const std::string& codec_name,
                       std::int32_t codec_version,
                       Encoder&& encode_identity) const
        -> FeatureBindingDescriptor<typename std::decay<decltype(
            encode_identity(std::declval<const Identity&>()))>::type> {
        if (codec_name.empty() || codec_version <= 0) {
            throw std::invalid_argument("identity codec name/version must be declared");
        }
        using Encoded = typename std::decay<decltype(
            encode_identity(std::declval<const Identity&>()))>::type;
        FeatureBindingDescriptor<Encoded> result;
        result.identity_codec_name = codec_name;
        result.identity_codec_version = codec_version;
        result.domain = options_.domain;
        result.axis = options_.axis;
        result.metadata = options_.metadata;
        result.unknown_policy = options_.unknown_policy;
        result.unknown_model_index = options_.unknown_model_index;
        result.model_capacity = options_.model_capacity;
        result.canonical_indices = options_.canonical_indices;
        result.model_indices = model_indices_;
        result.identities.reserve(canonical_ids_.size());
        for (const auto& identity : canonical_ids_) {
            result.identities.push_back(encode_identity(identity));
        }
        result.aliases.reserve(aliases_.size());
        for (const auto& alias : aliases_) {
            EncodedFeatureAlias<Encoded> encoded;
            encoded.identity = encode_identity(alias.identity);
            encoded.aliases.reserve(alias.aliases.size());
            for (const auto& value : alias.aliases) {
                encoded.aliases.push_back(encode_identity(value));
            }
            result.aliases.push_back(std::move(encoded));
        }
        return result;
    }

private:
    using Resolver = std::function<std::optional<ResolvedFeature>(const Identity&)>;

    FeatureBinding() = default;

    PreparedAlignment<Identity> prepare_impl(
        const std::vector<Identity>& source_order,
        const std::vector<Identity>& requested_ids,
        const std::vector<std::int64_t>* physical_positions) const {
        if (physical_positions && physical_positions->size() != source_order.size()) {
            throw std::invalid_argument("physical positions must match source occurrence count");
        }

        PreparedAlignment<Identity> result;
        result.source_ids_.reserve(source_order.size());
        result.binding_identity_ = binding_identity_;
        std::vector<std::pair<Identity, std::int64_t>> positions;
        positions.reserve(source_order.size());
        for (std::size_t i = 0; i < source_order.size(); ++i) {
            const auto resolved = resolve(source_order[i]);
            if (!resolved) {
                result.source_ids_.push_back(source_order[i]);
                continue;
            }
            for (const auto& prior : positions) {
                if (prior.first == resolved->canonical_id) {
                    throw std::invalid_argument(
                        "source feature identities must be unique after alias resolution");
                }
            }
            result.source_ids_.push_back(resolved->canonical_id);
            positions.emplace_back(resolved->canonical_id, static_cast<std::int64_t>(i));
        }

        std::vector<Identity> selected;
        selected.reserve(requested_ids.size());
        for (const auto& requested : requested_ids) {
            const auto resolved = resolve(requested);
            if (!resolved) {
                result.dropped_ids_.push_back(requested);
                continue;
            }
            if (std::find(selected.begin(), selected.end(), resolved->canonical_id)
                != selected.end()) {
                throw std::invalid_argument("requested feature identities must be unique");
            }
            const auto source_position = std::find_if(
                positions.begin(), positions.end(), [&](const auto& item) {
                    return item.first == resolved->canonical_id;
                });
            if (source_position == positions.end()) {
                throw std::out_of_range("requested feature is absent from source order");
            }
            const auto position = source_position->second;
            selected.push_back(resolved->canonical_id);
            result.canonical_ids_.push_back(resolved->canonical_id);
            result.source_positions_.push_back(position);
            result.canonical_indices_.push_back(resolved->canonical_index);
            result.model_indices_.push_back(resolved->model_index);
            if (physical_positions) {
                result.physical_positions_.push_back(
                    physical_positions->at(static_cast<std::size_t>(position)));
            }
        }
        result.has_physical_positions_ = physical_positions != nullptr;
        return result;
    }

    PreparedAlignment<Identity> map_occurrences_impl(
        const std::vector<Identity>& source_order,
        const std::vector<std::int64_t>* physical_positions) const {
        if (physical_positions && physical_positions->size() != source_order.size()) {
            throw std::invalid_argument("physical positions must match source occurrence count");
        }
        PreparedAlignment<Identity> result;
        result.source_ids_.reserve(source_order.size());
        result.binding_identity_ = binding_identity_;
        for (std::size_t i = 0; i < source_order.size(); ++i) {
            const auto resolved = resolve(source_order[i]);
            if (!resolved) {
                result.source_ids_.push_back(source_order[i]);
                result.dropped_ids_.push_back(source_order[i]);
                continue;
            }
            result.source_ids_.push_back(resolved->canonical_id);
            result.canonical_ids_.push_back(resolved->canonical_id);
            result.source_positions_.push_back(static_cast<std::int64_t>(i));
            result.canonical_indices_.push_back(resolved->canonical_index);
            result.model_indices_.push_back(resolved->model_index);
            if (physical_positions) result.physical_positions_.push_back(physical_positions->at(i));
        }
        result.has_physical_positions_ = physical_positions != nullptr;
        return result;
    }

    std::vector<Identity> canonical_ids_;
    std::vector<std::int64_t> model_indices_;
    FeatureBindingOptions options_;
    std::vector<FeatureAlias<Identity>> aliases_;
    std::shared_ptr<const std::uint8_t> binding_identity_;
    std::shared_ptr<const tokenization::GeneVocabulary> gene_vocabulary_;
    Resolver resolver_;

};

} // namespace cellerator::biology
