#pragma once

#include <Cellerator/execution/dependency/feature_binding.hh>
#include <Cellerator/layout/axis_binding.hh>
#include <Cellerator/layout/occurrence_map.hh>
#include <Cellerator/structure/coordinates.hh>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator::biology {

namespace binding_detail {

template<class Identity>
struct FeatureBindingAdapterAccess;

template<class Identity, class = void>
struct is_hashable : std::false_type {};

template<class Identity>
struct is_hashable<Identity, std::void_t<decltype(
    std::hash<Identity>{}(std::declval<const Identity&>()))>> : std::true_type {};

template<class Identity, bool Hashable = is_hashable<Identity>::value>
class IdentityPositions;

template<class Identity>
class IdentityPositions<Identity, true> {
public:
    void reserve(std::size_t size) { positions_.reserve(size); }
    bool insert(const Identity& identity, std::size_t position) {
        return positions_.emplace(identity, position).second;
    }
    std::optional<std::size_t> find(const Identity& identity) const {
        const auto found = positions_.find(identity);
        if (found == positions_.end()) return std::nullopt;
        return found->second;
    }

private:
    std::unordered_map<Identity, std::size_t> positions_;
};

template<class Identity>
class IdentityPositions<Identity, false> {
public:
    void reserve(std::size_t size) { positions_.reserve(size); }
    bool insert(const Identity& identity, std::size_t position) {
        if (find(identity)) return false;
        positions_.emplace_back(identity, position);
        return true;
    }
    std::optional<std::size_t> find(const Identity& identity) const {
        const auto found = std::find_if(
            positions_.begin(), positions_.end(), [&](const auto& entry) {
                return entry.first == identity;
            });
        if (found == positions_.end()) return std::nullopt;
        return found->second;
    }

private:
    std::vector<std::pair<Identity, std::size_t>> positions_;
};

} // namespace binding_detail

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
    std::optional<structure::LogicalAxis> axis;
    std::optional<layout::AxisBinding> physical_axis;
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
    std::optional<std::vector<std::int64_t>> canonical_indices;
    std::vector<std::int64_t> model_indices;
    std::vector<EncodedFeatureAlias<EncodedIdentity>> aliases;
    std::optional<structure::LogicalAxis> axis;
    std::optional<layout::AxisBinding> physical_axis;
    std::map<std::string, std::string> metadata;
    UnknownIdentityPolicy unknown_policy = UnknownIdentityPolicy::error;
    std::optional<std::int64_t> unknown_model_index;
    std::optional<std::int64_t> model_capacity;
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
        const bool canonical_indices_declared = !options.canonical_indices.empty();
        if (canonical_ids.empty()) {
            throw std::invalid_argument("feature binding requires at least one identity");
        }
        if (canonical_ids.size() > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("feature count exceeds int64 range");
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
        if (canonical_indices_declared
            && std::any_of(options.canonical_indices.begin(),
                           options.canonical_indices.end(),
                           [](const auto& index) { return !index.has_value(); })) {
            throw std::invalid_argument(
                "declared canonical indices must provide one integer for every identity");
        }
        std::unordered_set<std::int64_t> unique_canonical_indices;
        unique_canonical_indices.reserve(options.canonical_indices.size());
        for (std::size_t i = 0; i < options.canonical_indices.size(); ++i) {
            if (options.canonical_indices[i] && *options.canonical_indices[i] < 0) {
                throw std::invalid_argument("canonical indices must be nonnegative");
            }
            if (options.canonical_indices[i]
                && !unique_canonical_indices.insert(*options.canonical_indices[i]).second) {
                throw std::invalid_argument("canonical indices must be unique");
            }
        }
        if (options.domain.empty()) {
            throw std::invalid_argument("feature domain must not be empty");
        }
        switch (options.unknown_policy) {
        case UnknownIdentityPolicy::error:
        case UnknownIdentityPolicy::drop:
        case UnknownIdentityPolicy::map_to_token:
            break;
        default:
            throw std::invalid_argument("unknown identity policy is not supported");
        }
        if (options.axis
            && (!structure::valid_logical_axis(*options.axis)
                || options.axis->extent != canonical_ids.size())) {
            throw std::invalid_argument(
                "logical axis must be valid and match the ordered feature identities");
        }
        if (options.physical_axis) {
            options.physical_axis->validate();
            if (options.physical_axis->logical.extent != canonical_ids.size()) {
                throw std::invalid_argument(
                    "physical axis binding extent must match the ordered feature identities");
            }
            if (options.axis && *options.axis != options.physical_axis->logical) {
                throw std::invalid_argument(
                    "physical axis binding must refer to the declared logical axis");
            }
            if (!options.axis) options.axis = options.physical_axis->logical;
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
        result.canonical_indices_declared_ = canonical_indices_declared;
        result.aliases_.clear();
        result.positions_.reserve(result.canonical_ids_.size());
        for (std::size_t i = 0; i < result.canonical_ids_.size(); ++i) {
            if (!result.positions_.insert(result.canonical_ids_[i], i)) {
                throw std::invalid_argument("canonical feature identities must be unique");
            }
        }
        result.binding_identity_ = std::make_shared<const std::uint8_t>(0);
        return result;
    }

    static FeatureBinding from_ids(
        std::span<const Identity> canonical_ids,
        std::span<const std::int64_t> model_indices = {},
        FeatureBindingOptions options = {}) {
        return from_ids(
            std::vector<Identity>(canonical_ids.begin(), canonical_ids.end()),
            std::vector<std::int64_t>(model_indices.begin(), model_indices.end()),
            std::move(options));
    }

    const std::vector<Identity>& canonical_ids() const noexcept { return canonical_ids_; }
    const std::vector<Identity>& identities() const noexcept { return canonical_ids_; }
    const std::vector<std::int64_t>& model_indices() const noexcept {
        return model_indices_;
    }
    const std::vector<std::optional<std::int64_t>>& canonical_indices() const noexcept {
        return options_.canonical_indices;
    }
    bool canonical_indices_declared() const noexcept {
        return canonical_indices_declared_;
    }
    const std::string& domain() const noexcept { return options_.domain; }
    const std::optional<structure::LogicalAxis>& axis() const noexcept {
        return options_.axis;
    }
    const std::optional<layout::AxisBinding>& physical_axis() const noexcept {
        return options_.physical_axis;
    }
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
        const auto position = positions_.find(identity);
        if (position) {
            const auto i = *position;
            return ResolvedFeature{
                canonical_ids_[i], options_.canonical_indices[i], model_indices_[i]};
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

    execution::FeatureSourceBinding<Identity> prepare(
        const std::vector<Identity>& source_order) const {
        return prepare_impl(source_order, source_order, nullptr);
    }

    execution::FeatureSourceBinding<Identity> prepare(
        std::span<const Identity> source_order) const {
        const std::vector<Identity> owned_source(source_order.begin(), source_order.end());
        return prepare_impl(owned_source, owned_source, nullptr);
    }

    execution::FeatureSourceBinding<Identity> prepare(
        const std::vector<Identity>& source_order,
        const std::vector<Identity>& requested_ids) const {
        return prepare_impl(source_order, requested_ids, nullptr);
    }

    execution::FeatureSourceBinding<Identity> prepare(
        std::span<const Identity> source_order,
        std::span<const Identity> requested_ids) const {
        return prepare_impl(
            std::vector<Identity>(source_order.begin(), source_order.end()),
            std::vector<Identity>(requested_ids.begin(), requested_ids.end()), nullptr);
    }

    execution::FeatureSourceBinding<Identity> prepare(
        const structure::OccurrenceCoordinates<Identity>& source,
        const std::vector<Identity>& requested_ids,
        const layout::OccurrenceMap* physical = nullptr) const {
        validate_physical_map(source, physical);
        return prepare_impl(source.identities(), requested_ids, physical);
    }

    execution::FeatureSourceBinding<Identity> prepare(
        const structure::OccurrenceCoordinates<Identity>& source,
        const layout::OccurrenceMap* physical = nullptr) const {
        return prepare(source, source.identities(), physical);
    }

    // Map every occurrence independently; repeated canonical IDs are retained.
    execution::FeatureSourceBinding<Identity> map_occurrences(
        const std::vector<Identity>& source_order) const {
        return map_occurrences_impl(source_order, nullptr);
    }

    execution::FeatureSourceBinding<Identity> map_occurrences(
        std::span<const Identity> source_order) const {
        return map_occurrences_impl(
            std::vector<Identity>(source_order.begin(), source_order.end()), nullptr);
    }

    execution::FeatureSourceBinding<Identity> map_occurrences(
        const structure::OccurrenceCoordinates<Identity>& source,
        const layout::OccurrenceMap* physical = nullptr) const {
        validate_physical_map(source, physical);
        return map_occurrences_impl(source.identities(), physical);
    }

    void validate_alignment(
        const execution::FeatureSourceBinding<Identity>& alignment) const {
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
        result.physical_axis = options_.physical_axis;
        if (options_.axis && (!options_.physical_axis || options_.axis->extent == 0)) {
            result.schema_version = 2;
        }
        result.metadata = options_.metadata;
        result.unknown_policy = options_.unknown_policy;
        result.unknown_model_index = options_.unknown_model_index;
        result.model_capacity = options_.model_capacity;
        if (canonical_indices_declared_) {
            std::vector<std::int64_t> indices;
            indices.reserve(options_.canonical_indices.size());
            for (const auto index : options_.canonical_indices) {
                indices.push_back(*index);
            }
            result.canonical_indices = std::move(indices);
        }
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

    friend struct binding_detail::FeatureBindingAdapterAccess<Identity>;

    static void validate_physical_map(
        const structure::OccurrenceCoordinates<Identity>& source,
        const layout::OccurrenceMap* physical) {
        if (physical && !physical->matches(source.order_token())) {
            throw std::invalid_argument(
                "physical occurrence map belongs to a different occurrence order");
        }
    }

    execution::FeatureSourceBinding<Identity> prepare_impl(
        const std::vector<Identity>& source_order,
        const std::vector<Identity>& requested_ids,
        const layout::OccurrenceMap* physical) const {
        if (source_order.size() > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("source occurrence count exceeds int64 range");
        }

        execution::FeatureSourceBinding<Identity> result;
        result.source_ids_.reserve(source_order.size());
        result.binding_identity_ = binding_identity_;
        binding_detail::IdentityPositions<Identity> positions;
        positions.reserve(source_order.size());
        for (std::size_t i = 0; i < source_order.size(); ++i) {
            const auto resolved = resolve(source_order[i]);
            if (!resolved) {
                result.source_ids_.push_back(source_order[i]);
                continue;
            }
            if (!positions.insert(resolved->canonical_id, i)) {
                throw std::invalid_argument(
                    "source feature identities must be unique after alias resolution");
            }
            result.source_ids_.push_back(resolved->canonical_id);
        }

        binding_detail::IdentityPositions<Identity> selected;
        selected.reserve(requested_ids.size());
        for (const auto& requested : requested_ids) {
            const auto resolved = resolve(requested);
            if (!resolved) {
                result.dropped_ids_.push_back(requested);
                continue;
            }
            if (!selected.insert(resolved->canonical_id, 0)) {
                throw std::invalid_argument("requested feature identities must be unique");
            }
            const auto source_position = positions.find(resolved->canonical_id);
            if (!source_position) {
                throw std::out_of_range("requested feature is absent from source order");
            }
            const auto position = static_cast<std::int64_t>(*source_position);
            result.canonical_ids_.push_back(resolved->canonical_id);
            result.source_positions_.push_back(position);
            result.canonical_indices_.push_back(resolved->canonical_index);
            result.model_indices_.push_back(resolved->model_index);
        }
        if (physical) {
            result.occurrence_map_ = physical->project(
                std::span<const std::int64_t>(result.source_positions_));
        }
        return result;
    }

    execution::FeatureSourceBinding<Identity> map_occurrences_impl(
        const std::vector<Identity>& source_order,
        const layout::OccurrenceMap* physical) const {
        if (source_order.size() > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("source occurrence count exceeds int64 range");
        }
        execution::FeatureSourceBinding<Identity> result;
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
        }
        if (physical) {
            result.occurrence_map_ = physical->project(
                std::span<const std::int64_t>(result.source_positions_));
        }
        return result;
    }

    std::vector<Identity> canonical_ids_;
    binding_detail::IdentityPositions<Identity> positions_;
    std::vector<std::int64_t> model_indices_;
    FeatureBindingOptions options_;
    std::vector<FeatureAlias<Identity>> aliases_;
    bool canonical_indices_declared_ = false;
    std::shared_ptr<const std::uint8_t> binding_identity_;
    Resolver resolver_;

};

} // namespace cellerator::biology
