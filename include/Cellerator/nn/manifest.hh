#pragma once

#include <Cellerator/nn/relations.hh>

#include <filesystem>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace cellerator::biology {
template<class EncodedIdentity>
struct FeatureBindingDescriptor;
} // namespace cellerator::biology

namespace cellerator::nn {

inline constexpr std::string_view biological_manifest_format =
    "cellerator-biological-model-manifest";
inline constexpr int biological_manifest_version = 1;

// Descriptors remain serialized JSON text at the public boundary. This keeps
// the native contract independent of the private JSON implementation while
// allowing typed Cellerator biology classes to encode their own identities.
struct BiologicalDeclaration {
    std::string module_path;
    std::string kind;
    std::string descriptor_json;
};

struct BiologicalRelationDeclaration {
    std::string name;
    std::string descriptor_json;
};

enum class BiologicalViewSourceKind {
    base_result,
    mapping_key,
    callable,
    module_selector,
};

struct BiologicalViewDeclaration {
    std::string module_path;
    std::string name;
    std::optional<std::string> output_binding_json;
    bool queryable = false;
    BiologicalViewSourceKind source_kind = BiologicalViewSourceKind::base_result;
    std::optional<std::string> source_key;
    std::optional<std::string> source_id;
};

struct BiologicalManifestWeights {
    std::string filename;
    std::string sha256;
};

// The v1 contract shared with python.cellerator.torch.manifest. An absent
// weights record is serialized as JSON null, exactly as in the Python API.
struct BiologicalManifest {
    std::vector<BiologicalDeclaration> declarations;
    std::vector<BiologicalRelationDeclaration> relations;
    std::vector<BiologicalViewDeclaration> views;
    std::optional<BiologicalManifestWeights> weights;
};

// Parse and serialize the shared, versioned manifest. Parsing rejects
// duplicate keys, malformed records, non-finite values, and payload-bearing
// descriptor fields. Identity encoding belongs to typed biological bindings;
// arbitrary identity types must be encoded with an explicit caller codec.
[[nodiscard]] BiologicalManifest parse_biological_manifest(std::string_view json);
[[nodiscard]] std::string serialize_biological_manifest(
    const BiologicalManifest& manifest);

// Render the typed native FeatureBinding descriptor using the same v1 wire
// fields as Python FeatureBinding.to_manifest(). The descriptor's encoded ID
// values are opaque JSON fragments produced by its explicit identity codec.
[[nodiscard]] std::string serialize_feature_binding_descriptor(
    const cellerator::biology::FeatureBindingDescriptor<std::string>& descriptor);

// Small typed encoders for the identity wire records accepted above. A custom
// identity codec supplies its own JSON-safe payload fragment and declares a
// stable codec name/version; no identity is serialized through `repr`.
[[nodiscard]] std::string feature_identity_json(std::string_view value);
[[nodiscard]] std::string feature_identity_json(bool value);
[[nodiscard]] std::string feature_identity_json(std::int64_t value);
[[nodiscard]] std::string feature_identity_json(std::uint64_t value);
[[nodiscard]] std::string feature_identity_json(double value);
[[nodiscard]] std::string feature_identity_none_json();
[[nodiscard]] std::string feature_identity128_json(std::uint64_t low,
                                                   std::uint64_t high);
[[nodiscard]] std::string feature_custom_identity_json(
    std::string_view codec_name,
    std::int32_t codec_version,
    std::string_view encoded_value_json);
[[nodiscard]] std::string feature_tuple_identity_json(
    const std::vector<std::string>& item_identity_json);

// PreparedRelation's endpoint tokens establish order and extent but do not
// contain identities. Supply the source and target occurrence declarations
// here; their identity vectors are serialized in the same v1 form as Python
// PreparedRelation.to_manifest(). Non-built-in identity types require an
// explicit encoder returning one JSON value as text.
[[nodiscard]] std::string relation_identity_json(std::string_view value);
[[nodiscard]] std::string relation_identity_json(bool value);
[[nodiscard]] std::string relation_identity_json(std::int64_t value);
[[nodiscard]] std::string relation_identity_json(std::uint64_t value);
[[nodiscard]] std::string relation_identity_json(double value);
[[nodiscard]] std::string relation_identity_none_json();
[[nodiscard]] std::string relation_identity128_json(std::uint64_t low,
                                                    std::uint64_t high);
[[nodiscard]] std::string relation_tuple_identity_json(
    const std::vector<std::string>& item_identity_json);

namespace detail {

template<class T>
struct relation_tuple_traits {
    static constexpr bool value = false;
};

template<class... T>
struct relation_tuple_traits<std::tuple<T...>> {
    static constexpr bool value = true;
};

template<class T>
struct relation_persistent_identity_traits {
    static constexpr bool value = false;
};

template<class Tag>
struct relation_persistent_identity_traits<cellerator::execution::persistent_identity<Tag>> {
    static constexpr bool value = true;
};

template<class T, class Codec>
std::string encode_relation_identity(const T& value, const Codec& codec);

template<class Tuple, class Codec, std::size_t... Index>
std::vector<std::string> encode_relation_tuple_items(
    const Tuple& value,
    const Codec& codec,
    std::index_sequence<Index...>) {
    std::vector<std::string> result;
    result.reserve(sizeof...(Index));
    (result.push_back(encode_relation_identity(
         std::get<Index>(value),
         codec)), ...);
    return result;
}

template<class T, class Codec>
std::string encode_relation_identity(const T& value, const Codec& codec) {
    using Identity = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<Identity, std::string>) {
        return relation_identity_json(std::string_view(value));
    } else if constexpr (std::is_same_v<Identity, bool>) {
        return relation_identity_json(value);
    } else if constexpr (std::is_integral_v<Identity> && std::is_signed_v<Identity>) {
        return relation_identity_json(static_cast<std::int64_t>(value));
    } else if constexpr (std::is_integral_v<Identity> && std::is_unsigned_v<Identity>) {
        return relation_identity_json(static_cast<std::uint64_t>(value));
    } else if constexpr (std::is_floating_point_v<Identity>) {
        return relation_identity_json(static_cast<double>(value));
    } else if constexpr (std::is_same_v<Identity, std::nullptr_t>) {
        return relation_identity_none_json();
    } else if constexpr (relation_persistent_identity_traits<Identity>::value) {
        return relation_identity128_json(value.low, value.high);
    } else if constexpr (relation_tuple_traits<Identity>::value) {
        auto items = encode_relation_tuple_items(
            value, codec, std::make_index_sequence<std::tuple_size_v<Identity>>{});
        return relation_tuple_identity_json(items);
    } else {
        if constexpr (std::is_same_v<std::remove_cv_t<Codec>, std::nullptr_t>) {
            throw std::invalid_argument("custom relation endpoint identities require an explicit codec");
        } else {
            if constexpr (requires { static_cast<bool>(codec); }) {
                if (!static_cast<bool>(codec))
                    throw std::invalid_argument("custom relation endpoint identities require an explicit codec");
            }
            return codec(value);
        }
    }
}

[[nodiscard]] std::string serialize_prepared_relation_descriptor(
    const PreparedRelation& relation,
    const std::vector<std::string>& source_order_json,
    const std::vector<std::string>& target_order_json);

} // namespace detail

template<class SourceIdentity, class TargetIdentity,
         class SourceCodec = std::nullptr_t,
         class TargetCodec = std::nullptr_t>
[[nodiscard]] std::string serialize_relation_descriptor(
    const PreparedRelation& relation,
    const cellerator::biology::OccurrenceCoordinates<SourceIdentity>& source,
    const cellerator::biology::OccurrenceCoordinates<TargetIdentity>& target,
    SourceCodec source_codec = {},
    TargetCodec target_codec = {},
    std::optional<std::uint64_t> expected_generation = std::nullopt,
    std::optional<at::Device> expected_device = std::nullopt) {
    const auto source_extent = static_cast<std::int64_t>(source.identities().size());
    const auto target_extent = static_cast<std::int64_t>(target.identities().size());
    const auto generation = expected_generation.value_or(relation.generation());
    const auto device = expected_device.value_or(relation.device());
    relation.validate_for(source.order_token(), target.order_token(),
                          source_extent, target_extent, generation, device);

    std::vector<std::string> source_order;
    source_order.reserve(source.identities().size());
    for (const auto& identity : source.identities())
        source_order.push_back(detail::encode_relation_identity(identity, source_codec));
    std::vector<std::string> target_order;
    target_order.reserve(target.identities().size());
    for (const auto& identity : target.identities())
        target_order.push_back(detail::encode_relation_identity(identity, target_codec));
    return detail::serialize_prepared_relation_descriptor(relation, source_order, target_order);
}

// Compare the complete stored contract (including the optional weights record)
// to the caller's expected declaration before loading parameter values.
void validate_biological_manifest(
    std::string_view stored_json,
    const BiologicalManifest& expected,
    const std::optional<std::filesystem::path>& weights_path = std::nullopt);

// UTF-8 JSON sidecar IO. Writes use a same-directory temporary file followed
// by rename so readers observe either the old or complete new manifest.
void save_biological_manifest(
    const std::filesystem::path& path,
    const BiologicalManifest& manifest,
    const std::optional<std::filesystem::path>& weights_path = std::nullopt);
[[nodiscard]] BiologicalManifest load_biological_manifest(
    const std::filesystem::path& path,
    const std::optional<std::filesystem::path>& weights_path = std::nullopt);

} // namespace cellerator::nn
