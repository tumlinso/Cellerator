#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
    module,
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
