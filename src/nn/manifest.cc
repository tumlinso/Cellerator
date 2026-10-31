#include <Cellerator/nn/manifest.hh>
#include <Cellerator/biology/feature_binding.hh>

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <fstream>
#include <array>
#include <chrono>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <set>
#include <stdexcept>
#include <system_error>
#include <unordered_set>

namespace cellerator::nn {
namespace {

using Json = nlohmann::json;

const std::unordered_set<std::string>& payload_keys() {
    static const std::unordered_set<std::string> keys{
        "history", "histories", "context", "contexts", "conditioning",
        "latent", "latents", "activation", "activations", "cell_state",
        "cell_states", "event_values", "observations", "measurement_values",
        "values", "sample_values", "example", "examples", "batch",
        "batch_values", "latent_activations", "history_values",
    };
    return keys;
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

Json parse_json(std::string_view text, const char* description) {
    try {
        std::vector<std::set<std::string>> object_keys;
        auto callback = [&object_keys](int, Json::parse_event_t event, Json& parsed) {
            if (event == Json::parse_event_t::object_start) {
                object_keys.emplace_back();
            } else if (event == Json::parse_event_t::key) {
                if (object_keys.empty())
                    throw std::invalid_argument("JSON parser lost object-key context");
                const auto key = parsed.get<std::string>();
                if (!object_keys.back().insert(key).second)
                    throw std::invalid_argument("duplicate JSON object key " + key);
            } else if (event == Json::parse_event_t::object_end) {
                if (!object_keys.empty()) object_keys.pop_back();
            }
            return true;
        };
        return Json::parse(text.begin(), text.end(), callback, true, false);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("invalid ") + description + ": " + error.what());
    }
}

void validate_payload_free(const Json& value) {
    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (payload_keys().count(lower_ascii(it.key())))
                throw std::invalid_argument(
                    "biological manifest must not serialize payload field " + it.key());
            validate_payload_free(it.value());
        }
    } else if (value.is_array()) {
        for (const auto& item : value) validate_payload_free(item);
    } else if (value.is_number_float() && !std::isfinite(value.get<double>())) {
        throw std::invalid_argument("biological manifest numbers must be finite");
    }
}

const Json& required(const Json& object, const char* key, Json::value_t type) {
    if (!object.is_object() || !object.contains(key) || object.at(key).type() != type)
        throw std::invalid_argument(std::string("manifest field '") + key + "' has an invalid shape");
    return object.at(key);
}

const Json& required_any(const Json& object, const char* key) {
    if (!object.is_object() || !object.contains(key))
        throw std::invalid_argument(std::string("manifest is missing field '") + key + "'");
    return object.at(key);
}

void require_exact_keys(const Json& object,
                        std::initializer_list<const char*> expected,
                        const char* description) {
    if (!object.is_object())
        throw std::invalid_argument(std::string(description) + " must be an object");
    std::set<std::string> expected_set;
    for (const auto* key : expected) expected_set.insert(key);
    std::set<std::string> actual_set;
    for (auto it = object.begin(); it != object.end(); ++it) actual_set.insert(it.key());
    if (actual_set != expected_set)
        throw std::invalid_argument(std::string(description) + " has unsupported or missing fields");
}

std::string required_string(const Json& object, const char* key, bool nonempty) {
    const auto& value = required(object, key, Json::value_t::string);
    auto text = value.get<std::string>();
    if (nonempty && text.empty())
        throw std::invalid_argument(std::string("manifest field '") + key + "' must be nonempty");
    return text;
}

Json descriptor_value(std::string_view text, const char* description) {
    auto value = parse_json(text, description);
    validate_payload_free(value);
    return value;
}

void validate_weights(const Json& value) {
    if (value.is_null()) return;
    if (!value.is_object())
        throw std::invalid_argument("manifest weights must be a file record or null");
    require_exact_keys(value, {"filename", "sha256"}, "manifest weights");
    const auto filename = required_string(value, "filename", true);
    const auto digest = required_string(value, "sha256", true);
    if (std::filesystem::path(filename).filename() != filename)
        throw std::invalid_argument("manifest weight filename must not contain a directory");
    if (digest.size() != 64 || !std::all_of(digest.begin(), digest.end(), [](unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }))
        throw std::invalid_argument("manifest weight SHA-256 must be lowercase hexadecimal");
}

Json to_json(const BiologicalManifest& manifest) {
    Json declarations = Json::array();
    std::set<std::pair<std::string, std::string>> declaration_keys;
    for (const auto& declaration : manifest.declarations) {
        if (declaration.kind.empty())
            throw std::invalid_argument("biological declaration kinds must be nonempty");
        if (!declaration_keys.emplace(declaration.module_path, declaration.kind).second)
            throw std::invalid_argument("duplicate biological declaration at module path");
        declarations.push_back({
            {"module_path", declaration.module_path},
            {"kind", declaration.kind},
            {"descriptor", descriptor_value(declaration.descriptor_json,
                                             "biological declaration descriptor")},
        });
    }

    Json relations = Json::array();
    std::set<std::string> relation_names;
    for (const auto& relation : manifest.relations) {
        if (relation.name.empty())
            throw std::invalid_argument("explicit relation names must be nonempty");
        if (!relation_names.insert(relation.name).second)
            throw std::invalid_argument("duplicate explicit relation name");
        relations.push_back({
            {"name", relation.name},
            {"descriptor", descriptor_value(relation.descriptor_json,
                                             "biological relation descriptor")},
        });
    }

    Json views = Json::array();
    std::set<std::pair<std::string, std::string>> view_keys;
    for (const auto& view : manifest.views) {
        if (view.name.empty())
            throw std::invalid_argument("biological view names must be nonempty");
        if (!view_keys.emplace(view.module_path, view.name).second)
            throw std::invalid_argument("duplicate biological view name at module path");
        const char* source_kind = nullptr;
        switch (view.source_kind) {
            case BiologicalViewSourceKind::base_result: source_kind = "base_result"; break;
            case BiologicalViewSourceKind::mapping_key: source_kind = "mapping_key"; break;
            case BiologicalViewSourceKind::callable: source_kind = "callable"; break;
            case BiologicalViewSourceKind::module_selector: source_kind = "module"; break;
            default: throw std::invalid_argument("unsupported biological view source kind");
        }
        if (view.source_kind == BiologicalViewSourceKind::mapping_key) {
            if (!view.source_key || view.source_key->empty())
                throw std::invalid_argument("mapping-key views require a nonempty source key");
        } else if (view.source_key) {
            throw std::invalid_argument("only mapping-key views may declare source_key");
        }
        if (view.source_kind == BiologicalViewSourceKind::callable ||
            view.source_kind == BiologicalViewSourceKind::module_selector) {
            if (!view.source_id || view.source_id->empty())
                throw std::invalid_argument("callable and module views require an explicit source_id");
        } else if (view.source_id) {
            throw std::invalid_argument("non-callable views must not declare source_id");
        }
        Json output_binding = nullptr;
        if (view.output_binding_json)
            output_binding = descriptor_value(*view.output_binding_json,
                                              "biological output binding");
        views.push_back({
            {"module_path", view.module_path},
            {"name", view.name},
            {"output_binding", std::move(output_binding)},
            {"queryable", view.queryable},
            {"source_kind", source_kind},
            {"source_key", view.source_key ? Json(*view.source_key) : Json(nullptr)},
            {"source_id", view.source_id ? Json(*view.source_id) : Json(nullptr)},
        });
    }

    Json weights = nullptr;
    if (manifest.weights) {
        weights = Json{{"filename", manifest.weights->filename},
                       {"sha256", manifest.weights->sha256}};
        validate_weights(weights);
    }

    return Json{
        {"format", biological_manifest_format},
        {"version", biological_manifest_version},
        {"declarations", std::move(declarations)},
        {"relations", std::move(relations)},
        {"views", std::move(views)},
        {"weights", std::move(weights)},
    };
}

BiologicalManifestWeights weight_record(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
        throw std::invalid_argument("biological manifest weights_path must name a regular file");
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open paired biological weight file");

    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("cannot initialize SHA-256 for biological weight file");
    std::array<char, 1 << 16> buffer{};
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = stream.gcount();
        if (count > 0 && EVP_DigestUpdate(context.get(), buffer.data(),
                                          static_cast<std::size_t>(count)) != 1)
            throw std::runtime_error("cannot hash biological weight file");
    }
    if (!stream.eof()) throw std::runtime_error("cannot read paired biological weight file");

    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 || digest_size != 32)
        throw std::runtime_error("cannot finish SHA-256 for biological weight file");
    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_size; ++i)
        hex << std::setw(2) << static_cast<unsigned int>(digest[i]);
    return {path.filename().string(), hex.str()};
}

void verify_weight_pair(
    const std::optional<BiologicalManifestWeights>& expected,
    const std::optional<std::filesystem::path>& weights_path) {
    if (!expected) {
        if (weights_path)
            throw std::invalid_argument("manifest does not declare a paired weight file");
        return;
    }
    if (!weights_path)
        throw std::invalid_argument("weights_path is required to verify the manifest weight record");
    const auto actual = weight_record(*weights_path);
    if (actual.filename != expected->filename || actual.sha256 != expected->sha256)
        throw std::invalid_argument("weight file name or SHA-256 does not match biological manifest");
}

std::uint64_t nonnegative_identity_word(const Json& value, const char* description) {
    if (value.is_number_unsigned()) return value.get<std::uint64_t>();
    if (value.is_number_integer()) {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value >= 0) return static_cast<std::uint64_t>(signed_value);
    }
    throw std::invalid_argument(std::string(description) + " must be a nonnegative integer");
}

void validate_encoded_identity(const Json& identity,
                               const std::string& codec_name,
                               std::int32_t codec_version) {
    if (!identity.is_object())
        throw std::invalid_argument("encoded feature identities must use tagged objects");
    const auto type = required_string(identity, "type", true);
    if (type == "str") {
        require_exact_keys(identity, {"type", "value"}, "string identity");
        if (!identity.at("value").is_string())
            throw std::invalid_argument("string identity value must be a string");
    } else if (type == "bool") {
        require_exact_keys(identity, {"type", "value"}, "Boolean identity");
        if (!identity.at("value").is_boolean())
            throw std::invalid_argument("Boolean identity value must be a Boolean");
    } else if (type == "int") {
        require_exact_keys(identity, {"type", "value"}, "integer identity");
        if (!identity.at("value").is_number_integer() &&
            !identity.at("value").is_number_unsigned())
            throw std::invalid_argument("integer identity value must be an integer");
    } else if (type == "float") {
        require_exact_keys(identity, {"type", "value"}, "floating identity");
        if (!identity.at("value").is_number())
            throw std::invalid_argument("floating identity value must be numeric");
    } else if (type == "none") {
        require_exact_keys(identity, {"type"}, "none identity");
    } else if (type == "identity128") {
        require_exact_keys(identity, {"type", "low", "high"}, "128-bit identity");
        (void)nonnegative_identity_word(identity.at("low"), "identity low word");
        (void)nonnegative_identity_word(identity.at("high"), "identity high word");
    } else if (type == "tuple") {
        require_exact_keys(identity, {"type", "items"}, "tuple identity");
        if (!identity.at("items").is_array())
            throw std::invalid_argument("tuple identity items must be an array");
        for (const auto& item : identity.at("items"))
            validate_encoded_identity(item, codec_name, codec_version);
    } else if (type == "custom") {
        require_exact_keys(identity, {"type", "codec", "version", "value"},
                           "custom identity");
        if (!identity.at("codec").is_string() || identity.at("codec").get<std::string>().empty())
            throw std::invalid_argument("custom identity codec name must be nonempty");
        const auto& version = identity.at("version");
        if ((!version.is_number_integer() && !version.is_number_unsigned()) ||
            version.get<std::int64_t>() <= 0)
            throw std::invalid_argument("custom identity codec version must be positive");
        if (identity.at("codec").get<std::string>() != codec_name ||
            version.get<std::int32_t>() != codec_version)
            throw std::invalid_argument("custom identity codec does not match the declared identity codec");
        validate_payload_free(identity.at("value"));
    } else {
        throw std::invalid_argument("unsupported tagged feature identity type " + type);
    }
}

Json identity128(std::uint64_t low, std::uint64_t high) {
    return Json{{"type", "identity128"}, {"low", low}, {"high", high}};
}

BiologicalManifest from_json(const Json& value) {
    if (!value.is_object())
        throw std::invalid_argument("biological manifest must be a JSON object");
    if (required_string(value, "format", true) != biological_manifest_format)
        throw std::invalid_argument("unsupported biological manifest format");
    const auto& version = required_any(value, "version");
    if ((!version.is_number_integer() && !version.is_number_unsigned()) ||
        version.get<std::int64_t>() != biological_manifest_version)
        throw std::invalid_argument("unsupported biological manifest version");

    BiologicalManifest manifest;
    const auto& declarations = required(value, "declarations", Json::value_t::array);
    for (const auto& declaration : declarations) {
        if (!declaration.is_object())
            throw std::invalid_argument("manifest declarations must contain objects");
        require_exact_keys(declaration, {"module_path", "kind", "descriptor"},
                           "manifest declaration");
        BiologicalDeclaration item;
        item.module_path = required_string(declaration, "module_path", false);
        item.kind = required_string(declaration, "kind", true);
        const auto& descriptor = required_any(declaration, "descriptor");
        validate_payload_free(descriptor);
        item.descriptor_json = descriptor.dump(-1, ' ', false, Json::error_handler_t::strict);
        manifest.declarations.push_back(std::move(item));
    }

    const auto& relations = required(value, "relations", Json::value_t::array);
    for (const auto& relation : relations) {
        if (!relation.is_object())
            throw std::invalid_argument("manifest relations must contain objects");
        require_exact_keys(relation, {"name", "descriptor"}, "manifest relation");
        BiologicalRelationDeclaration item;
        item.name = required_string(relation, "name", true);
        const auto& descriptor = required_any(relation, "descriptor");
        validate_payload_free(descriptor);
        item.descriptor_json = descriptor.dump(-1, ' ', false, Json::error_handler_t::strict);
        manifest.relations.push_back(std::move(item));
    }

    const auto& views = required(value, "views", Json::value_t::array);
    for (const auto& entry : views) {
        if (!entry.is_object())
            throw std::invalid_argument("manifest views must contain objects");
        require_exact_keys(entry, {"module_path", "name", "output_binding", "queryable",
                                   "source_kind", "source_key", "source_id"},
                           "manifest view");
        BiologicalViewDeclaration item;
        item.module_path = required_string(entry, "module_path", false);
        item.name = required_string(entry, "name", true);
        const auto& output_binding = required_any(entry, "output_binding");
        if (!output_binding.is_null()) {
            validate_payload_free(output_binding);
            item.output_binding_json = output_binding.dump(-1, ' ', false,
                                                            Json::error_handler_t::strict);
        }
        const auto& queryable = required(entry, "queryable", Json::value_t::boolean);
        item.queryable = queryable.get<bool>();
        const auto source_kind = required_string(entry, "source_kind", true);
        if (source_kind == "base_result") item.source_kind = BiologicalViewSourceKind::base_result;
        else if (source_kind == "mapping_key") item.source_kind = BiologicalViewSourceKind::mapping_key;
        else if (source_kind == "callable") item.source_kind = BiologicalViewSourceKind::callable;
        else if (source_kind == "module") item.source_kind = BiologicalViewSourceKind::module_selector;
        else throw std::invalid_argument("unsupported biological view source kind");

        const auto& source_key = required_any(entry, "source_key");
        if (!source_key.is_null()) {
            if (!source_key.is_string())
                throw std::invalid_argument("view source_key must be a string or null");
            item.source_key = source_key.get<std::string>();
        }
        const auto& source_id = required_any(entry, "source_id");
        if (!source_id.is_null()) {
            if (!source_id.is_string())
                throw std::invalid_argument("view source_id must be a string or null");
            item.source_id = source_id.get<std::string>();
        }
        manifest.views.push_back(std::move(item));
    }

    const auto& weights = required_any(value, "weights");
    validate_weights(weights);
    if (!weights.is_null()) {
        manifest.weights = BiologicalManifestWeights{
            required_string(weights, "filename", true),
            required_string(weights, "sha256", true),
        };
    }

    // Re-serializing performs uniqueness and source-selector checks and rejects
    // undeclared representation fields that could not be retained by this API.
    (void)to_json(manifest);
    return manifest;
}

} // namespace

BiologicalManifest parse_biological_manifest(std::string_view json) {
    auto parsed = parse_json(json, "biological manifest");
    const auto manifest = from_json(parsed);
    // v1 has exactly these six outer fields. Unknown fields are rejected so a
    // parse/serialize cycle cannot silently drop a declaration.
    const std::set<std::string> expected{
        "format", "version", "declarations", "relations", "views", "weights",
    };
    std::set<std::string> actual;
    for (auto it = parsed.begin(); it != parsed.end(); ++it) actual.insert(it.key());
    if (actual != expected)
        throw std::invalid_argument("biological manifest contains unsupported top-level fields");
    return manifest;
}

std::string serialize_biological_manifest(const BiologicalManifest& manifest) {
    return to_json(manifest).dump(-1, ' ', false, Json::error_handler_t::strict);
}

std::string serialize_feature_binding_descriptor(
    const cellerator::biology::FeatureBindingDescriptor<std::string>& descriptor) {
    if (descriptor.schema_version != 1)
        throw std::invalid_argument("unsupported native FeatureBinding descriptor version");
    if (descriptor.identity_codec_name.empty() || descriptor.identity_codec_version <= 0)
        throw std::invalid_argument("native FeatureBinding identity codec must declare a name and version");
    if (descriptor.domain.empty())
        throw std::invalid_argument("native FeatureBinding domain must be nonempty");
    if (descriptor.model_indices.size() != descriptor.identities.size())
        throw std::invalid_argument("native FeatureBinding identities and model indices must align");
    if (descriptor.canonical_indices &&
        descriptor.canonical_indices->size() != descriptor.identities.size())
        throw std::invalid_argument("native FeatureBinding canonical indices must align with identities");

    Json identities = Json::array();
    for (const auto& encoded : descriptor.identities) {
        auto identity = parse_json(encoded, "encoded FeatureBinding identity");
        validate_encoded_identity(identity, descriptor.identity_codec_name,
                                  descriptor.identity_codec_version);
        identities.push_back(std::move(identity));
    }

    Json aliases = Json::array();
    for (const auto& alias : descriptor.aliases) {
        auto encoded_identity = parse_json(alias.identity, "encoded FeatureBinding alias identity");
        validate_encoded_identity(encoded_identity, descriptor.identity_codec_name,
                                  descriptor.identity_codec_version);
        Json encoded_aliases = Json::array();
        for (const auto& value : alias.aliases) {
            auto encoded = parse_json(value, "encoded FeatureBinding identity alias");
            validate_encoded_identity(encoded, descriptor.identity_codec_name,
                                      descriptor.identity_codec_version);
            encoded_aliases.push_back(std::move(encoded));
        }
        aliases.push_back({{"identity", std::move(encoded_identity)},
                           {"aliases", std::move(encoded_aliases)}});
    }

    Json axis = nullptr;
    if (descriptor.axis) {
        descriptor.axis->validate();
        axis = Json{
            {"domain", identity128(descriptor.axis->domain.low, descriptor.axis->domain.high)},
            {"order", identity128(descriptor.axis->order.low, descriptor.axis->order.high)},
            {"geometry", identity128(descriptor.axis->geometry.low, descriptor.axis->geometry.high)},
            {"partition", identity128(descriptor.axis->partition.low, descriptor.axis->partition.high)},
            {"extent", descriptor.axis->extent},
        };
    }

    Json canonical_indices = nullptr;
    if (descriptor.canonical_indices) canonical_indices = *descriptor.canonical_indices;
    Json metadata = Json::object();
    for (const auto& [key, value] : descriptor.metadata) metadata[key] = value;

    auto result = Json{
        {"schema_version", descriptor.schema_version},
        {"domain", descriptor.domain},
        {"identities", std::move(identities)},
        {"canonical_indices", std::move(canonical_indices)},
        {"axis", std::move(axis)},
        {"model_indices", descriptor.model_indices},
        {"aliases", std::move(aliases)},
        {"metadata", std::move(metadata)},
        {"unknown_policy", cellerator::biology::unknown_policy_name(descriptor.unknown_policy)},
        {"unknown_model_index", descriptor.unknown_model_index
                                     ? Json(*descriptor.unknown_model_index) : Json(nullptr)},
        {"model_capacity", descriptor.model_capacity
                               ? Json(*descriptor.model_capacity) : Json(nullptr)},
    };
    validate_payload_free(result);
    return result.dump(-1, ' ', false, Json::error_handler_t::strict);
}

std::string feature_identity_json(std::string_view value) {
    return Json{{"type", "str"}, {"value", value}}.dump(-1, ' ', false);
}

std::string feature_identity_json(bool value) {
    return Json{{"type", "bool"}, {"value", value}}.dump(-1, ' ', false);
}

std::string feature_identity_json(std::int64_t value) {
    return Json{{"type", "int"}, {"value", value}}.dump(-1, ' ', false);
}

std::string feature_identity_json(std::uint64_t value) {
    return Json{{"type", "int"}, {"value", value}}.dump(-1, ' ', false);
}

std::string feature_identity_json(double value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("feature identity float must be finite");
    return Json{{"type", "float"}, {"value", value}}.dump(-1, ' ', false);
}

std::string feature_identity_none_json() {
    return Json{{"type", "none"}}.dump(-1, ' ', false);
}

std::string feature_identity128_json(std::uint64_t low, std::uint64_t high) {
    return identity128(low, high).dump(-1, ' ', false);
}

std::string feature_custom_identity_json(
    std::string_view codec_name,
    std::int32_t codec_version,
    std::string_view encoded_value_json) {
    if (codec_name.empty() || codec_version <= 0)
        throw std::invalid_argument("custom feature identity requires a codec name and positive version");
    auto value = parse_json(encoded_value_json, "custom feature identity payload");
    validate_payload_free(value);
    return Json{{"type", "custom"},
                {"codec", codec_name},
                {"version", codec_version},
                {"value", std::move(value)}}
        .dump(-1, ' ', false, Json::error_handler_t::strict);
}

std::string feature_tuple_identity_json(
    const std::vector<std::string>& item_identity_json) {
    Json items = Json::array();
    for (const auto& fragment : item_identity_json) {
        auto item = parse_json(fragment, "tuple feature identity item");
        validate_encoded_identity(item, "", 0);
        if (item.at("type") == "custom")
            throw std::invalid_argument("tuple feature identities cannot contain custom identity records");
        items.push_back(std::move(item));
    }
    return Json{{"type", "tuple"}, {"items", std::move(items)}}
        .dump(-1, ' ', false);
}

std::string relation_identity_json(std::string_view value) {
    return Json(value).dump(-1, ' ', false);
}

std::string relation_identity_json(bool value) {
    return Json(value).dump(-1, ' ', false);
}

std::string relation_identity_json(std::int64_t value) {
    return Json(value).dump(-1, ' ', false);
}

std::string relation_identity_json(std::uint64_t value) {
    return Json(value).dump(-1, ' ', false);
}

std::string relation_identity_json(double value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("relation endpoint float identity must be finite");
    return Json(value).dump(-1, ' ', false);
}

std::string relation_identity_none_json() {
    return "null";
}

std::string relation_identity128_json(std::uint64_t low, std::uint64_t high) {
    return Json{{"kind", "cellerator.identity.v1"}, {"low", low}, {"high", high}}
        .dump(-1, ' ', false);
}

std::string relation_tuple_identity_json(
    const std::vector<std::string>& item_identity_json) {
    Json items = Json::array();
    for (const auto& fragment : item_identity_json) {
        auto item = parse_json(fragment, "tuple relation identity item");
        validate_payload_free(item);
        items.push_back(std::move(item));
    }
    return Json{{"kind", "tuple"}, {"items", std::move(items)}}
        .dump(-1, ' ', false);
}

namespace detail {

std::string serialize_prepared_relation_descriptor(
    const PreparedRelation& relation,
    const std::vector<std::string>& source_order_json,
    const std::vector<std::string>& target_order_json) {
    relation.validate_storage();
    if (static_cast<std::int64_t>(source_order_json.size()) != relation.source_count() ||
        static_cast<std::int64_t>(target_order_json.size()) != relation.target_count())
        throw std::invalid_argument("relation endpoint identity counts do not match prepared relation extents");

    Json source_order = Json::array();
    for (const auto& fragment : source_order_json) {
        auto value = parse_json(fragment, "encoded relation source identity");
        validate_payload_free(value);
        source_order.push_back(std::move(value));
    }
    Json target_order = Json::array();
    for (const auto& fragment : target_order_json) {
        auto value = parse_json(fragment, "encoded relation target identity");
        validate_payload_free(value);
        target_order.push_back(std::move(value));
    }

    const auto axis_json = [](const std::optional<biology::AxisDescriptor>& axis) -> Json {
        if (!axis) return nullptr;
        axis->validate();
        const auto identity = [](std::uint64_t low, std::uint64_t high) {
            return Json{{"low", low}, {"high", high}};
        };
        return Json{
            {"domain", identity(axis->domain.low, axis->domain.high)},
            {"order", identity(axis->order.low, axis->order.high)},
            {"geometry", identity(axis->geometry.low, axis->geometry.high)},
            {"partition", identity(axis->partition.low, axis->partition.high)},
            {"extent", axis->extent},
        };
    };

    return Json{
        {"kind", "cellerator.prepared_relation.v1"},
        {"source_order", std::move(source_order)},
        {"target_order", std::move(target_order)},
        {"source_offsets", relation.source_offsets()},
        {"target_offsets", relation.target_offsets()},
        {"source_extent", relation.source_count()},
        {"target_extent", relation.target_count()},
        {"row_ptr", relation.host_row_ptr()},
        {"col_indices", relation.host_col_indices()},
        {"source_axis", axis_json(relation.source_axis())},
        {"target_axis", axis_json(relation.target_axis())},
        {"structural_generation", relation.generation()},
    }.dump(-1, ' ', false, Json::error_handler_t::strict);
}

} // namespace detail

void validate_biological_manifest(
    std::string_view stored_json,
    const BiologicalManifest& expected,
    const std::optional<std::filesystem::path>& weights_path) {
    const auto stored = parse_biological_manifest(stored_json);
    const auto stored_json_value = to_json(stored);
    const auto expected_json_value = to_json(expected);
    if (stored_json_value.at("declarations") != expected_json_value.at("declarations") ||
        stored_json_value.at("relations") != expected_json_value.at("relations") ||
        stored_json_value.at("views") != expected_json_value.at("views"))
        throw std::invalid_argument("stored biological manifest does not match the expected model declaration");
    verify_weight_pair(stored.weights, weights_path);
}

void save_biological_manifest(
    const std::filesystem::path& path,
    const BiologicalManifest& manifest,
    const std::optional<std::filesystem::path>& weights_path) {
    auto sidecar = manifest;
    if (weights_path) sidecar.weights = weight_record(*weights_path);
    const auto payload = serialize_biological_manifest(sidecar) + "\n";
    const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    std::filesystem::create_directories(parent);
    static std::atomic<std::uint64_t> sequence{0};
    auto temporary = path;
    temporary += ".tmp." + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()) + "." +
        std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    try {
        {
            std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
            if (!stream) throw std::runtime_error("cannot create biological manifest temporary file");
            stream.write(payload.data(), static_cast<std::streamsize>(payload.size()));
            stream.flush();
            if (!stream) throw std::runtime_error("cannot write biological manifest");
        }
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

BiologicalManifest load_biological_manifest(
    const std::filesystem::path& path,
    const std::optional<std::filesystem::path>& weights_path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open biological manifest");
    std::string contents((std::istreambuf_iterator<char>(stream)),
                         std::istreambuf_iterator<char>());
    if (!stream.eof() && stream.fail())
        throw std::runtime_error("cannot read biological manifest");
    auto manifest = parse_biological_manifest(contents);
    if (weights_path) verify_weight_pair(manifest.weights, weights_path);
    return manifest;
}

} // namespace cellerator::nn
