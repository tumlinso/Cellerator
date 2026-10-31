#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Cellerator/nn/manifest.hh>
#include <Cellerator/biology/feature_binding.hh>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

using namespace cellerator::nn;

namespace {

BiologicalManifest sample_manifest() {
    BiologicalManifest manifest;
    manifest.declarations.push_back({
        "",
        "feature_binding",
        R"({"domain":"gene","identities":[{"type":"custom","codec":"example.gene-id","version":3,"value":{"id":"g-1"}}],"model_indices":[7],"canonical_indices":[0],"aliases":[],"metadata":{"reference":"example-v1"},"unknown_policy":"error","unknown_model_index":null,"model_capacity":8})",
    });
    manifest.relations.push_back({
        "regulation",
        R"({"kind":"cellerator.prepared_relation.v1","source_order":[{"type":"str","value":"g-1"}],"target_order":[{"type":"str","value":"program-1"}],"source_offsets":[0,1],"target_offsets":[0,1],"source_extent":1,"target_extent":1,"row_ptr":[0,1],"col_indices":[0],"source_axis":null,"target_axis":null,"structural_generation":5})",
    });
    BiologicalViewDeclaration view;
    view.module_path = "readouts";
    view.name = "atac";
    view.output_binding_json =
        R"({"domain":"peak","identities":[{"type":"str","value":"peak-a"}],"model_indices":[12],"canonical_indices":null,"aliases":[],"metadata":{"reference":"hg38"},"unknown_policy":"error","unknown_model_index":null,"model_capacity":16})";
    view.queryable = true;
    view.source_kind = BiologicalViewSourceKind::mapping_key;
    view.source_key = "regulatory";
    manifest.views.push_back(std::move(view));
    return manifest;
}

template<class F>
void rejects(F&& function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error(message);
}

std::filesystem::path temporary_directory() {
    auto path = std::filesystem::temp_directory_path();
    path /= "cellerator-nn-manifest-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(path);
    return path;
}

void shared_v1_json_roundtrip() {
    const auto manifest = sample_manifest();
    const auto json = serialize_biological_manifest(manifest);
    assert(json.find("cellerator-biological-model-manifest") != std::string::npos);
    assert(json.find("\"version\":1") != std::string::npos);
    assert(json.find("\"relations\":[{\"descriptor\":") != std::string::npos);
    assert(json.find("\"source_id\":null") != std::string::npos);
    assert(json.find("\"weights\":null") != std::string::npos);

    const auto parsed = parse_biological_manifest(json);
    assert(serialize_biological_manifest(parsed) == json);
    validate_biological_manifest(json, manifest);

    auto changed = manifest;
    changed.declarations[0].descriptor_json =
        R"({"domain":"gene","identities":[{"type":"custom","codec":"example.gene-id","version":3,"value":{"id":"g-2"}}],"model_indices":[7],"canonical_indices":[0],"aliases":[],"metadata":{"reference":"example-v1"},"unknown_policy":"error","unknown_model_index":null,"model_capacity":8})";
    rejects([&] { validate_biological_manifest(json, changed); },
            "changed biological declarations must reject");
}

void native_feature_binding_uses_the_python_wire_schema() {
    using cellerator::biology::FeatureBinding;
    using cellerator::biology::FeatureBindingOptions;
    FeatureBindingOptions options;
    options.domain = "gene";
    options.canonical_indices = {0};
    options.model_capacity = 8;
    options.metadata = {{"reference", "example-v1"}};
    auto binding = FeatureBinding<std::string>::from_ids(
        {"gene-a"}, {7}, std::move(options));
    const auto descriptor = binding.to_descriptor(
        "cellerator.json", 1,
        [](const std::string& identity) {
            return feature_identity_json(identity);
        });
    const auto json = serialize_feature_binding_descriptor(descriptor);
    assert(json.find("\"schema_version\":1") != std::string::npos);
    assert(json.find("\"identities\":[{\"type\":\"str\",\"value\":\"gene-a\"}]")
           != std::string::npos);
    assert(json.find("\"model_indices\":[7]") != std::string::npos);
    assert(json.find("\"canonical_indices\":[0]") != std::string::npos);
    assert(json.find("identity_codec_name") == std::string::npos);

    struct CustomIdentity {
        int value = 0;
        bool operator==(const CustomIdentity&) const = default;
    };
    auto custom_binding = FeatureBinding<CustomIdentity>::from_ids({{5}}, {2});
    const auto custom_descriptor = custom_binding.to_descriptor(
        "example.custom-id", 3,
        [](const CustomIdentity& identity) {
            const auto payload = std::string("{\"id\":")
                                 + std::to_string(identity.value) + "}";
            return feature_custom_identity_json("example.custom-id", 3, payload);
        });
    const auto custom_json = serialize_feature_binding_descriptor(custom_descriptor);
    assert(custom_json.find("\"codec\":\"example.custom-id\"") != std::string::npos);
    assert(custom_json.find("\"version\":3") != std::string::npos);
    const auto tuple = feature_tuple_identity_json({
        feature_identity_json("chromosome-1"),
        feature_identity_json(std::int64_t{4}),
    });
    assert(tuple.find("\"type\":\"tuple\"") != std::string::npos);

    rejects([&] {
        auto bad = custom_descriptor;
        bad.identities[0] =
            R"({"type":"custom","codec":"other.codec","version":3,"value":{"id":5}})";
        (void)serialize_feature_binding_descriptor(bad);
    }, "custom identity must match an explicit codec declaration");
}

void prepared_relations_require_and_serialize_identity_endpoints() {
    using cellerator::biology::OccurrenceCoordinates;
    using cellerator::nn::CsrSupport;
    using cellerator::nn::PreparedRelation;
    const auto source = OccurrenceCoordinates<std::string>::from_ids(
        {"gene-a", "gene-a", "gene-b"}, {0, 2, 3}, {}, 4);
    const auto target = OccurrenceCoordinates<std::string>::from_ids(
        {"program-a", "program-b"}, {0, 1, 2}, {}, 4);
    const auto relation = PreparedRelation::from_csr(
        CsrSupport{{0, 2, 3}, {0, 0, 2}},
        source.order_token(), target.order_token(), 9);

    const auto descriptor = serialize_relation_descriptor(
        relation, source, target, {}, {}, 9, at::Device(at::kCPU));
    assert(descriptor.find("\"source_order\":[\"gene-a\",\"gene-a\",\"gene-b\"]")
           != std::string::npos);
    assert(descriptor.find("\"target_order\":[\"program-a\",\"program-b\"]")
           != std::string::npos);
    assert(descriptor.find("\"row_ptr\":[0,2,3]") != std::string::npos);
    assert(descriptor.find("\"col_indices\":[0,0,2]") != std::string::npos);
    assert(descriptor.find("\"structural_generation\":9") != std::string::npos);

    const auto changed_order = OccurrenceCoordinates<std::string>::from_ids(
        {"gene-b", "gene-a", "gene-a"}, {0, 2, 3}, {}, 4);
    rejects([&] {
        (void)serialize_relation_descriptor(relation, changed_order, target);
    }, "relation endpoints must match the prepared order tokens");
    rejects([&] {
        (void)serialize_relation_descriptor(
            relation, source, target, {}, {}, 8, at::Device(at::kCPU));
    }, "relation structural generation must match the expected generation");

    struct CustomEndpoint {
        int value = 0;
        bool operator==(const CustomEndpoint&) const = default;
    };
    const auto custom_source = OccurrenceCoordinates<CustomEndpoint>::from_ids(
        {{1}}, {}, {}, 0);
    const auto custom_target = OccurrenceCoordinates<CustomEndpoint>::from_ids(
        {{2}}, {}, {}, 0);
    const auto custom_relation = PreparedRelation::from_csr(
        CsrSupport{{0, 1}, {0}}, custom_source.order_token(),
        custom_target.order_token(), 3);
    rejects([&] {
        (void)serialize_relation_descriptor(custom_relation, custom_source, custom_target);
    }, "custom relation endpoint identities require an explicit codec");
    const auto custom_descriptor = serialize_relation_descriptor(
        custom_relation, custom_source, custom_target,
        [](const CustomEndpoint& value) {
            return relation_identity_json(std::int64_t{value.value});
        },
        [](const CustomEndpoint& value) {
            return relation_identity_json(std::int64_t{value.value});
        });
    assert(custom_descriptor.find("\"source_order\":[1]") != std::string::npos);

    using NestedCustom = std::tuple<std::string, CustomEndpoint>;
    const auto nested_source = OccurrenceCoordinates<NestedCustom>::from_ids(
        {NestedCustom{"gene", CustomEndpoint{7}}}, {}, {}, 0);
    const auto nested_relation = PreparedRelation::from_csr(
        CsrSupport{{0, 1}, {0}}, nested_source.order_token(),
        custom_target.order_token(), 3);
    const auto nested_descriptor = serialize_relation_descriptor(
        nested_relation, nested_source, custom_target,
        [](const CustomEndpoint& value) {
            return relation_identity_json(std::int64_t{value.value});
        },
        [](const CustomEndpoint& value) {
            return relation_identity_json(std::int64_t{value.value});
        });
    assert(nested_descriptor.find(
        "\"source_order\":[{\"items\":[\"gene\",7],\"kind\":\"tuple\"}]")
        != std::string::npos);
}

void sidecar_hash_and_parser_guards() {
    const auto directory = temporary_directory();
    try {
        const auto weights = directory / "weights.pt";
        const auto sidecar = directory / "weights.bio.json";
        {
            std::ofstream stream(weights, std::ios::binary);
            stream << "torch archive bytes";
        }
        const auto manifest = sample_manifest();
        save_biological_manifest(sidecar, manifest, weights);
        const auto loaded = load_biological_manifest(sidecar);
        assert(loaded.weights.has_value());
        assert(loaded.weights->filename == "weights.pt");
        assert(loaded.weights->sha256.size() == 64);
        validate_biological_manifest(
            serialize_biological_manifest(loaded), manifest, weights);

        {
            std::ofstream stream(weights, std::ios::binary | std::ios::trunc);
            stream << "different bytes";
        }
        rejects([&] {
            validate_biological_manifest(
                serialize_biological_manifest(loaded), manifest, weights);
        }, "changed paired weight bytes must reject");

        rejects([&] {
            (void)parse_biological_manifest(
                R"({"format":"cellerator-biological-model-manifest","version":1,"version":1,"declarations":[],"relations":[],"views":[],"weights":null})");
        }, "duplicate JSON keys must reject");

        auto bad = sample_manifest();
        bad.declarations[0].descriptor_json = R"({"history":["not a declaration"]})";
        rejects([&] { (void)serialize_biological_manifest(bad); },
                "history payload must not be serialized into biological meaning");
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
}

} // namespace

int main() {
    shared_v1_json_roundtrip();
    native_feature_binding_uses_the_python_wire_schema();
    prepared_relations_require_and_serialize_identity_endpoints();
    sidecar_hash_and_parser_guards();
}
