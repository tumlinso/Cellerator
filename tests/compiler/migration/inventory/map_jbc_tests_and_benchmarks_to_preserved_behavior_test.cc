#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace {

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.compare(0, prefix.size(), prefix) == 0;
}

bool contains_any(const std::string& value,
                  const std::initializer_list<const char*> needles) {
    for (const auto* needle : needles) {
        if (value.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string basename(const std::string& path) {
    const auto separator = path.find_last_of('/');
    return separator == std::string::npos ? path : path.substr(separator + 1);
}

std::string primary_behavior(const std::string& repository,
                             const std::string& path) {
    const auto component = [&](const std::string& prefix) {
        const auto begin = prefix.size();
        const auto end = path.find('/', begin);
        return path.substr(begin, end - begin);
    };
    if (repository == "CE") {
        if (starts_with(path, "tests/jbc/")) {
            return "CE " + component("tests/jbc/");
        }
        if (starts_with(path, "bench/jbc/multi_extent/")) {
            return "CE multi_extent";
        }
        if (starts_with(path, "bench/jbc/cross_operation/")) {
            return "CE cross_operation";
        }
    }
    if (repository == "CS") {
        if (starts_with(path, "tests/jbc/discovery/")) {
            return "CS discovery/" + component("tests/jbc/discovery/");
        }
        if (starts_with(path, "tests/jbc/grammar/")) {
            return "CS grammar/" + component("tests/jbc/grammar/");
        }
        if (starts_with(path, "tests/jbc/")) {
            return "CS " + component("tests/jbc/");
        }
        if (starts_with(path, "bench/jbc/bicluster/")) {
            return "CS discovery/bicluster";
        }
        if (starts_with(path, "bench/jbc/trajectory/")) {
            return "CS discovery/trajectory";
        }
        if (starts_with(path, "bench/jbc/grammar/")) {
            return "CS grammar/induced";
        }
        if (starts_with(path, "bench/jbc/runtime/")) {
            return "CS runtime";
        }
    }
    return {};
}

std::string primary_form(const std::string& repository,
                         const std::string& path) {
    if (repository == "CE" &&
        path == "bench/jbc/cross_operation/CE-JBC-X08.md") {
        return "non-promotion result";
    }
    if (starts_with(path, "bench/")) {
        return "benchmark fixture";
    }
    const std::string file = basename(path);
    if (file.find("promotion") != std::string::npos ||
        path.find("/validation/atom_ablation") != std::string::npos ||
        path.find("/validation/compiler_ablation") != std::string::npos ||
        path.find("/validation/null_transform") != std::string::npos ||
        file.find("exact_oracle_null_benchmark") != std::string::npos) {
        return "promotion evidence";
    }
    if (contains_any(path, {"fault", "recovery", "invalid", "malformed",
                            "mismatch", "overflow", "duplicate", "stale"})) {
        return "malformed-input test";
    }
    if (path.find("/validation/") != std::string::npos ||
        path.find("/verification/") != std::string::npos ||
        contains_any(path, {"exact_", "stability", "property"})) {
        return "property test";
    }
    return "unit test";
}

void add_files(const std::filesystem::path& root, const std::string& repository,
               const std::string& prefix,
               std::set<std::pair<std::string, std::string>>& files) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file()) {
            const auto relative = std::filesystem::relative(entry.path(), root);
            files.emplace(repository, prefix + relative.generic_string());
        }
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 4,
                "usage: evidence_map_test <receipt> <Cellerator root> <CellShard root>");
        const std::filesystem::path ce(argv[2]);
        const std::filesystem::path cs(argv[3]);
        std::set<std::pair<std::string, std::string>> files;
        add_files(ce / "tests/jbc", "CE", "tests/jbc/", files);
        add_files(ce / "bench/jbc", "CE", "bench/jbc/", files);
        add_files(cs / "tests/jbc", "CS", "tests/jbc/", files);
        add_files(cs / "bench/jbc", "CS", "bench/jbc/", files);
        require(files.size() == 430, "unexpected JBC evidence-set size");

        std::map<std::string, int> behavior_counts;
        std::map<std::string, int> form_counts;
        for (const auto& [repository, path] : files) {
            const auto behavior = primary_behavior(repository, path);
            require(!behavior.empty(), repository + ":" + path + " has no behavior");
            ++behavior_counts[behavior];
            ++form_counts[primary_form(repository, path)];
        }

        const std::map<std::string, int> expected_behaviors{{
            {"CE interfaces", 12}, {"CE fragment", 14},
            {"CE decomposition", 18}, {"CE atom_plane", 10},
            {"CE multi_extent", 10}, {"CE external_cost", 6},
            {"CE resumption", 10}, {"CE cross_operation", 9},
            {"CE verification", 7}, {"CS atom", 20},
            {"CS atom_store", 30}, {"CS basis", 17},
            {"CS certification", 16}, {"CS composition", 24},
            {"CS evidence", 16}, {"CS global_ir", 14},
            {"CS grammar/explicit", 10}, {"CS grammar/induced", 12},
            {"CS partial", 18}, {"CS runtime", 26},
            {"CS superatom", 8}, {"CS validation", 36},
            {"CS discovery/bicluster", 9}, {"CS discovery/co_support", 11},
            {"CS discovery/factor_topic", 6}, {"CS discovery/motif", 8},
            {"CS discovery/multimodal", 10},
            {"CS discovery/operation_trace", 8},
            {"CS discovery/overlap", 6},
            {"CS discovery/sequence_compat", 6},
            {"CS discovery/support_signature", 10},
            {"CS discovery/trajectory", 13},
        }};
        require(behavior_counts == expected_behaviors,
                "protected-behavior counts changed");

        const std::map<std::string, int> expected_forms{{
            {"unit test", 352}, {"property test", 48},
            {"malformed-input test", 8}, {"promotion evidence", 14},
            {"benchmark fixture", 7}, {"non-promotion result", 1},
        }};
        require(form_counts == expected_forms, "primary evidence-form counts changed");

        std::ifstream receipt_stream(argv[1]);
        require(receipt_stream.good(), "could not open evidence-map receipt");
        const std::string receipt((std::istreambuf_iterator<char>(receipt_stream)),
                                  std::istreambuf_iterator<char>());
        for (const auto& [form, count] : expected_forms) {
            require(receipt.find("| " + form + " | " + std::to_string(count) + " |") !=
                        std::string::npos,
                    "receipt omits evidence form: " + form);
        }

        constexpr std::array<const char*, 31> reusable_subsystems{{
            "CE semantic interfaces", "CE atom-fragment preparation",
            "CE decomposition catalog", "CE atom/value planes",
            "CE multi-extent binding and candidate",
            "CE external complete-cost exchange", "CE lowering resumption",
            "CE aggregate/package surface", "CS atom model", "CS evidence atlas",
            "CS exact certification", "CS support-signature discovery",
            "CS co-support discovery", "CS bicluster discovery",
            "CS overlap discovery", "CS motif discovery",
            "CS factor/topic discovery", "CS operation-trace discovery",
            "CS trajectory discovery", "CS multimodal discovery",
            "CS sequence compatibility discovery", "CS composition",
            "CS explicit grammar", "CS induced grammar experiment",
            "CS basis selection", "CS superatoms", "CS persistent partials",
            "CS global graph and schedule", "CS atom store", "CS runtime v2",
            "CS integrated validation/package matrix",
        }};
        for (const auto* subsystem : reusable_subsystems) {
            require(receipt.find(subsystem) != std::string::npos,
                    std::string("reusable subsystem lacks evidence mapping: ") + subsystem);
        }
        require(std::filesystem::exists(
                    cs / "docs/JBC/evidence/biological_novelty_readiness.md"),
                "missing non-promotion readiness evidence");

        std::cout << "mapped 430 JBC evidence files to 32 behaviors and 6 forms\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
