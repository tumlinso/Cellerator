#include <Cellerator/compiler/migration/define_cellerator_ownership_of_evidence_and_proposal_dis_v1.hh>

#include <array>
#include <cstdio>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace migration = Cellerator::compiler::migration;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string command_output(const std::string& command) {
    std::array<char, 4096> buffer{};
    std::string output;
    FILE* stream = popen(command.c_str(), "r");
    if (stream == nullptr) throw std::runtime_error("cannot inspect source tree");
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), stream) != nullptr)
        output.append(buffer.data());
    require(pclose(stream) == 0, "source inventory command failed");
    return output;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char character : value) {
        if (character == '\'') quoted += "'\\''";
        else quoted += character;
    }
    return quoted + "'";
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: test CELLSHARD_GIT_DIR SOURCE_COMMIT");
        const std::string commit = argv[2];
        require(commit == "b9749ad3e5146a04f847533d8c6f1a54146aed20",
                "unexpected CellShard source commit");
        const std::string command = "git --git-dir=" + shell_quote(argv[1])
            + " ls-tree -r --name-only " + shell_quote(commit)
            + " -- include/CellShard/compiler/evidence"
              " include/CellShard/compiler/discovery src/compiler/evidence"
              " src/compiler/discovery";
        const std::string inventory = command_output(command);
        require(!inventory.empty(), "empty evidence/discovery source inventory");

        std::set<std::string_view> prefixes;
        for (const auto& mapping : migration::evidence_ownership_map_v1) {
            require(!mapping.source_prefix.empty(), "empty source prefix");
            require(mapping.destination_namespace.starts_with(
                        "Cellerator::compiler::profile::"),
                    "destination is outside Cellerator profile/planning ownership");
            require(prefixes.insert(mapping.source_prefix).second,
                    "duplicate source prefix");
        }

        std::size_t source_count = 0;
        std::size_t begin = 0;
        while (begin < inventory.size()) {
            const std::size_t end = inventory.find('\n', begin);
            const std::string_view source(inventory.data() + begin,
                (end == std::string::npos ? inventory.size() : end) - begin);
            if (!source.empty()) {
                ++source_count;
                bool covered = false;
                for (const auto& mapping : migration::evidence_ownership_map_v1)
                    covered = covered || source.starts_with(mapping.source_prefix);
                require(covered, "unmapped evidence-producing source: "
                                   + std::string(source));
            }
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        require(source_count >= 100, "evidence/discovery inventory unexpectedly small");

        constexpr migration::proposal_evidence_identity_v1 proposal{
            11, 22, 33, 44, 55};
        static_assert(migration::valid_proposal_evidence_identity_v1(proposal));
        static_assert(!migration::authorizes_execution(proposal));
        require(sizeof(proposal) == 40,
                "proposal identity acquired non-semantic/address storage");

        std::cout << "validated " << source_count
                  << " evidence/discovery sources across "
                  << migration::evidence_ownership_map_v1.size()
                  << " ownership prefixes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
