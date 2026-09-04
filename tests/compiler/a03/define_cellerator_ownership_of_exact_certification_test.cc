#include <Cellerator/compiler/migration/define_cellerator_ownership_of_exact_certification_v1.hh>

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace migration = Cellerator::compiler::migration;

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
std::string read(const std::string& path) {
    std::ifstream stream(path);
    require(bool(stream), "cannot read " + path);
    std::ostringstream text; text << stream.rdbuf(); return text.str();
}
std::string quote(const std::string& value) {
    std::string result = "'";
    for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c);
    return result + "'";
}
std::string output(const std::string& command) {
    std::array<char, 2048> buffer{}; std::string result;
    FILE* stream = popen(command.c_str(), "r"); require(stream, "popen failed");
    while (fgets(buffer.data(), int(buffer.size()), stream)) result += buffer.data();
    require(pclose(stream) == 0, "git inventory failed"); return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 4, "usage: test CELLSHARD_GIT COMMIT CELLERATOR_ROOT");
        const std::string inventory = output("git --git-dir=" + quote(argv[1])
            + " ls-tree -r --name-only " + quote(argv[2])
            + " -- include/CellShard/compiler/certification");
        std::set<std::string> names;
        std::istringstream lines(inventory);
        for (std::string line; std::getline(lines, line);) {
            names.insert(line.substr(line.find_last_of('/') + 1));
        }
        require(names.size() == migration::exact_certification_map_v1.size(),
                "certification source count changed");
        for (const auto& row : migration::exact_certification_map_v1) {
            require(names.erase(std::string(row.source_header)) == 1,
                    "missing or duplicate certification mapping");
            require(!row.planning_ir_contract.empty(), "empty Planning IR target");
        }
        require(names.empty(), "unmapped certification header");

        const std::string root = argv[3];
        const auto logical = read(root + "/include/Cellerator/execution/joint_compiler/logical_coverage_v1.hh");
        const auto roles = read(root + "/include/Cellerator/execution/joint_compiler/coverage_roles_v1.hh");
        const auto cover = read(root + "/include/Cellerator/geometry/relation_cover.hh");
        require(logical.find("duplicate_member") != std::string::npos,
                "current exact coverage lacks duplicate proof");
        require(roles.find("missing_exact_certification") != std::string::npos,
                "current coverage role lacks certification guard");
        require(cover.find("duplicate_logical_edge") != std::string::npos,
                "relation cover lacks exact edge duplicate proof");

        migration::exact_certification_prerequisites_v1 all{
            true, true, true, true, true, true, true};
        require(migration::may_certify_execution_v1(all), "complete proof rejected");
        for (int omitted = 0; omitted < 7; ++omitted) {
            auto incomplete = all;
            switch (omitted) {
            case 0: incomplete.canonical_identities = false; break;
            case 1: incomplete.sorted_unique_members = false; break;
            case 2: incomplete.complete_relation_edges = false; break;
            case 3: incomplete.contribution_owners = false; break;
            case 4: incomplete.residual_accounted = false; break;
            case 5: incomplete.inverse_recovery = false; break;
            case 6: incomplete.dependency_closure = false; break;
            }
            require(!migration::may_certify_execution_v1(incomplete),
                    "incomplete proof accepted");
        }
        std::cout << "validated 16 exact-certification migrations\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
