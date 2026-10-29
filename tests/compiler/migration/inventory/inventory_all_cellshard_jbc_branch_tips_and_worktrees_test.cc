#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct branch_record {
    const char* name;
    const char* tip;
    int changed_path_count;
    bool has_remote;
};

constexpr const char* common_base =
    "7762a5925fe18b2ca45ab8a436f3461804ed2ad9";

constexpr std::array<branch_record, 24> branches{{
    {"jbc/cs-atom-core", "09e324f4bff4b759400e6489a0625154d388d682", 41, true},
    {"jbc/cs-basis", "0927aebcbc59571dad02d82ec64548dfc3364f00", 34, true},
    {"jbc/cs-certification", "2df82289e03bc8763ceeef8433828c2a82a7dae0", 32, true},
    {"jbc/cs-composition", "5030e73ba63ef03df4d1691da866d4eadb432cf1", 48, true},
    {"jbc/cs-disc-bicluster", "483dddb0ab56dbf76d60db20149bb8fa5f4f6f0a", 17, true},
    {"jbc/cs-disc-cosupport", "68010879022a0f32ad95607d885d6f8b0444a7ad", 21, true},
    {"jbc/cs-disc-factor", "0e11cda6dc5f6af20668e86f7a3d6bebce50c78d", 12, true},
    {"jbc/cs-disc-motif", "e1487965d0ee3ced30c1970ec00453946ac57716", 15, true},
    {"jbc/cs-disc-multimodal", "8b80a05784d4ac86a5f322ee3e4b4b4d82191d65", 20, true},
    {"jbc/cs-disc-optrace", "38f15b2f7275d2fac004537d2837bce7a85f237b", 16, true},
    {"jbc/cs-disc-overlap", "3b4d9184f2f5ad14576ef5ee5c83be01e7976688", 12, true},
    {"jbc/cs-disc-sequence", "30012b4860793e6f06a42df7f07481c4434a6531", 12, true},
    {"jbc/cs-disc-signature", "d064cb9d950693fcf33e6d8dc6cd602c6a0aa021", 19, true},
    {"jbc/cs-disc-trajectory", "e1816a2498b42c1fb02007a845d6d704f99bb3e9", 25, true},
    {"jbc/cs-evidence-core", "2115d742a65e25f5ee034e1d3f824fb9b8366ec1", 35, true},
    {"jbc/cs-explicit-grammar", "764f643bbbd9a97eb4ada84c076158df18d04cde", 20, true},
    {"jbc/cs-global-ir", "b4dd91db9700a092159dc2f550eb8300826fa8bc", 33, true},
    {"jbc/cs-induced-grammar", "f1774b1c5ea831081e85f7a382145bfff144b6f1", 22, true},
    {"jbc/cs-integrate", "9f6527276ae53367fe6b699bcdf48467d40f8ab3", 661, true},
    {"jbc/cs-partials", "a971bc28d275842b201417c8279deb627070faf3", 35, true},
    {"jbc/cs-persistence", "27de0b3a1b2083793b678dbf6b5c495efec081ea", 73, true},
    {"jbc/cs-projections-final", "45aa4bb5ccb4d98a5d54b76663a9d5d05a620591", 318, false},
    {"jbc/cs-runtime", "da73cf20031d9d56c74da1f11276dfe7560725f1", 63, true},
    {"jbc/cs-superatom", "ceb108c214b366af60d29d92f6ca61cca7c85154", 15, true},
}};

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (const char character : value) {
        quoted += character == '\'' ? "'\\''" : std::string(1, character);
    }
    return quoted + "'";
}

std::string run(const std::string& command) {
    std::array<char, 4096> buffer{};
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("could not execute: " + command);
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    if (pclose(pipe) != 0) {
        throw std::runtime_error("command failed: " + command);
    }
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
        output.pop_back();
    }
    return output;
}

int line_count(const std::string& value) {
    if (value.empty()) {
        return 0;
    }
    int count = 1;
    for (const char character : value) {
        count += character == '\n' ? 1 : 0;
    }
    return count;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: inventory_test <CellShard repository>");
        const std::string git = "git -C " + shell_quote(argv[1]) + " ";

        require(run(git + "rev-parse main") ==
                    "b9749ad3e5146a04f847533d8c6f1a54146aed20",
                "unexpected CellShard main");
        require(run(git + "rev-parse origin/main") == run(git + "rev-parse main"),
                "CellShard main and origin/main differ");

        const std::string refs =
            run(git + "for-each-ref --format='%(refname:short)' 'refs/heads/jbc/*'");
        require(line_count(refs) == static_cast<int>(branches.size()),
                "unexpected local jbc branch count");

        const std::string worktrees = run(git + "worktree list --porcelain");
        int live_jbc_worktrees = 0;
        for (const auto& branch : branches) {
            require(run(git + "rev-parse " + shell_quote(branch.name)) == branch.tip,
                    std::string("tip mismatch: ") + branch.name);
            require(run(git + "merge-base main " + shell_quote(branch.name)) == branch.tip,
                    std::string("tip is not already merged: ") + branch.name);

            const auto changed_paths = run(
                git + "diff --name-only " + common_base + ".." + shell_quote(branch.name));
            require(line_count(changed_paths) == branch.changed_path_count,
                    std::string("changed-path count mismatch: ") + branch.name);

            const std::string remote_command = git + "rev-parse --verify " +
                shell_quote(std::string("refs/remotes/origin/") + branch.name) +
                " 2>/dev/null || printf absent";
            const auto remote_tip = run("sh -c " + shell_quote(remote_command));
            if (branch.has_remote) {
                require(remote_tip == branch.tip,
                        std::string("remote tip mismatch: ") + branch.name);
            } else {
                require(remote_tip == "absent",
                        std::string("unexpected remote branch: ") + branch.name);
            }

            const std::string marker =
                std::string("branch refs/heads/") + branch.name;
            const bool has_worktree = worktrees.find(marker) != std::string::npos;
            if (std::string(branch.name) == "jbc/cs-projections-final") {
                require(!has_worktree, "projection-final unexpectedly has a worktree");
            } else {
                require(has_worktree, std::string("missing worktree: ") + branch.name);
                const auto marker_position = worktrees.find(marker);
                const auto path_start = worktrees.rfind("worktree ", marker_position);
                require(path_start != std::string::npos,
                        std::string("missing worktree path: ") + branch.name);
                const auto path_end = worktrees.find('\n', path_start);
                const auto path = worktrees.substr(path_start + 9,
                                                   path_end - path_start - 9);
                require(run("git -C " + shell_quote(path) +
                            " status --porcelain=v1 --untracked-files=all").empty(),
                        std::string("dirty worktree: ") + branch.name);
                ++live_jbc_worktrees;
            }
        }

        require(live_jbc_worktrees == 23, "unexpected live JBC worktree count");
        require(worktrees.find("/tmp/cs-jbc-main-delivery-20260901") != std::string::npos &&
                    worktrees.find("prunable") != std::string::npos,
                "missing prunable delivery worktree record");
        require(run(git + "status --porcelain=v1 --untracked-files=all").empty(),
                "CellShard main worktree is dirty");

        std::cout << "validated 24 reachable JBC tips, 23 live worktrees, and exact changed-path counts\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
