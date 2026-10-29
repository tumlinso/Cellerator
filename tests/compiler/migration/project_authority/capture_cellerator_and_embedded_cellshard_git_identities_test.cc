#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

bool require_text(const std::string& document,
                  const std::string& expected,
                  const char* description) {
    if (document.find(expected) != std::string::npos) {
        return true;
    }
    std::cerr << "missing " << description << ": " << expected << '\n';
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1
        ? argv[1]
        : "planning/cellerator-compiler-preledger-v1/evidence/"
          "capture_cellerator_and_embedded_cellshard_git_identities.md";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "unable to open Git identity receipt: " << path << '\n';
        return 1;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string document = buffer.str();

    bool valid = true;
    valid &= require_text(document, "branch: `main`", "canonical branch");
    valid &= require_text(document,
        "HEAD: `31e491ed29de0fcde70259cbeab8c5c7ad353485`",
        "canonical HEAD");
    valid &= require_text(document, "clean: `false`", "reported dirty checkout");
    valid &= require_text(document,
        "gitlink: `b9749ad3e5146a04f847533d8c6f1a54146aed20`",
        "CellShard gitlink");
    valid &= require_text(document, "embedded working-tree identity: unavailable",
        "uninitialized embedded checkout");
    valid &= require_text(document, "This equality does not make the two repository reads atomic",
        "non-atomic observation warning");
    valid &= require_text(document, "`ce-ccp1-l-a01`", "A01 worktree");
    valid &= require_text(document, "`ce-ccp1-l-a02`", "A02 worktree");
    valid &= require_text(document, "`ce-ccp1-l-a03`", "A03 worktree");
    valid &= require_text(document, "`ce-ccp1-l-a04`", "A04 worktree");
    valid &= require_text(document, "`ce-jbc-l-crossop`", "JBC crossop worktree");
    valid &= require_text(document, "`ce-jbc-l-decomposition`", "JBC decomposition worktree");
    valid &= require_text(document, "`ce-jbc-l-external-cost`", "JBC external-cost worktree");
    valid &= require_text(document, "`ce-jbc-l-fragment`", "JBC fragment worktree");
    valid &= require_text(document, "`ce-jbc-l-interfaces`", "JBC interfaces worktree");
    valid &= require_text(document, "`ce-jbc-l-multiatom`", "JBC multiatom worktree");
    valid &= require_text(document, "`ce-jbc-l-planes`", "JBC planes worktree");
    valid &= require_text(document, "`ce-jbc-l-resumption`", "JBC resumption worktree");
    valid &= require_text(document, "`ce-jbc-l-verify-integrate`", "JBC integration worktree");
    valid &= require_text(document, "`/tmp/ce-jbc-main-delivery-20260901`",
        "prunable registered worktree");
    valid &= require_text(document, "dirty: Project Control projections",
        "reported dirty JBC integration worktree");
    valid &= require_text(document, "path unavailable, cleanliness unavailable",
        "unavailable cleanliness state");

    return valid ? 0 : 1;
}
