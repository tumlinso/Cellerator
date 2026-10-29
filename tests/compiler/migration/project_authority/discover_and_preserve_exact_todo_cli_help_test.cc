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
          "discover_and_preserve_exact_todo_cli_help.md";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "unable to open CLI discovery receipt: " << path << '\n';
        return 1;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string document = buffer.str();

    bool valid = true;
    valid &= require_text(document,
        "project-control plan [-h] {compile,validate,apply}",
        "Project Control plan help");
    valid &= require_text(document,
        "project-control plan validate [-h] --project PROJECT --file FILE",
        "Project Control validate help");
    valid &= require_text(document,
        "project-control plan apply [-h] --project PROJECT --file FILE",
        "Project Control apply help");
    valid &= require_text(document, "zero `would_add` entries",
        "collision preview result");
    valid &= require_text(document, "Revision remained 3922",
        "read-only verification");
    valid &= require_text(document, "todo plan diff [-h]",
        "Todo collision audit help");
    valid &= require_text(document, "todo semantic workflow [-h]",
        "workflow verification help");
    valid &= require_text(document, "no separate `plan activate-run`",
        "run activation discovery");
    valid &= require_text(document, "--confirm PREPARE-RUN-WORKSPACES",
        "workspace preparation confirmation");
    valid &= require_text(document, "There is no separate `claim_task` tool",
        "claim entrypoint contract");
    valid &= require_text(document, "candidate that might write was executed merely",
        "discovery safety result");

    return valid ? 0 : 1;
}
