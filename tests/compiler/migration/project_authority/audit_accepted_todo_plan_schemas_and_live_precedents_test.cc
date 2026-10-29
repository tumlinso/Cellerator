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
          "audit_accepted_todo_plan_schemas_and_live_precedents.md";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "unable to open plan schema audit: " << path << '\n';
        return 1;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string document = buffer.str();

    bool valid = true;
    valid &= require_text(document, "database migration version: 10",
        "database migration version");
    valid &= require_text(document, "accepted legacy plan schema version: 2",
        "legacy plan schema");
    valid &= require_text(document,
        "accepted first-class workflow plan schema version: 3",
        "workflow plan schema");
    valid &= require_text(document, "ce-exop-plan.json` | 3 | valid",
        "CE-EXOP precedent validation");
    valid &= require_text(document, ".todo-orchestrator/ce-ptr-plan.json` | 3 | valid",
        "CE-PTR precedent validation");
    valid &= require_text(document, "557 tasks, 44 checkpoints, 512 gates",
        "Part One precedent validation");
    valid &= require_text(document,
        "`contract_split`. A workspace `integration_task_id`",
        "workspace modes");
    valid &= require_text(document, "command` and `benchmark` gates",
        "observed gate shapes");
    valid &= require_text(document,
        "declared interface owner. Project Control hashes",
        "interface publication ownership");
    valid &= require_text(document,
        "interface back to draft",
        "interface lifecycle preservation");
    valid &= require_text(document, "is an auxiliary catalog, not a native plan root",
        "JBC interface catalog classification");
    valid &= require_text(document, "input rather than misclassified",
        "invalid AppleDouble sidecar classification");

    return valid ? 0 : 1;
}
