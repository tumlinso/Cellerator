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
          "capture_the_reconciled_project_control_authority_cursor.md";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "unable to open authority cursor receipt: " << path << '\n';
        return 1;
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string document = buffer.str();

    bool valid = true;
    valid &= require_text(document,
        "\"project_uuid\": \"0ccaac37-dbbf-448e-a5f8-def197a70aba\"",
        "project UUID");
    valid &= require_text(document, "\"todo_revision\": 3904",
        "Todo revision");
    valid &= require_text(document, "\"workflow_revision\": 3904",
        "workflow revision");
    valid &= require_text(document,
        "\"todo_semantic_fingerprint\": \"fb53d368f34add25316a3aab81251f9dd19d43dcc5c4a672f1054cd271de3837\"",
        "Todo semantic fingerprint");
    valid &= require_text(document,
        "\"workflow_semantic_fingerprint\": \"fb53d368f34add25316a3aab81251f9dd19d43dcc5c4a672f1054cd271de3837\"",
        "workflow semantic fingerprint");
    valid &= require_text(document, "\"run_id\": \"CE-CCP1-RUN-V1\"",
        "active run");
    valid &= require_text(document, "\"task_id\": \"CE-CCP1-A01-001\"",
        "active claim");
    valid &= require_text(document,
        "\"project_overview_observed_at\": \"2026-09-03T19:34:50Z\"",
        "overview observation time");
    valid &= require_text(document,
        "\"agent_status_observed_at\": \"2026-09-03T19:34:54Z\"",
        "claim observation time");
    valid &= require_text(document, "\"missing_provider_fields\": []",
        "explicit missing-provider field list");
    valid &= require_text(document, "\"todo_revision_before\": 3904",
        "pre-read Todo revision");
    valid &= require_text(document, "\"todo_revision_after\": 3904",
        "post-read Todo revision");
    valid &= require_text(document, "\"semantic_fingerprint_unchanged\": true",
        "read-only semantic validation");
    valid &= require_text(document, "transient observer caveat",
        "provider outage caveat");

    return valid ? 0 : 1;
}
