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
          "audit_program_identifier_and_artifact_name_collisions.md";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "unable to open collision audit: " << path << '\n';
        return 1;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string document = buffer.str();

    bool valid = true;
    valid &= require_text(document, "zero unresolved identifier or artifact-name",
        "zero-collision disposition");
    valid &= require_text(document, "| Tasks | 557 | 557 | 0 |",
        "task uniqueness");
    valid &= require_text(document, "| Interfaces | 41 | 41 | 0 |",
        "interface uniqueness");
    valid &= require_text(document, "| Checkpoints | 44 | 44 | 0 |",
        "checkpoint uniqueness");
    valid &= require_text(document, "| Barriers | 10 | 10 | 0 |",
        "barrier uniqueness");
    valid &= require_text(document, "| Lanes | 38 | 38 | 0 |",
        "lane uniqueness");
    valid &= require_text(document, "| Runs | 1 | 1 | 0 |",
        "run uniqueness");
    valid &= require_text(document, "| Locks | 11 | 11 | 0 |",
        "lock uniqueness");
    valid &= require_text(document, "| Gates | 512 | 512 | 0 |",
        "gate uniqueness");
    valid &= require_text(document, "Exact identifier intersections across those namespaces: `0`",
        "cross-namespace uniqueness");
    valid &= require_text(document, "All seven repeated paths are resolved",
        "artifact path overlap disposition");
    valid &= require_text(document, "`CELLPK01`", "CPK1 magic");
    valid &= require_text(document, "`CELLCSG1`", "CSG1 magic");
    valid &= require_text(document, "`CELLEX02`", "CPE2 magic");
    valid &= require_text(document, "`.ceprofile`", "profile suffix");
    valid &= require_text(document, "`CE-CCP1-I15-CEIR-TEXT`",
        "CEIR artifact interface");
    valid &= require_text(document, "No CEIR binary magic or mandatory",
        "deferred CEIR identity freeze");
    valid &= require_text(document, "a filename suffix\n  is never the semantic switch",
        "source activation rule");
    valid &= require_text(document, "no Part Two format is reserved",
        "Part Two boundary");

    return valid ? 0 : 1;
}
