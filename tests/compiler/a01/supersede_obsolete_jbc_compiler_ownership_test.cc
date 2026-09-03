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
          "supersede_obsolete_jbc_compiler_ownership.md";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "unable to open JBC supersession record: " << path << '\n';
        return 1;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string document = buffer.str();

    bool valid = true;
    valid &= require_text(document, "Evidence and proposal discovery",
        "discovery supersession");
    valid &= require_text(document, "Independent exact certification",
        "certification supersession");
    valid &= require_text(document, "Typed composition and grammar",
        "grammar supersession");
    valid &= require_text(document, "Basis, no-basis, and superatoms",
        "basis supersession");
    valid &= require_text(document, "Global operation/program IR",
        "global IR supersession");
    valid &= require_text(document, "Portable schedule compilation",
        "portable schedule supersession");
    valid &= require_text(document, "CellShard boundary retained",
        "retained CellShard boundary");
    valid &= require_text(document, "no-code-loss migration policy",
        "source preservation policy");
    valid &= require_text(document, "No file under `planning/jbc-preledger-v1/`",
        "historical plan preservation");
    valid &= require_text(document, "and no old run\nidentity is modified",
        "historical run preservation");
    valid &= require_text(document, "`CE-JBC-RUN-V1` is not reused",
        "run identity non-reuse");
    valid &= require_text(document, "59e10ad8963f703c1879957ee0fd5e92b",
        "historical architecture hash");
    valid &= require_text(document, "inventories/jbc_source_migration.csv",
        "migration inventory authority");
    valid &= require_text(document, "Deep CellShard application and runtime",
        "Part Two boundary");

    return valid ? 0 : 1;
}
