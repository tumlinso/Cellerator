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
          "hash_and_index_every_language_and_ir_authority_document.md";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "unable to open language inventory: " << path << '\n';
        return 1;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string document = buffer.str();

    bool valid = true;
    valid &= require_text(document,
        "`docs/language/cellerator-language-specification.md` | 104647",
        "language specification size");
    valid &= require_text(document,
        "`22329eef2e84e6b48d7d304e77c2ed65b75f29721d65f499680faa08c8efe15b`",
        "language specification SHA-256");
    valid &= require_text(document,
        "`docs/language/cellerator-programming-guide.md` | 77857",
        "programming guide size");
    valid &= require_text(document,
        "`5fd78c016fddf3876e20c291affd9042f9dfc854ac520c0834dcb20663e20c26`",
        "programming guide SHA-256");
    valid &= require_text(document, "tracked, index/worktree clean",
        "repository status");
    valid &= require_text(document, "H1 1, H2 30, H3 165, H4-H6 0",
        "specification heading counts");
    valid &= require_text(document, "H1 1, H2 36, H3 129, H4-H6 0",
        "guide heading counts");
    valid &= require_text(document,
        "18. Intermediate representation as a programming feature",
        "language specification IR heading");
    valid &= require_text(document, "16. IR inspection",
        "programming guide IR heading");
    valid &= require_text(document,
        "Externally supplied standalone IR design document: absent",
        "external IR document disposition");
    valid &= require_text(document, "returned exactly the two paths above",
        "complete directory enumeration");

    return valid ? 0 : 1;
}
