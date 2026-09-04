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
          "publish_the_part_one_compiler_charter.md";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "unable to open Part One charter: " << path << '\n';
        return 1;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string document = buffer.str();

    bool valid = true;
    valid &= require_text(document, "### `cellerator`", "compiler product");
    valid &= require_text(document, "### Public CEIR family", "public CEIR scope");
    valid &= require_text(document, "Semantic IR", "Semantic IR");
    valid &= require_text(document, "Planning IR", "Planning IR");
    valid &= require_text(document, "Realization IR", "Realization IR");
    valid &= require_text(document, "### `libCellerator`, standard library, and SDK",
        "SDK product");
    valid &= require_text(document, "### `celleratord`", "language server product");
    valid &= require_text(document, "Useful existing JBC source, tests, and evidence",
        "JBC preservation");
    valid &= require_text(document, "CellShard is a concrete downstream",
        "CellShard boundary");
    valid &= require_text(document, "## Explicit Part Two deferral",
        "Part Two boundary");
    valid &= require_text(document, "does not remove ordinary",
        "low-level freedom");
    valid &= require_text(document, "pointer, template, custom-layout, CUDA",
        "pointer and native control");
    valid &= require_text(document, "milestones M00 through M90",
        "root completion boundary");
    valid &= require_text(document, "06_SOURCE_LANGUAGE_IMPLEMENTATION_PLAN.md",
        "source-language plan review");
    valid &= require_text(document, "07_SEMANTIC_IR_IMPLEMENTATION_PLAN.md",
        "IR plan review");
    valid &= require_text(document, "select CUDA configuration `auto`",
        "approved CUDA decision");

    return valid ? 0 : 1;
}
