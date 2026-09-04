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
          "freeze_the_architecture_reconciliation_receipt.md";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "unable to open architecture receipt: " << path << '\n';
        return 1;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string document = buffer.str();

    bool valid = true;
    valid &= require_text(document, "`CE-CCP1-I01-AUTHORITY-BASELINE` version 1",
        "published interface");
    valid &= require_text(document, "Part One product implementation and\npost-apply plan-semantic Todo mutation have not begun",
        "preimplementation boundary");
    valid &= require_text(document, "A01 validation Todos necessarily advanced",
        "workflow mutation qualification");
    valid &= require_text(document, "0ccaac37-dbbf-448e-a5f8-def197a70aba",
        "project UUID");
    valid &= require_text(document, "fb53d368f34add25316a3aab81251f9dd",
        "semantic fingerprint");
    valid &= require_text(document, "22329eef2e84e6b48d7d304e77c2ed65",
        "language specification hash");
    valid &= require_text(document, "5fd78c016fddf3876e20c291affd9042",
        "programming guide hash");
    valid &= require_text(document, "next_task` is the authoritative",
        "claim operation");
    valid &= require_text(document, "Cellerator owns the compiler",
        "Cellerator compiler ownership");
    valid &= require_text(document, "CellShard retains concrete artifacts",
        "CellShard boundary");
    valid &= require_text(document, "General JIT and deep CellShard",
        "Part Two deferral");
    valid &= require_text(document, "ordinary C++ pointer, reference, template",
        "low-level freedom");
    valid &= require_text(document, "Evaluated-not-promoted is a legitimate",
        "negative result policy");
    valid &= require_text(document, "41b70efc8d4608443871af6e3ec2fae0",
        "authority receipt hash");
    valid &= require_text(document, "93ebe7e4417d0862c49f6a24767c5141",
        "collision receipt hash");
    valid &= require_text(document, "must not silently reinterpret",
        "downstream drift rule");

    return valid ? 0 : 1;
}
