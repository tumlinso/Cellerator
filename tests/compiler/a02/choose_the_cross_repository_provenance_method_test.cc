#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string read_text(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot read " + path.string());
    }
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

std::string trim(std::string value) {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
        value.erase(value.begin());
    }
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
        value.pop_back();
    }
    if (value.size() >= 2 && value.front() == '`' && value.back() == '`') {
        value = value.substr(1, value.size() - 2);
    }
    return value;
}

std::vector<std::string> cells(const std::string& line) {
    std::vector<std::string> result;
    std::size_t begin = 1;
    while (begin < line.size()) {
        const std::size_t end = line.find('|', begin);
        if (end == std::string::npos) {
            break;
        }
        result.push_back(trim(line.substr(begin, end - begin)));
        begin = end + 1;
    }
    return result;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool hex_id(const std::string& value, std::size_t size) {
    if (value.size() != size) {
        return false;
    }
    for (const unsigned char character : value) {
        if (!std::isxdigit(character)) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: test RECEIPT CELLSHARD_ROOT");
        const std::string receipt = read_text(argv[1]);
        const fs::path cellshard_root = argv[2];

        for (const char* trailer : {
                 "JBC-Migration-ID:", "Source-Repository:", "Source-Branch:",
                 "Source-Commit:", "Source-Path:", "Source-Todo:",
                 "Implementing-Todo:", "Source-Test:", "Provenance-Receipt:",
                 "Migration-Disposition:", "Patch-SHA256:"}) {
            require(receipt.find(trailer) != std::string::npos,
                    std::string("missing trailer rule ") + trailer);
        }
        require(receipt.find("No patch was generated or applied") != std::string::npos,
                "dry run does not explicitly forbid patch application");
        require(receipt.find("git apply --check") != std::string::npos,
                "missing non-mutating patch preflight");
        require(receipt.find("git format-patch --full-index --binary") != std::string::npos,
                "missing binary full-index export rule");

        const std::size_t begin = receipt.find("<!-- PROVENANCE-ROWS-BEGIN -->");
        const std::size_t end = receipt.find("<!-- PROVENANCE-ROWS-END -->");
        require(begin < end, "missing provenance row markers");
        std::set<std::string> migrations;
        std::istringstream lines(receipt.substr(begin, end - begin));
        for (std::string line; std::getline(lines, line);) {
            const auto fields = cells(line);
            if (fields.empty() || fields[0].rfind("JBC-D", 0) != 0) {
                continue;
            }
            require(fields.size() == 7, "malformed provenance row: " + line);
            require(migrations.insert(fields[0]).second,
                    "duplicate provenance row " + fields[0]);
            require(fields[1] == "PX-CS-b9749ad3", "wrong source envelope");
            require(fs::is_regular_file(cellshard_root / fields[2]),
                    "missing source path " + fields[2]);
            require(hex_id(fields[3], 40), "invalid source blob " + fields[3]);
            require(fs::is_regular_file(cellshard_root / fields[4]),
                    "missing test evidence " + fields[4]);
            require(hex_id(fields[5], 40), "invalid test blob " + fields[5]);
            require(fields[6] == "rehome" || fields[6] == "adapt"
                        || fields[6] == "retain-distinct",
                    "invalid migration disposition " + fields[6]);
        }
        require(migrations.size() == 16, "expected 16 traceable migration rows");
        for (int number = 1; number <= 16; ++number) {
            std::ostringstream id;
            id << "JBC-D";
            if (number < 10) {
                id << '0';
            }
            id << number;
            require(migrations.count(id.str()) == 1, "missing " + id.str());
        }

        require(receipt.find("b9749ad3e5146a04f847533d8c6f1a54146aed20")
                    != std::string::npos,
                "missing full source commit");
        require(receipt.find("CE-CCP1-A02-009/DRY-RUN-NOT-APPLIED")
                    != std::string::npos,
                "dry-run implementing Todo is not explicit");

        std::cout << "validated 16 metadata-only provenance rows\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
