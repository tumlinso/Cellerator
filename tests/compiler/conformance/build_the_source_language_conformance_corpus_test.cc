#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct case_v1 {
    std::string section;
    std::string source;
    bool accepted = true;
};

std::vector<case_v1> load_corpus(const std::string& path) {
    std::ifstream input(path);
    assert(input);
    std::vector<case_v1> cases;
    std::string line;
    std::string section;
    std::string example;
    bool cpp = false;
    bool rejected_designs = false;
    while (std::getline(input, line)) {
        if (line.rfind("## ", 0) == 0) {
            section = line.substr(3);
            rejected_designs = section.rfind("27.", 0) == 0;
        } else if (line.rfind("### ", 0) == 0) {
            section = line.substr(4);
            if (rejected_designs)
                cases.push_back({section, section, false});
        }
        if (line == "```cpp") {
            cpp = true;
            example.clear();
            continue;
        }
        if (cpp && line == "```") {
            cases.push_back({section, example, true});
            cpp = false;
            continue;
        }
        if (cpp)
            example += line + '\n';
        if (rejected_designs && line.rfind("- ", 0) == 0)
            cases.push_back({section, line.substr(2), false});
    }
    return cases;
}

std::string deterministic_diagnostics(const std::vector<case_v1>& corpus) {
    std::vector<std::string> diagnostics;
    for (const auto& test : corpus)
        if (!test.accepted)
            diagnostics.push_back(test.section + ": rejected: " + test.source);
    std::sort(diagnostics.begin(), diagnostics.end());
    std::ostringstream output;
    for (const auto& diagnostic : diagnostics)
        output << diagnostic << '\n';
    return output.str();
}
}  // namespace

int main() {
    const auto corpus = load_corpus("docs/language/cellerator-language-specification.md");
    assert(corpus.size() > 80);
    assert(std::all_of(corpus.begin(), corpus.end(), [](const case_v1& test) {
        return !test.section.empty() && !test.source.empty();
    }));
    const auto positive = std::count_if(corpus.begin(), corpus.end(),
                                        [](const case_v1& test) { return test.accepted; });
    const auto negative = corpus.size() - static_cast<std::size_t>(positive);
    assert(positive > 60 && negative > 5);
    assert(deterministic_diagnostics(corpus) == deterministic_diagnostics(corpus));
}
