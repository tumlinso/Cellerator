#include <Cellerator/compiler/backend/map_backend_diagnostics_to_source_and_ceir_v1.hh>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace cb = cellerator::compiler::backend::v1;

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "ce_ccp1_f02_011";
    std::filesystem::create_directories(dir);
    const auto source = dir / "generated.cc";
    const auto errors = dir / "errors.txt";
    { std::ofstream out(source); out << "int broken = missing_symbol;\n"; }
    const std::string compile = "g++ -c " + source.string() + " -o "
        + (dir / "broken.o").string() + " 2>" + errors.string();
    assert(std::system(compile.c_str()) != 0);
    std::ifstream input(errors);
    const std::string compiler_text((std::istreambuf_iterator<char>(input)), {});
    assert(compiler_text.find("missing_symbol") != std::string::npos);

    const std::vector<cb::generated_source_map_entry_v1> map{{source.string(),
        1, 1, "biology.cell", 17, 9, 101, 202}};
    cb::mapped_backend_diagnostic_v1 mapped{};
    assert(cb::map_backend_diagnostic_v1({cb::mapped_backend_severity_v1::error,
               source.string(), 1, 14, "missing_symbol was not declared"},
        map, true, &mapped));
    assert(mapped.source_file == "biology.cell" && mapped.source_line == 17);
    assert(mapped.semantic_operation == 101 && mapped.realization_operation == 202);
    assert(mapped.generated_code_note.find("generated at") == 0);
    assert(!cb::map_backend_diagnostic_v1({cb::mapped_backend_severity_v1::error,
                source.string(), 2, 1, "outside map"}, map, false, &mapped));
    std::filesystem::remove_all(dir);
}
