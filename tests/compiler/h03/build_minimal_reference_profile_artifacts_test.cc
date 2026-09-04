#include <array>
#include <cassert>
#include <string>
struct row { const char* kind; const char* id; unsigned extent; };
std::string serialize(const std::array<row,3>& rows) {
    std::string out{"kind,id,extent\n"};
    for (const auto& r: rows) out += std::string(r.kind)+","+r.id+","+std::to_string(r.extent)+"\n";
    return out;
}
int main() {
    const std::array<row,3> rows{{{"domain","gene",4},{"domain","cell",4},{"relation","gene_to_cell",4}}};
    assert(serialize(rows) == serialize(rows));
    assert(serialize(rows).find("gene_to_cell") != std::string::npos);
}
