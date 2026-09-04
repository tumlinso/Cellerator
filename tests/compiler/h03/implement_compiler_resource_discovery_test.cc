#include <cassert>
#include <filesystem>
namespace fs = std::filesystem;
fs::path resource(const fs::path& executable, const fs::path& override_path, const char* kind) {
    if (!override_path.empty()) return override_path;
    return (executable.parent_path()/".."/"share"/"cellerator"/kind).lexically_normal();
}
int main() {
    assert(resource("/old/bin/cellerator", {}, "stdlib") == "/old/share/cellerator/stdlib");
    assert(resource("/moved/bin/cellerator", {}, "profiles") == "/moved/share/cellerator/profiles");
    assert(resource("/moved/bin/cellerator", "/explicit/p", "profiles") == "/explicit/p");
}
