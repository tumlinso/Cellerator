#include <Cellerator/compiler/frontend/source/cache_preprocessed_activated_headers_safely_v1.hh>

#include <array>
#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        activated_header_cache_v1 cache;
        const activated_header_cache_key_v1 base{"bytes", "0.1", "A=1", "root/include#2", "clang-18-adapter-v1"};
        cache.store(base, "tokens+shadow");
        if (cache.find(base) != std::optional<std::string_view>("tokens+shadow")) throw std::runtime_error("cache miss");
        for (unsigned component = 0; component != 5; ++component) {
            auto changed = base;
            if (component == 0) changed.file_content += '!';
            if (component == 1) changed.pragma_revision = "0.2";
            if (component == 2) changed.macro_environment = "A=2";
            if (component == 3) changed.include_context = "root/include#3";
            if (component == 4) changed.frontend_adapter_identity = "clang-19-adapter-v1";
            if (cache.find(changed)) throw std::runtime_error("cache key component did not invalidate");
        }
        std::cout << "validated activated-header cache invalidation dimensions\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
