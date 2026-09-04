#include <Cellerator/compiler/tooling/deliver_celleratord_c_parity_milestone_v1.hh>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string resource_directory;
    bool host_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--resource-dir" && i + 1 < argc)
            resource_directory = argv[++i];
        else if (argument == "--host-only")
            host_only = true;
    }
    if (resource_directory.empty()) {
        std::cerr << "celleratord: --resource-dir is required\n";
        return 2;
    }
    std::cout << "celleratord language-server v1 resource-dir="
              << resource_directory << " host-only=" << (host_only ? "true" : "false")
              << '\n';
    return host_only ? 0 : 3;
}
