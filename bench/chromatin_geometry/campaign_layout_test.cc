#include "campaign_runner.hh"
#include <cstdio>
#include <string>
int main() {
    std::string error;
    if (!cgs::run_campaign_layout_checks("", &error)) {
        std::fprintf(stderr, "campaign layout checks: %s\n", error.c_str());
        return 1;
    }
    std::puts("campaign layout checks passed");
    return 0;
}
