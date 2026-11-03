#include <Cellerator/compiler/backend/nvcc/nvcc_option_and_architecture_mapping_v1.hh>
#include <cassert>
#include <vector>
int main() {
    using namespace cellerator::compiler::backend::nvcc::v1;
    nvcc_options options{{80, 70}, {90}, "gcc", 20, 3, false, true, true,
                         {"cudart"}, {"--use_fast_math"}};
    const auto arguments = make_nvcc_argv(options);
    assert(arguments && (*arguments)[1] == "-std=c++20");
    assert((*arguments)[5] == "-lineinfo");
    assert((*arguments)[7].find("sm_70") != std::string::npos);

    option_status status{};
    options.cxx_standard = 17;
    assert(!make_nvcc_argv(options, &status) && status == option_status::invalid_standard);
    options.cxx_standard = 23;
    assert(!make_nvcc_argv(options, &status) && status == option_status::invalid_standard);

    options.cxx_standard = 20;
    for (const auto& standard_override :
         std::vector<std::vector<std::string>>{{"-std=c++17"}, {"--std=c++17"},
                                               {"-std", "c++17"}, {"--std", "c++17"}}) {
        options.user_options = standard_override;
        assert(!make_nvcc_argv(options, &status) && status == option_status::unsafe_override);
    }

    options.user_options = {"-arch=sm_80"};
    assert(!make_nvcc_argv(options, &status) && status == option_status::unsafe_override);
}
