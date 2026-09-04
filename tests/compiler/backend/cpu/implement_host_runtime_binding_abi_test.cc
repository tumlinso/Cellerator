#include <Cellerator/compiler/backend/implement_host_runtime_binding_abi_v1.hh>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {

cellerator_host_status_v1 add_constant(
    void*, const cellerator_host_binding_v1* binding) {
    const auto* input = static_cast<const float*>(binding->operands[0].data);
    auto* output = static_cast<float*>(binding->operands[1].data);
    const auto value = *static_cast<const float*>(binding->constants[0].data);
    output[0] = input[0] + value;
    return CELLERATOR_HOST_SUCCESS_V1;
}

}  // namespace

int main() {
    float input = 40;
    float output = 0;
    float constant = 2;
    cellerator_host_operand_v1 operands[]{
        {&input, sizeof(input), sizeof(float), CELLERATOR_HOST_INPUT_V1},
        {&output, sizeof(output), sizeof(float), CELLERATOR_HOST_OUTPUT_V1}};
    const cellerator_host_constant_v1 constants[]{{&constant, sizeof(constant)}};
    const cellerator_host_prepared_stage_v1 stages[]{{add_constant, nullptr}};
    char workspace[16]{};
    cellerator_host_binding_v1 binding{CELLERATOR_HOST_BINDING_ABI_VERSION_V1,
        sizeof(cellerator_host_binding_v1), operands, 2, constants, 1,
        workspace, sizeof(workspace), 8, stages, 1};
    assert(cellerator_host_execute_v1(&binding) == CELLERATOR_HOST_SUCCESS_V1);
    assert(output == 42);
    binding.workspace_bytes = 1;
    assert(cellerator_host_execute_v1(&binding)
        == CELLERATOR_HOST_INSUFFICIENT_WORKSPACE_V1);

    const auto dir = std::filesystem::temp_directory_path() / "ce_ccp1_f02_010";
    std::filesystem::create_directories(dir);
    const auto source = dir / "consumer.c";
    const auto executable = dir / "consumer";
    { std::ofstream out(source); out
        << "#include <Cellerator/compiler/backend/implement_host_runtime_binding_abi_v1.hh>\n"
           "int main(void){ struct cellerator_host_binding_v1 b={0};"
           "b.abi_version=CELLERATOR_HOST_BINDING_ABI_VERSION_V1;"
           "b.struct_size=sizeof(b); return cellerator_host_execute_v1(&b);}\n"; }
    const std::string compile = "cc -I" CELLERATOR_TEST_INCLUDE_DIR " "
        + source.string() + " " CELLERATOR_TEST_BINDING_OBJECT " -lstdc++ -o "
        + executable.string();
    assert(std::system(compile.c_str()) == 0);
    assert(std::system(executable.c_str()) == 0);
    std::filesystem::remove_all(dir);
}
