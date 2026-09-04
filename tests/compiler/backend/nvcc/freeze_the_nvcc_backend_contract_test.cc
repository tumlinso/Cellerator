#include <Cellerator/compiler/backend/nvcc/freeze_the_nvcc_backend_contract_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::backend::nvcc::v1;
    compilation_job generated{job_kind::device_compile, "field.generated.cu", "field.o",
                              {70u, 80u}, {"cudart", "cellerator_runtime"},
                              {{"field.generated.cu", "model.cell", 1u, 9u}}, true, false};
    assert(validate_job(generated) == contract_status::ok);
    generated.input_is_generated = false;
    assert(validate_job(generated) == contract_status::cellerator_source_reaches_nvcc);

    compilation_job fallthrough{job_kind::device_compile, "kernel.cu", "kernel.o",
                                {70u}, {}, {}, false, true};
    assert(validate_job(fallthrough) == contract_status::ok);
    fallthrough.kind = job_kind::host_compile;
    assert(validate_job(fallthrough) == contract_status::invalid_fallthrough);
}
