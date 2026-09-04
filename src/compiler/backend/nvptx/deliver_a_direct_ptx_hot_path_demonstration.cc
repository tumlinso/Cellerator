#include <Cellerator/compiler/backend/nvptx/deliver_a_direct_ptx_hot_path_demonstration_v1.hh>

#include <sstream>

namespace Cellerator::compiler::backend::nvptx {

unit_degree_relation_ptx_result_v1 lower_unit_degree_relation_apply_directly_to_ptx_v1(
    const unit_degree_relation_ptx_request_v1& request) {
    unit_degree_relation_ptx_result_v1 result;
    if (request.row_count == 0u || request.dense_input_count == 0u) return result;
    if (request.target_sm_major != 7u || request.target_sm_minor != 0u) {
        result.status = unit_degree_relation_ptx_status_v1::unsupported_target;
        return result;
    }
    result.status = unit_degree_relation_ptx_status_v1::success;
    result.kernel_symbol = "ce_unit_degree_relation_apply_sm70";
    result.restrictions = {
        "exactly one logical edge per output row",
        "uint32 column indices are prevalidated below dense_input_count",
        "float32 relation, dense input, accumulation, and output",
        "contiguous device-resident arrays with no aliasing",
        "SM70 target with one independent output writer per row",
    };
    std::ostringstream ptx;
    ptx << ".version 7.0\n"
        << ".target sm_70\n"
        << ".address_size 64\n\n"
        << ".visible .entry " << result.kernel_symbol << "(\n"
        << "    .param .u64 columns,\n"
        << "    .param .u64 weights,\n"
        << "    .param .u64 input,\n"
        << "    .param .u64 output,\n"
        << "    .param .u32 rows\n"
        << ")\n{\n"
        << "    .reg .pred %p<2>;\n"
        << "    .reg .b32 %r<8>;\n"
        << "    .reg .b64 %rd<10>;\n"
        << "    .reg .f32 %f<4>;\n\n"
        << "    ld.param.u64 %rd1, [columns];\n"
        << "    ld.param.u64 %rd2, [weights];\n"
        << "    ld.param.u64 %rd3, [input];\n"
        << "    ld.param.u64 %rd4, [output];\n"
        << "    ld.param.u32 %r1, [rows];\n"
        << "    mov.u32 %r2, %tid.x;\n"
        << "    mov.u32 %r3, %ctaid.x;\n"
        << "    mov.u32 %r4, %ntid.x;\n"
        << "    mad.lo.u32 %r5, %r3, %r4, %r2;\n"
        << "    setp.ge.u32 %p1, %r5, %r1;\n"
        << "    @%p1 bra done;\n"
        << "    mul.wide.u32 %rd5, %r5, 4;\n"
        << "    add.u64 %rd6, %rd1, %rd5;\n"
        << "    ld.global.u32 %r6, [%rd6];\n"
        << "    mul.wide.u32 %rd7, %r6, 4;\n"
        << "    add.u64 %rd8, %rd2, %rd5;\n"
        << "    ld.global.f32 %f1, [%rd8];\n"
        << "    add.u64 %rd9, %rd3, %rd7;\n"
        << "    ld.global.f32 %f2, [%rd9];\n"
        << "    mul.rn.f32 %f3, %f1, %f2;\n"
        << "    add.u64 %rd8, %rd4, %rd5;\n"
        << "    st.global.f32 [%rd8], %f3;\n"
        << "done:\n"
        << "    ret;\n"
        << "}\n";
    result.ptx = ptx.str();
    return result;
}

}  // namespace Cellerator::compiler::backend::nvptx
