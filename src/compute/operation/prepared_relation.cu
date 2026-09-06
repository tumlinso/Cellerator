#include <Cellerator/compute/operation/prepared_relation.hh>
#include <Cellerator/compute/operation/operation_core.hh>

namespace cellerator::compute::relation {
namespace {
namespace core = cellerator::compute::math::core;
// Persistent identities remain value-owned. Compact runtime handles are local
// registry slots, never truncated low words of biological identities.
struct native_contract {
    operation_descriptor semantic{};
    core::numeric_policy numeric{};
    core::structure_set_key structures{};
    core::operation_problem problem{};
    execution::axis_identity source{{1,1},{1,1},{1,1},{1,1}};
    execution::axis_identity destination{{2,1},{2,1},{2,1},{2,1}};
    execution::axis_identity column{{3,1},{3,1},{3,1},{3,1}};
};
status adapt(const operation_descriptor& op, native_contract& out) noexcept {
    auto result = validate(op);
    if (!result) return result;
    if (op.dense_width != 1)
        return {status_code::unsupported_width, "native pair supports N1 only"};
    const auto& a = op.arithmetic;
    if (a.relation_storage != execution::numeric_type::f16
        || a.input_storage != execution::numeric_type::f32
        || a.multiply != execution::numeric_type::f32
        || a.accumulation != execution::numeric_type::f32
        || a.output_storage != execution::numeric_type::f32
        || !a.permit_fma || !a.permit_reassociation
        || a.nonfinite != nonfinite_policy::propagate)
        return {status_code::unsupported_numeric_policy, "FMP1/CTP1 require f16/f32 and permitted FMA/reassociation with propagation"};
    if (op.update != output_update::overwrite || op.input_output_aliasing_legal)
        return {status_code::unsupported_semantics, "native pair requires nonaliasing overwrite"};
    out = {};
    out.semantic = op;
    out.numeric.sparse_storage = a.relation_storage;
    out.numeric.dense_storage = a.input_storage;
    out.numeric.multiply = a.multiply;
    out.numeric.accumulation = a.accumulation;
    out.numeric.output_storage = a.output_storage;
    out.numeric.scalar = execution::numeric_type::f32;
    out.structures.count = 1;
    out.structures.structures[0] = {op.topology.identity, {1,1}, op.topology.epoch};
    out.problem.operation = {1, op.direction == orientation::forward ? 1u : 2u};
    out.problem.input_count = out.problem.output_count = 1;
    out.problem.logical_work_items = op.topology.edge_count;
    return {};
}
} // namespace
} // namespace cellerator::compute::relation

#include <Cellerator/compute/candidate/feature_major_small_n_candidate.hh>
#include <Cellerator/compute/candidate/transpose_backward_candidate.hh>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

namespace cellerator::compute::relation {
namespace cm = cellerator::compute::math;
namespace {
status physical(cm::physical_view_status s) {
    return s ? status{} : status{status_code::invalid_argument, s.message};
}
status cuda_status(cudaError_t s) {
    return s == cudaSuccess ? status{} : status{status_code::cuda_failure, cudaGetErrorString(s)};
}
status check_topology(const topology_descriptor& t, const csr_host_view& csr) {
    // Leave room for terminal indices and candidate grid rounding.
    constexpr auto maximum = std::uint64_t(std::numeric_limits<std::uint32_t>::max()) - 128;
    if (t.source.extent > maximum || t.destination.extent > maximum || t.edge_count > maximum)
        return {status_code::unsupported_semantics, "physical provider exceeds uint32 local-index limits"};
    if (!csr.row_offsets || csr.row_offset_count < t.destination.extent + 1
        || csr.source_index_count < t.edge_count || (t.edge_count && !csr.source_indices))
        return {status_code::insufficient_capacity, "CSR arrays do not cover topology"};
    if (csr.row_offsets[0] != 0 || csr.row_offsets[t.destination.extent] != t.edge_count)
        return {status_code::invalid_shape, "CSR endpoints disagree with edge count"};
    for (std::uint64_t row=0; row<t.destination.extent; ++row) {
        auto begin=csr.row_offsets[row], end=csr.row_offsets[row+1];
        if (end<begin || end>t.edge_count)
            return {status_code::invalid_shape, "CSR offsets are not bounded monotonic offsets"};
        std::vector<std::uint32_t> seen;
        for(auto e=begin;e<end;++e) {
            if(csr.source_indices[e]>=t.source.extent)
                return {status_code::invalid_shape, "CSR source index is outside source axis"};
            seen.push_back(csr.source_indices[e]);
        }
        std::sort(seen.begin(),seen.end());
        if(std::adjacent_find(seen.begin(),seen.end())!=seen.end())
            return {status_code::unsupported_semantics, "masked provider cannot retain duplicate endpoint edges"};
    }
    return {};
}
// Cold interchange into the existing CP-BP masked grammar. Each feature block
// spans at most 32 declared source positions; rows are never permuted. Edge IDs
// travel in a separate map, since physical block order may differ from CSR.
struct cold_tiles {
    std::vector<std::uint32_t> features, blocks, tile_offsets{0}, tile_blocks,
        cell_masks, entry_offsets{0}, gene_masks, value_offsets{0}, logical;
    std::vector<std::uint16_t> zero_values;
    cellpack::persistent_packing_payload_view view{};
    cold_tiles(const topology_descriptor& t,const csr_host_view& csr) {
        auto rows=std::uint32_t(t.destination.extent), cols=std::uint32_t(t.source.extent);
        features.resize(cols);std::iota(features.begin(),features.end(),0);
        for(std::uint32_t b=0;b<cols;b+=32)blocks.push_back(b);
        blocks.push_back(cols);
        for(std::uint32_t first=0;first<rows;first+=32) {
            // block -> lane -> sorted (feature-bit, logical-edge) pairs
            std::map<std::uint32_t,std::map<std::uint32_t,std::map<std::uint32_t,std::uint32_t>>> grouped;
            for(auto row=first;row<std::min(rows,first+32);++row)
                for(auto e=csr.row_offsets[row];e<csr.row_offsets[row+1];++e)
                    grouped[csr.source_indices[e]/32][row-first][csr.source_indices[e]%32]=e;
            for(const auto& block:grouped) {
                tile_blocks.push_back(block.first);std::uint32_t mask=0;
                for(const auto& row:block.second) {
                    mask |= std::uint32_t(1)<<row.first;std::uint32_t genes=0;
                    for(const auto& edge:row.second) {genes |= std::uint32_t(1)<<edge.first;logical.push_back(edge.second);}
                    gene_masks.push_back(genes);value_offsets.push_back(logical.size());
                }
                cell_masks.push_back(mask);entry_offsets.push_back(gene_masks.size());
            }
            tile_offsets.push_back(tile_blocks.size());
        }
        zero_values.resize(t.edge_count);
        view.payload_schema_version=cellpack::persistent_packing_payload_schema_version;
        view.payload_kind=cellpack::persistent_packing_payload_kind;view.payload_identity=1;
        view.image_base=zero_values.data();view.image_bytes=zero_values.size()*2;
        view.plan.semantic_plan_schema_version=cellpack::packing_plan_semantic_schema_version;
        view.plan.geometry_identity_version=cellpack::feature_block_geometry_identity_version;
        view.plan.feature_count=cols;view.plan.feature_block_count=blocks.size()-1;
        view.plan.feature_block_geometry_identity=1;view.plan.feature_block_offsets=blocks.data();
        view.plan.feature_permutation=features.data();
        view.order.ordering_identity=1;
        auto& w=view.tiles;
        w.tile_schema_version=cellpack::warp_tile_schema_version;w.record_schema_version=cellpack::cell_block_record_schema_version;
        w.semantic_plan_schema_version=cellpack::packing_plan_semantic_schema_version;
        w.geometry_identity_version=cellpack::feature_block_geometry_identity_version;
        w.order_schema_version=cellpack::local_cell_order_schema_version;
        w.feature_block_geometry_identity=1;w.ordering_identity=1;w.tile_identity=1;
        w.full_row_count=w.row_count=rows;w.feature_count=cols;w.feature_block_count=blocks.size()-1;
        w.tile_row_width=32;w.tile_count=tile_offsets.size()-1;w.nnz_count=t.edge_count;
        w.tile_block_count=tile_blocks.size();w.row_block_entry_count=gene_masks.size();w.value_size_bytes=2;
        w.feature_axis_fingerprint=1;w.feature_axis_fingerprint_version=1;w.row_domain_identity=1;
        w.tile_block_offsets=tile_offsets.data();w.tile_block_ids=tile_blocks.data();w.tile_block_cell_masks=cell_masks.data();
        w.block_row_entry_offsets=entry_offsets.data();w.row_block_gene_masks=gene_masks.data();
        w.row_block_value_offsets=value_offsets.data();w.values=zero_values.data();
    }
};
} // namespace
struct prepared_relation_pair {
    native_contract forward{},transpose{};
    int device=0;cudaStream_t stream=nullptr;bool poisoned=false;
    void* forward_payload=nullptr;void* transpose_payload=nullptr;
    std::uint32_t* logical_map=nullptr;void* values=nullptr;
    cm::feature_major_projection_view forward_view{};
    cm::transpose_projection_view transpose_view{};
    core::feature_major_small_n_prepared_state forward_state{};
    core::transpose_backward_prepared_state transpose_state{};
    core::prepared_operation forward_operation{},transpose_operation{};
    preparation_report report{};
    ~prepared_relation_pair() {
        int prior=-1;cudaGetDevice(&prior);
        if(prior!=device)cudaSetDevice(device);
        cudaStreamSynchronize(stream);
        if(values)cudaFree(values);if(logical_map)cudaFree(logical_map);
        if(transpose_payload)cudaFree(transpose_payload);if(forward_payload)cudaFree(forward_payload);
        if(prior>=0 && prior!=device)cudaSetDevice(prior);
    }
};
status prepare_relation_pair(const operation_descriptor& forward,const operation_descriptor& transpose,
    const csr_host_view& topology,const preparation_options& options,cudaStream_t stream,
    prepared_relation_pair** out) noexcept {
    if(!out)return {status_code::invalid_argument,"output slot is null"};
    if(*out){*out=nullptr;return {status_code::invalid_argument,"output slot must initially be null"};}
    *out=nullptr;
    try {
        native_contract f{},t{};auto s=adapt(forward,f);if(!s)return s;s=adapt(transpose,t);if(!s)return s;
        auto reverse=transpose;reverse.direction=orientation::forward;
        if(forward.direction!=orientation::forward || transpose.direction!=orientation::transpose || !equivalent(forward,reverse))
            return {status_code::invalid_argument,"forward and transpose must describe one mathematical relation"};
        s=check_topology(forward.topology,topology);if(!s)return s;
        int current=-1;s=cuda_status(cudaGetDevice(&current));if(!s)return s;
        if(current!=options.device_ordinal)return {status_code::incompatible_device,"prepare on the caller current device"};
        unsigned flags=0;s=cuda_status(cudaStreamGetFlags(stream,&flags));if(!s)return s;
        std::unique_ptr<prepared_relation_pair> p(new prepared_relation_pair);
        p->forward=f;p->transpose=t;p->device=current;p->stream=stream;
        auto& report=p->report;report.structure=forward.topology.identity;report.epoch=forward.topology.epoch;
        // Pair-local projection registry identities distinguish the two actual layouts.
        report.forward_projection={forward.topology.identity.low ^ 0x464d5031ULL,forward.topology.identity.high ^ 0x535331ULL};
        if(!execution::valid_identity(report.forward_projection))report.forward_projection.low=1;
        report.transpose_projection=report.forward_projection;report.transpose_projection.high^=0x43545031ULL;
        report.forward_candidate=core::feature_major_small_n_candidate().name;
        report.transpose_candidate=core::transpose_backward_n1_candidate().name;
        if(forward.topology.edge_count) {
            cold_tiles cold(forward.topology,topology);
            cm::feature_major_projection_build_request request{forward.topology.identity,{1,1},forward.topology.epoch,report.forward_projection,{1,1},cold.view};
            cm::feature_major_projection_requirements fr{};
            s=physical(cm::query_feature_major_projection_requirements_host(request,&fr));if(!s)return s;
            std::vector<unsigned char> fh(fr.payload_bytes);cm::feature_major_projection_view fv{};
            s=physical(cm::build_feature_major_projection_host(request,{fh.data(),fh.size()},&fv));if(!s)return s;
            // FMP source-value positions originally name cold CP-BP order. Compose
            // with CSR logical positions before CTP derives its bidirectional maps.
            auto* map=reinterpret_cast<std::uint32_t*>(fh.data()+fv.header.source_value_positions_offset);
            for(std::uint64_t i=0;i<forward.topology.edge_count;++i)map[i]=cold.logical[map[i]];
            cm::transpose_projection_build_request tr{report.transpose_projection,{2,1},fv};
            cm::transpose_projection_requirements rr{};s=physical(cm::query_transpose_projection_requirements_host(tr,&rr));if(!s)return s;
            std::vector<unsigned char> th(rr.payload_bytes);cm::transpose_projection_view tv{};
            s=physical(cm::build_transpose_projection_host(tr,{th.data(),th.size()},&tv));if(!s)return s;
            auto bytes=fr.payload_bytes+rr.payload_bytes+forward.topology.edge_count*6;
            if(options.persistent_byte_limit && bytes>options.persistent_byte_limit)
                return {status_code::insufficient_capacity,"projection and mutable value storage exceed persistent limit"};
            s=cuda_status(cudaMalloc(&p->forward_payload,fr.payload_bytes));if(!s)return s;
            s=cuda_status(cudaMalloc(&p->transpose_payload,rr.payload_bytes));if(!s)return s;
            s=cuda_status(cudaMalloc(reinterpret_cast<void**>(&p->logical_map),forward.topology.edge_count*4));if(!s)return s;
            s=cuda_status(cudaMalloc(&p->values,forward.topology.edge_count*2));if(!s)return s;
            auto upload=[&](void* dst,const void* src,std::size_t bytes) {
                auto submitted=cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,stream);
                auto completed=cudaStreamSynchronize(stream);
                return cuda_status(submitted==cudaSuccess?completed:submitted);
            };
            s=upload(p->forward_payload,fh.data(),fh.size());if(!s)return s;
            s=upload(p->transpose_payload,th.data(),th.size());if(!s)return s;
            s=upload(p->logical_map,map,forward.topology.edge_count*4);if(!s)return s;
            // Cold host arrays cease to be borrowed when preparation returns.
            s=cuda_status(cudaStreamSynchronize(stream));if(!s)return s;
            s=physical(cm::rebind_feature_major_projection(fv,p->forward_payload,fh.size(),&p->forward_view));if(!s)return s;
            s=physical(cm::rebind_transpose_projection(tv,p->transpose_payload,th.size(),&p->transpose_view));if(!s)return s;
            core::projection_key fk{report.forward_projection,{1,1},core::projection_kind::native_feature_major,cm::feature_major_projection_schema_version,cm::feature_major_projection_variant};
            core::projection_key tk{report.transpose_projection,{2,1},core::projection_kind::transpose_or_backward,cm::transpose_projection_schema_version,cm::transpose_projection_variant};
            auto fs=core::prepare_feature_major_small_n_operation(f.problem,f.structures,fk,f.numeric,{},p->forward_view,current,1,f.source,f.destination,f.column,&p->forward_state,&p->forward_operation);
            if(!fs)return {status_code::unsupported_semantics,fs.message};
            auto ts=core::prepare_transpose_backward_n1_operation(t.problem,t.structures,tk,t.numeric,{},p->transpose_view,current,t.source,t.destination,t.column,&p->transpose_state,&p->transpose_operation);
            if(!ts)return {status_code::unsupported_semantics,ts.message};
        }
        report.topology_preparations=1;*out=p.release();return {};
    }catch(const std::bad_alloc&){return {status_code::insufficient_capacity,"cold preparation allocation failed"};}
    catch(...){return {status_code::invalid_argument,"cold projection construction failed"};}
}
status inspect(const prepared_relation_pair& p,preparation_report* out) noexcept {
    if(!out)return {status_code::invalid_argument,"report is null"};*out=p.report;return {};
}
void destroy(prepared_relation_pair* p) noexcept {delete p;}
} // namespace cellerator::compute::relation

namespace cellerator::compute::relation {
namespace {
status submit(prepared_relation_pair& p,orientation direction,const void* input,void* output) noexcept {
    const auto& contract=direction==orientation::forward?p.forward:p.transpose;
    const auto& op=contract.semantic;
    const auto count=result_axis(op).extent;
    if(!op.topology.edge_count)
        return count?cuda_status(cudaMemsetAsync(output,0,count*sizeof(float),p.stream)):status{};
    execution::device_location location{execution::residency_kind::device,{},p.device,0};
    execution::relation_structure relation{{1,1},op.topology.epoch,contract.source,contract.destination,{1,1},op.topology.edge_count};
    execution::value_plane plane{};
    plane.structure={1,1};plane.structure_epoch_value=op.topology.epoch;plane.values=p.values;
    plane.location=location;plane.numeric={execution::numeric_type::f16,execution::numeric_type::f32,execution::numeric_type::f32,0};
    plane.quantization.kind=execution::quantization_kind::none;plane.layout=execution::value_layout_kind::projection_local_order;
    plane.generation={p.report.latest_enqueued_generation.value?p.report.latest_enqueued_generation.value:1};
    plane.element_count=op.topology.edge_count;plane.value_bytes=op.topology.edge_count*2;
    execution::value_binding value{&plane,plane.generation};
    execution::biological_operand_view in{},out{};
    auto dense=[&](execution::biological_operand_view& view,void* pointer,execution::axis_identity major,std::uint64_t rows) {
        view.kind=execution::operand_kind::dense_tensor;auto& d=view.storage.dense;
        d.data=pointer;d.location=location;d.value_type=execution::numeric_type::f32;d.rank=2;
        d.axes[0]=major;d.axes[1]=contract.column;d.shape[0]=rows;d.shape[1]=1;d.stride[0]=d.stride[1]=1;
    };
    dense(in,const_cast<void*>(input),direction==orientation::forward?contract.source:contract.destination,input_axis(op).extent);
    dense(out,output,direction==orientation::forward?contract.destination:contract.source,count);
    execution::launch_bindings launch{};launch.structures=&relation;launch.inputs=&in;launch.outputs=&out;launch.values=&value;
    launch.input_count=launch.output_count=launch.value_count=launch.structure_count=1;
    launch.stream={p.stream,p.device,0};launch.workspace={nullptr,0,location};
    const auto& prepared=direction==orientation::forward?p.forward_operation:p.transpose_operation;
    auto s=core::run_prepared_operation(prepared,launch);
    return s?status{}:status{status_code::cuda_failure,s.message};
}
} // namespace
} // namespace cellerator::compute::relation

namespace cellerator::compute::relation {
namespace {
status check_context(const prepared_relation_pair& p,int device,cudaStream_t stream) noexcept {
    if(p.poisoned)return {status_code::invalid_state,"pair is poisoned by failed CUDA submission"};
    if(stream!=p.stream)return {status_code::incompatible_stream,"pair belongs to one caller stream"};
    int current=-1;auto s=cuda_status(cudaGetDevice(&current));if(!s)return s;
    if(device!=p.device || current!=p.device)return {status_code::incompatible_device,"pair belongs to one current device"};
    return {};
}
status check_pointer(const void* pointer,std::uint64_t bytes,int device,std::size_t alignment) noexcept {
    if(!bytes)return {};
    auto address=reinterpret_cast<std::uintptr_t>(pointer);
    if(!pointer || address%alignment || bytes>std::numeric_limits<std::uintptr_t>::max()-address)
        return {status_code::insufficient_capacity,"missing, misaligned or overflowing device range"};
    cudaPointerAttributes attributes{};
    auto error=cudaPointerGetAttributes(&attributes,pointer);
    if(error!=cudaSuccess){cudaGetLastError();return {status_code::invalid_argument,"pointer is not accessible device storage"};}
    if((attributes.type!=cudaMemoryTypeDevice && attributes.type!=cudaMemoryTypeManaged)
        || attributes.device!=device)
        return {status_code::incompatible_device,"pointer residency differs from pair device"};
    return {};
}
bool overlaps(const void* a,std::uint64_t a_bytes,const void* b,std::uint64_t b_bytes) noexcept {
    if(!a_bytes || !b_bytes)return false;
    auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<y+b_bytes && y<x+a_bytes;
}
__global__ void gather_values(const std::uint16_t* logical,const std::uint32_t* map,
                             std::uint16_t* packed,std::uint32_t count) {
    auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)packed[i]=logical[map[i]];
}
} // namespace
status publish_values(prepared_relation_pair& p,const device_values_binding& binding,cudaStream_t stream) noexcept {
    auto s=check_context(p,binding.device_ordinal,stream);if(!s)return s;
    const auto& topology=p.forward.semantic.topology;
    if(!execution::same_identity(binding.structure,topology.identity) || binding.epoch.value!=topology.epoch.value)
        return {status_code::stale_structure,"value structure or epoch differs from prepared topology"};
    if(!execution::same_identity(binding.logical_edge_order,topology.logical_edge_order))
        return {status_code::incompatible_order,"value logical edge order differs from prepared topology"};
    if(!binding.generation.value || binding.generation.value<=p.report.latest_enqueued_generation.value)
        return {status_code::stale_generation,"publication requires a strictly increasing nonzero generation"};
    if(binding.count<topology.edge_count)return {status_code::insufficient_capacity,"value buffer count is too small"};
    s=check_pointer(binding.f16_data,topology.edge_count*2,p.device,2);if(!s)return s;
    if(overlaps(binding.f16_data,topology.edge_count*2,p.values,topology.edge_count*2))
        return {status_code::invalid_argument,"logical input cannot alias packed value storage"};
    if(topology.edge_count) {
        gather_values<<<(topology.edge_count+255)/256,256,0,stream>>>(
            static_cast<const std::uint16_t*>(binding.f16_data),p.logical_map,
            static_cast<std::uint16_t*>(p.values),topology.edge_count);
        s=cuda_status(cudaPeekAtLastError());
        if(!s){p.poisoned=true;return s;}
    }
    p.report.latest_enqueued_generation=binding.generation;++p.report.value_refreshes;return {};
}
} // namespace cellerator::compute::relation

namespace cellerator::compute::relation {
namespace {
bool matching_axis(const axis_descriptor& a,const axis_descriptor& b) noexcept {
    const auto& x=a.identity;const auto& y=b.identity;
    return a.extent==b.extent && x.header.schema_version==y.header.schema_version
        && x.header.kind==y.header.kind && x.header.byte_count==y.header.byte_count
        && execution::same_identity(x.domain,y.domain) && execution::same_identity(x.order,y.order)
        && execution::same_identity(x.geometry,y.geometry) && execution::same_identity(x.partition,y.partition);
}
status check_launch(const prepared_relation_pair& p,const operation_descriptor& op,
    const device_state_view& input,const device_result_view& output,
    execution::value_generation expected,cudaStream_t stream) noexcept {
    auto s=check_context(p,input.device_ordinal,stream);if(!s)return s;
    if(output.device_ordinal!=p.device)return {status_code::incompatible_device,"output belongs to another device"};
    native_contract checked{};s=adapt(op,checked);if(!s)return s;
    const auto& prepared=op.direction==orientation::forward?p.forward.semantic:p.transpose.semantic;
    if(!execution::same_identity(op.topology.identity,prepared.topology.identity) || op.topology.epoch.value!=prepared.topology.epoch.value)
        return {status_code::stale_structure,"launch topology identity or epoch is stale"};
    if(!equivalent(op,prepared))return {status_code::invalid_argument,"launch semantics differ from prepared contract"};
    if(!expected.value || expected.value!=p.report.latest_enqueued_generation.value)
        return {status_code::stale_generation,"launch must consume latest enqueued value generation"};
    if(!matching_axis(input.axis,input_axis(op)) || !matching_axis(output.axis,result_axis(op)))
        return {status_code::invalid_axis,"launch axis identity, order or extent is incompatible"};
    auto input_count=input_axis(op).extent,output_count=result_axis(op).extent;
    if(input.count<input_count || output.count<output_count)
        return {status_code::insufficient_capacity,"state or result buffer count is too small"};
    s=check_pointer(input.data,input_count*4,p.device,4);if(!s)return s;
    s=check_pointer(output.data,output_count*4,p.device,4);if(!s)return s;
    if(overlaps(input.data,input_count*4,output.data,output_count*4))
        return {status_code::invalid_argument,"input and output byte ranges overlap"};
    return {};
}
} // namespace
status enqueue(prepared_relation_pair& p,const operation_descriptor& op,
    const device_state_view& input,const device_result_view& output,
    execution::value_generation expected,cudaStream_t stream) noexcept {
    auto s=check_launch(p,op,input,output,expected,stream);if(!s)return s;
    s=submit(p,op.direction,input.data,output.data);
    if(!s){p.poisoned=true;return s;}
    if(op.direction==orientation::forward)++p.report.accepted_forward_launches;
    else ++p.report.accepted_transpose_launches;
    return {};
}
} // namespace cellerator::compute::relation
