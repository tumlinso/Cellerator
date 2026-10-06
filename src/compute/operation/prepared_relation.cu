#include <Cellerator/compute/candidate/sparse/project.hh>
#include <Cellerator/execution/native_value_instance/atom_binding.hh>
#include <Cellerator/compute/operation/prepared_relation.hh>
#include <Cellerator/compute/operation/operation_core.hh>
#include "device_elementwise.cuh"
#include <limits>

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
    // The FP16 projection candidates are intentionally N1/N16 only.  The
    // authoritative FP32 route below is a retained CSR implementation and
    // has no padded-width restriction.
    if (op.arithmetic.relation_storage == execution::numeric_type::f16 &&
        op.dense_width != 1 && op.dense_width != 16)
        return {status_code::unsupported_width, "FP16 prepared projections support N1 and N16 only"};
    const auto& a = op.arithmetic;
    const bool fp16 = a.relation_storage == execution::numeric_type::f16
        && a.input_storage == execution::numeric_type::f32
        && a.multiply == execution::numeric_type::f32
        && a.accumulation == execution::numeric_type::f32
        && a.output_storage == execution::numeric_type::f32;
    const bool fp32 = a.relation_storage == execution::numeric_type::f32
        && a.input_storage == execution::numeric_type::f32
        && a.multiply == execution::numeric_type::f32
        && a.accumulation == execution::numeric_type::f32
        && a.output_storage == execution::numeric_type::f32;
    const bool fp64 = (a.relation_storage == execution::numeric_type::f32
            || a.relation_storage == execution::numeric_type::f64)
        && (a.input_storage == execution::numeric_type::f32
            || a.input_storage == execution::numeric_type::f64)
        && (a.relation_storage == execution::numeric_type::f64
            || a.input_storage == execution::numeric_type::f64)
        && a.multiply == execution::numeric_type::f64
        && a.accumulation == execution::numeric_type::f64
        && a.output_storage == execution::numeric_type::f64;
    if ((!fp16 && !fp32 && !fp64) || !a.permit_fma || !a.permit_reassociation
        || a.nonfinite != nonfinite_policy::propagate)
        return {status_code::unsupported_numeric_policy, "prepared provider supports FP16/FP32 and FP32/FP64 relation-feature tuples with matching FP64 arithmetic"};
    if (op.input_output_aliasing_legal)
        return {status_code::unsupported_semantics, "native pair requires nonaliasing destination storage"};
    if (a.relation_storage == execution::numeric_type::f16
        && op.update != output_update::overwrite)
        return {status_code::unsupported_semantics, "f16 retained provider supports overwrite only"};
    out = {};
    out.semantic = op;
    out.numeric.sparse_storage = a.relation_storage;
    out.numeric.dense_storage = a.input_storage;
    out.numeric.multiply = a.multiply;
    out.numeric.accumulation = a.accumulation;
    out.numeric.output_storage = a.output_storage;
    out.numeric.scalar = a.output_storage;
    out.structures.count = 1;
    out.structures.structures[0] = {op.topology.identity, {1,1}, op.topology.epoch};
    out.problem.operation = {1, op.direction == orientation::forward ? 1u : 2u};
    out.problem.input_count = out.problem.output_count = 1;
    if (op.topology.edge_count > std::numeric_limits<std::uint64_t>::max() / op.dense_width)
        return {status_code::invalid_shape,"interaction count overflows 64 bits"};
    out.problem.logical_work_items = op.topology.edge_count * op.dense_width;
    return {};
}
} // namespace
} // namespace cellerator::compute::relation

#include <Cellerator/compute/candidate/feature_major_small_n_candidate.hh>
#include <Cellerator/compute/candidate/transpose_backward_candidate.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/relation_n16.cuh>
#include <Cellerator/compute/operation/relation_update.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/relation_gradient.cuh>
#include <Cellerator/runtime/relation_value_readiness.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/hybrid_gradient.cuh>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace cellerator::compute::relation {
namespace cm = cellerator::compute::math;
namespace gradient_provider = cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient;
namespace gradient_contract = cellerator::compute::architecture::providers::nvidia::sm70::contract;
namespace {
status physical(cm::physical_view_status s) {
    return s ? status{} : status{status_code::invalid_argument, s.message};
}
status cuda_status(cudaError_t s) {
    return s == cudaSuccess ? status{} : status{status_code::cuda_failure, cudaGetErrorString(s)};
}
status readiness_status(runtime::relation_readiness_status value) noexcept {
    using code=runtime::relation_readiness_status;
    switch(value) {
    case code::success:return {};
    case code::stale_generation:return {status_code::stale_generation,"readiness generation mismatch"};
    case code::identity_mismatch:return {status_code::stale_structure,"readiness identity mismatch"};
    case code::device_mismatch:return {status_code::incompatible_device,"readiness device mismatch"};
    case code::wrong_stream:return {status_code::incompatible_stream,"readiness stream mismatch"};
    case code::capture_unsupported:return {status_code::unsupported_semantics,"mutable readiness cannot be captured"};
    case code::cuda_failure:case code::producer_enqueue_failed:return {status_code::cuda_failure,"readiness submission failed"};
    case code::busy:case code::invalid_state:case code::poisoned:return {status_code::invalid_state,"readiness busy, closed or poisoned"};
    default:return {status_code::invalid_argument,"invalid readiness argument or ticket"};
    }
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
bool checked_bytes(std::uint64_t count,std::uint64_t width,std::uint64_t& result) noexcept {
    if(width && count>std::numeric_limits<std::uint64_t>::max()/width)return false;
    result=count*width;return true;
}
bool checked_add(std::uint64_t left,std::uint64_t right,std::uint64_t& result) noexcept {
    if(right>std::numeric_limits<std::uint64_t>::max()-left)return false;
    result=left+right;return true;
}
std::uint64_t numeric_bytes(execution::numeric_type type) noexcept {
    switch(type) {
    case execution::numeric_type::f16:case execution::numeric_type::bf16:return 2;
    case execution::numeric_type::f32:return 4;
    case execution::numeric_type::f64:return 8;
    default:return 0;
    }
}
std::uint64_t instance_values_bytes(const prepared_relation_pair& p) noexcept;
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
// Counted immutable projection storage. Host callers serialize cold preparation
// across instances sharing this owner; CUDA reads may use independent streams.
struct prepared_relation_structure {
    int device=0;
    std::uint64_t identity=0,base_bytes=0;
    void* forward_payload=nullptr;void* transpose_payload=nullptr;
    std::uint32_t *csr_forward=nullptr,*csr_transpose=nullptr;
    std::uint64_t csr_forward_bytes=0,csr_transpose_bytes=0;
    std::uint32_t* logical_map=nullptr;
    cm::feature_major_projection_view forward_view{};
    cm::transpose_projection_view transpose_view{};
    std::vector<std::uint32_t> logical_to_physical,physical_to_logical;
    std::vector<gradient_contract::edge_ref_v1> physical_edges;
    gradient_contract::edge_ref_v1* device_edges=nullptr;
    ~prepared_relation_structure(){
        int prior=-1;cudaGetDevice(&prior);if(prior!=device)cudaSetDevice(device);
        if(device_edges)cudaFree(device_edges);
        if(csr_forward)cudaFree(csr_forward);if(csr_transpose)cudaFree(csr_transpose);
        if(logical_map)cudaFree(logical_map);
        if(transpose_payload)cudaFree(transpose_payload);
        if(forward_payload)cudaFree(forward_payload);
        if(prior>=0 && prior!=device)cudaSetDevice(prior);
    }
};
namespace {
std::uint64_t next_pair_incarnation() noexcept {
    static std::atomic<std::uint64_t> serial{1};
    auto id=serial.load();
    do{if(id==std::numeric_limits<std::uint64_t>::max())return 0;}
    while(!serial.compare_exchange_weak(id,id+1));
    return id;
}
}
struct prepared_relation_pair {
    native_contract forward{},transpose{};
    int device=0;cudaStream_t stream=nullptr;bool poisoned=false;
    std::shared_ptr<prepared_relation_structure> structure;
    void* values=nullptr;
    float* authoritative_f32=nullptr;
    double* authoritative_f64=nullptr;
    bool derive_f16=false;
    core::feature_major_small_n_prepared_state forward_state{};
    core::transpose_backward_prepared_state transpose_state{};
    core::prepared_operation forward_operation{},transpose_operation{};
    preparation_report report{};
    relation_update_report updates{};
    runtime::relation_value_readiness readiness;
    runtime::relation_read_ticket reader_ticket{};
    value_read_lease active_lease{};
    gradient_provider::hybrid_gradient* hybrid=nullptr;
    bool hybrid_selected=false;
    std::uint64_t persistent_limit = 0, incarnation = 0;

    __half *source_scratch = nullptr, *cotangent_scratch = nullptr;
    execution::order_id physical_order{};
    bool gradient_prepared = false;
    relation_calculus_descriptor calculus{};
    gradient_preparation_options gradient_options{};
    gradient_stamp last_gradient{};
    edge_plane_view last_gradient_output{};
    ~prepared_relation_pair() {
        int prior=-1;cudaGetDevice(&prior);
        if(prior!=device)cudaSetDevice(device);
        (void)readiness.close();
        cudaStreamSynchronize(stream);
        gradient_provider::destroy_hybrid_gradient(hybrid);
        if(source_scratch)cudaFree(source_scratch);if(cotangent_scratch)cudaFree(cotangent_scratch);
        if(values)cudaFree(values);if(authoritative_f32)cudaFree(authoritative_f32);if(authoritative_f64)cudaFree(authoritative_f64);
        structure.reset();
        if(prior>=0 && prior!=device)cudaSetDevice(prior);
    }
};
namespace {
std::uint64_t instance_values_bytes(const prepared_relation_pair& p) noexcept {
    const auto edges=p.forward.semantic.topology.edge_count;
    switch(p.forward.semantic.arithmetic.relation_storage) {
    case execution::numeric_type::f16:return edges*2;
    case execution::numeric_type::f32:return edges*(4+(p.derive_f16?2:0));
    case execution::numeric_type::f64:return edges*8;
    default:return 0;
    }
}
bool float_gradient_contract(const prepared_relation_pair& p) noexcept {
    const auto& a=p.forward.semantic.arithmetic;
    return a.relation_storage==execution::numeric_type::f32
        && a.input_storage==execution::numeric_type::f32
        && a.multiply==execution::numeric_type::f32
        && a.accumulation==execution::numeric_type::f32
        && a.output_storage==execution::numeric_type::f32;
}
}
status prepare_relation_pair(const operation_descriptor& forward,const operation_descriptor& transpose,
    const csr_host_view& topology,const preparation_options& options,cudaStream_t stream,
    prepared_relation_pair** out) noexcept {
    if(!out)return {status_code::invalid_argument,"output slot is null"};
    if(*out)return {status_code::invalid_argument,"output slot must initially be null"};
    *out=nullptr;
    try {
        native_contract f{},t{};auto s=adapt(forward,f);if(!s)return s;s=adapt(transpose,t);if(!s)return s;
        auto reverse=transpose;reverse.direction=orientation::forward;
        if(forward.direction!=orientation::forward || transpose.direction!=orientation::transpose || !equivalent(forward,reverse))
            return {status_code::invalid_argument,"forward and transpose must describe one mathematical relation"};
        if (forward.arithmetic.relation_storage == execution::numeric_type::f32 && options.derive_f16 &&
            forward.dense_width != 1 && forward.dense_width != 16)
            return {status_code::unsupported_width, "derived FP16 evaluation supports N1 and N16 only"};
        if(options.derive_f16 && (forward.arithmetic.relation_storage!=execution::numeric_type::f32
            || forward.arithmetic.input_storage!=execution::numeric_type::f32
            || forward.arithmetic.multiply!=execution::numeric_type::f32
            || forward.arithmetic.accumulation!=execution::numeric_type::f32
            || forward.arithmetic.output_storage!=execution::numeric_type::f32))
            return {status_code::unsupported_numeric_policy,"derived FP16 evaluation requires the existing FP32 relation contract"};
        s=check_topology(forward.topology,topology);if(!s)return s;
        int current=-1;s=cuda_status(cudaGetDevice(&current));if(!s)return s;
        if(current!=options.device_ordinal)return {status_code::incompatible_device,"prepare on the caller current device"};
        cudaStreamCaptureStatus capture{};s=cuda_status(cudaStreamIsCapturing(stream,&capture));if(!s)return s;
        if(capture!=cudaStreamCaptureStatusNone)return {status_code::unsupported_semantics,"cold relation preparation cannot be captured"};
        // Query capture before stream flags: CUDA rejects that flags query during capture.
        unsigned flags=0;s=cuda_status(cudaStreamGetFlags(stream,&flags));if(!s)return s;
        std::unique_ptr<prepared_relation_pair> p(new prepared_relation_pair);
        p->forward=f;p->transpose=t;p->device=current;p->stream=stream;
        p->structure=std::make_shared<prepared_relation_structure>();p->structure->device=current;
        auto id=next_pair_incarnation();
        if(!id)return {status_code::invalid_state,"pair incarnation exhausted"};
        p->incarnation=id;p->structure->identity=id;p->derive_f16=options.derive_f16;p->persistent_limit=options.persistent_byte_limit;
        p->physical_order=forward.topology.logical_edge_order;
        p->physical_order.high^=0x464d503147524144ULL;
        if(!execution::valid_identity(p->physical_order))p->physical_order.low=1;
        auto& report=p->report;report.structure=forward.topology.identity;report.epoch=forward.topology.epoch;
        // Pair-local projection registry identities distinguish the two actual layouts.
        report.forward_projection={forward.topology.identity.low ^ 0x464d5031ULL,forward.topology.identity.high ^ 0x535331ULL};
        if(!execution::valid_identity(report.forward_projection))report.forward_projection.low=1;
        report.transpose_projection=report.forward_projection;report.transpose_projection.high^=0x43545031ULL;
        // XOR can map a valid biological identity to the reserved zero ID.
        // Here forward.low is zero and forward.high is nonzero, so low=1
        // also keeps the two pair-local projection identities distinct.
        if(!execution::valid_identity(report.transpose_projection))report.transpose_projection.low=1;
        report.forward_candidate=core::feature_major_small_n_candidate().name;
        report.transpose_candidate=forward.dense_width==1 ? core::transpose_backward_n1_candidate().name : core::transpose_backward_n16_candidate().name;
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
            p->structure->physical_to_logical.assign(map,map+forward.topology.edge_count);
            p->structure->logical_to_physical.resize(forward.topology.edge_count);
            p->structure->physical_edges.resize(forward.topology.edge_count);
            std::vector<std::uint32_t> logical_rows(forward.topology.edge_count);
            for(std::uint32_t row=0;row<forward.topology.destination.extent;++row)
                for(auto e=topology.row_offsets[row];e<topology.row_offsets[row+1];++e)logical_rows[e]=row;
            for(std::uint32_t slot=0;slot<forward.topology.edge_count;++slot) {
                auto logical=map[slot];p->structure->logical_to_physical[logical]=slot;
                p->structure->physical_edges[slot]={topology.source_indices[logical],logical_rows[logical],slot};
            }
            cm::transpose_projection_build_request tr{report.transpose_projection,{2,1},fv};
            cm::transpose_projection_requirements rr{};s=physical(cm::query_transpose_projection_requirements_host(tr,&rr));if(!s)return s;
            std::vector<unsigned char> th(rr.payload_bytes);cm::transpose_projection_view tv{};
            s=physical(cm::build_transpose_projection_host(tr,{th.data(),th.size()},&tv));if(!s)return s;
            const auto weight_type=forward.arithmetic.relation_storage;
            const bool f32=weight_type==execution::numeric_type::f32;
            const bool f64=weight_type==execution::numeric_type::f64;
            std::uint64_t edge_metadata_bytes=0,value_bytes=0,bytes=0;
            const auto per_edge= f32 ? (4+(p->derive_f16?2:0)) : f64 ? 8 : 2;
            if(!checked_bytes(forward.topology.edge_count,24,edge_metadata_bytes)
                || !checked_bytes(forward.topology.edge_count,per_edge,value_bytes)
                || !checked_add(fr.payload_bytes,rr.payload_bytes,bytes)
                || !checked_add(bytes,edge_metadata_bytes,bytes)
                || !checked_add(bytes,value_bytes,bytes))
                return {status_code::invalid_shape,"persistent allocation size overflows 64 bits"};
            p->updates.persistent_bytes=bytes;
            p->structure->base_bytes=bytes-value_bytes;
            if(options.persistent_byte_limit && bytes>options.persistent_byte_limit)
                return {status_code::insufficient_capacity,"projection and mutable value storage exceed persistent limit"};
            s=cuda_status(cudaMalloc(&p->structure->forward_payload,fr.payload_bytes));if(!s)return s;
            s=cuda_status(cudaMalloc(&p->structure->transpose_payload,rr.payload_bytes));if(!s)return s;
            s=cuda_status(cudaMalloc(reinterpret_cast<void**>(&p->structure->logical_map),forward.topology.edge_count*4));if(!s)return s;
            if(weight_type==execution::numeric_type::f16 || (f32 && p->derive_f16)){s=cuda_status(cudaMalloc(&p->values,forward.topology.edge_count*2));if(!s)return s;}
            if(f32){s=cuda_status(cudaMalloc(&p->authoritative_f32,forward.topology.edge_count*4));if(!s)return s;}
            if(f64){s=cuda_status(cudaMalloc(reinterpret_cast<void**>(&p->authoritative_f64),forward.topology.edge_count*8));if(!s)return s;}
            auto upload=[&](void* dst,const void* src,std::size_t bytes) {
                auto submitted=cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,stream);
                auto completed=cudaStreamSynchronize(stream);
                return cuda_status(submitted==cudaSuccess?completed:submitted);
            };
            s=upload(p->structure->forward_payload,fh.data(),fh.size());if(!s)return s;
            s=upload(p->structure->transpose_payload,th.data(),th.size());if(!s)return s;
            s=upload(p->structure->logical_map,map,forward.topology.edge_count*4);if(!s)return s;
            if(f32 || f64){
                auto build_csr=[&](bool transposed,std::uint32_t** device,std::uint64_t& footprint)->status {
                    const auto rows=transposed?forward.topology.source.extent:forward.topology.destination.extent;
                    const auto edges=forward.topology.edge_count;
                    std::vector<std::uint32_t> payload(rows+1+edges*2,0);
                    for(const auto& edge:p->structure->physical_edges)++payload[(transposed?edge.source_local:edge.destination_local)+1];
                    for(std::uint64_t row=1;row<=rows;++row)payload[row]+=payload[row-1];
                    std::vector<std::uint32_t> cursor(payload.begin(),payload.begin()+rows);
                    for(std::uint32_t physical=0;physical<edges;++physical){const auto& edge=p->structure->physical_edges[physical];
                        auto pos=cursor[transposed?edge.source_local:edge.destination_local]++;
                        payload[rows+1+pos]=transposed?edge.destination_local:edge.source_local;
                        payload[rows+1+edges+pos]=physical;}
                    footprint=payload.size()*4;
                    if(p->persistent_limit && (p->updates.persistent_bytes>p->persistent_limit
                        || footprint>p->persistent_limit-p->updates.persistent_bytes))
                        return {status_code::insufficient_capacity,"CSR projection exceeds persistent budget"};
                    auto result=cuda_status(cudaMalloc(device,footprint));if(!result)return result;
                    result=upload(*device,payload.data(),footprint);if(!result)return result;
                    p->structure->base_bytes+=footprint;p->updates.persistent_bytes+=footprint;return {};
                };
                s=build_csr(false,&p->structure->csr_forward,p->structure->csr_forward_bytes);if(!s)return s;
                s=build_csr(true,&p->structure->csr_transpose,p->structure->csr_transpose_bytes);if(!s)return s;
                report.forward_candidate=report.transpose_candidate=f64?"retained-csr-f64":"retained-csr-f32";
            }
            // Cold host arrays cease to be borrowed when preparation returns.
            s=cuda_status(cudaStreamSynchronize(stream));if(!s)return s;
            s=physical(cm::rebind_feature_major_projection(fv,p->structure->forward_payload,fh.size(),&p->structure->forward_view));if(!s)return s;
            s=physical(cm::rebind_transpose_projection(tv,p->structure->transpose_payload,th.size(),&p->structure->transpose_view));if(!s)return s;
            if (weight_type==execution::numeric_type::f16 || (f32 && p->derive_f16)) {
                core::projection_key fk{report.forward_projection,{1,1},core::projection_kind::native_feature_major,cm::feature_major_projection_schema_version,cm::feature_major_projection_variant};
                core::projection_key tk{report.transpose_projection,{2,1},core::projection_kind::transpose_or_backward,cm::transpose_projection_schema_version,cm::transpose_projection_variant};
                auto projected_numeric=f.numeric;projected_numeric.sparse_storage=execution::numeric_type::f16;
                auto fs=core::prepare_feature_major_small_n_operation(f.problem,f.structures,fk,projected_numeric,{},p->structure->forward_view,current,forward.dense_width,f.source,f.destination,f.column,&p->forward_state,&p->forward_operation);
                if(!fs)return {status_code::unsupported_semantics,fs.message};
                auto prepare_transpose = forward.dense_width == 1 ? core::prepare_transpose_backward_n1_operation : core::prepare_transpose_backward_n16_operation;
                auto ts=prepare_transpose(t.problem,t.structures,tk,projected_numeric,{},p->structure->transpose_view,current,t.source,t.destination,t.column,&p->transpose_state,&p->transpose_operation);
                if(!ts)return {status_code::unsupported_semantics,ts.message};
            }
        }
        if(!forward.topology.edge_count) {
            report.forward_projection={};report.transpose_projection={};
            report.forward_candidate=forward.topology.destination.extent?"device-zero-fill":"device-no-op";
            report.transpose_candidate=forward.topology.source.extent?"device-zero-fill":"device-no-op";
        }
        s=readiness_status(p->readiness.initialize(forward.topology.identity,forward.topology.epoch,current,stream));
        if(!s)return s;
        report.topology_preparations=1;*out=p.release();return {};
    }catch(const std::bad_alloc&){return {status_code::insufficient_capacity,"cold preparation allocation failed"};}
    catch(...){return {status_code::invalid_argument,"cold projection construction failed"};}
}
status create_relation_instance(const prepared_relation_pair& source,cudaStream_t stream,
    prepared_relation_pair** out) noexcept {
    if(!out || *out)return {status_code::invalid_argument,"new instance slot must be present and null"};
    if(source.poisoned)return {status_code::invalid_state,"cannot share a poisoned instance"};
    int current=-1;auto s=cuda_status(cudaGetDevice(&current));if(!s)return s;
    if(current!=source.device)return {status_code::incompatible_device,"instance must use structure device"};
    cudaStreamCaptureStatus capture{};s=cuda_status(cudaStreamIsCapturing(stream,&capture));if(!s)return s;
    if(capture!=cudaStreamCaptureStatusNone)return {status_code::unsupported_semantics,"instance preparation cannot be captured"};
    try {
        std::unique_ptr<prepared_relation_pair> p(new prepared_relation_pair);
        p->device=source.device;p->stream=stream;p->structure=source.structure;
        p->forward=source.forward;p->transpose=source.transpose;p->physical_order=source.physical_order;
        p->persistent_limit=source.persistent_limit;p->derive_f16=source.derive_f16;p->incarnation=next_pair_incarnation();
        if(!p->incarnation)return {status_code::invalid_state,"pair incarnation exhausted"};
        p->report=source.report;p->report.latest_enqueued_generation={};
        p->report.derived_f16_generation={};p->report.value_refreshes=0;p->report.accepted_forward_launches=0;p->report.accepted_transpose_launches=0;
        auto edges=p->forward.semantic.topology.edge_count;
        if(!checked_add(p->structure->base_bytes,instance_values_bytes(*p),p->updates.persistent_bytes))
            return {status_code::invalid_shape,"shared plus instance allocation overflows 64 bits"};
        if(edges){
            if(source.authoritative_f32){s=cuda_status(cudaMalloc(&p->authoritative_f32,edges*4));if(!s)return s;}
            if(source.authoritative_f64){s=cuda_status(cudaMalloc(reinterpret_cast<void**>(&p->authoritative_f64),edges*8));if(!s)return s;}
            const auto type=p->forward.semantic.arithmetic.relation_storage;
            if(type==execution::numeric_type::f16 || (type==execution::numeric_type::f32 && p->derive_f16)){s=cuda_status(cudaMalloc(&p->values,edges*2));if(!s)return s;}
            if (type==execution::numeric_type::f16 || (type==execution::numeric_type::f32 && p->derive_f16)) {
                // Candidate launch state points into the new instance, while views
                // refer to the one counted structure. No projection is rebuilt.
                const auto& f=p->forward;const auto& t=p->transpose;
                core::projection_key fk{p->report.forward_projection,{1,1},core::projection_kind::native_feature_major,cm::feature_major_projection_schema_version,cm::feature_major_projection_variant};
                core::projection_key tk{p->report.transpose_projection,{2,1},core::projection_kind::transpose_or_backward,cm::transpose_projection_schema_version,cm::transpose_projection_variant};
                auto projected_numeric=f.numeric;projected_numeric.sparse_storage=execution::numeric_type::f16;
                auto fs=core::prepare_feature_major_small_n_operation(f.problem,f.structures,fk,projected_numeric,{},p->structure->forward_view,current,f.semantic.dense_width,f.source,f.destination,f.column,&p->forward_state,&p->forward_operation);
                if(!fs)return {status_code::unsupported_semantics,fs.message};
                auto prepare_transpose=f.semantic.dense_width==1?core::prepare_transpose_backward_n1_operation:core::prepare_transpose_backward_n16_operation;
                auto ts=prepare_transpose(t.problem,t.structures,tk,projected_numeric,{},p->structure->transpose_view,current,t.source,t.destination,t.column,&p->transpose_state,&p->transpose_operation);
                if(!ts)return {status_code::unsupported_semantics,ts.message};
            }
        }
        s=readiness_status(p->readiness.initialize(p->forward.semantic.topology.identity,
            p->forward.semantic.topology.epoch,current,stream));if(!s)return s;
        *out=p.release();return {};
    }catch(const std::bad_alloc&){return {status_code::insufficient_capacity,"instance allocation failed"};}
}
status inspect(const prepared_relation_pair& p,preparation_report* out) noexcept {
    if(!out)return {status_code::invalid_argument,"report is null"};*out=p.report;
    out->structural_preparation_id=p.structure->identity;
    out->structural_instance_count=p.structure.use_count();
    out->shared_structural_bytes=p.structure->base_bytes+
        (p.structure->device_edges?p.structure->physical_edges.size()*sizeof(gradient_contract::edge_ref_v1):0);
    out->instance_value_bytes=instance_values_bytes(p);
    return {};
}
void destroy(prepared_relation_pair* p) noexcept {
    // Compatibility path cannot report busy. Preserve the pair and its borrow;
    // mutable callers use close_relation_pair to observe teardown status.
    (void)close_relation_pair(&p);
}
} // namespace cellerator::compute::relation

namespace cellerator::compute::relation {
namespace {
status submit_half(prepared_relation_pair& p,orientation direction,const void* input,void* output) noexcept {
    const auto& contract=direction==orientation::forward?p.forward:p.transpose;
    const auto& op=contract.semantic;
    const auto count=result_axis(op).extent;
    if(!op.topology.edge_count)
        return count?cuda_status(cudaMemsetAsync(output,0,count*op.dense_width*sizeof(float),p.stream)):status{};
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
        d.axes[0]=major;d.axes[1]=contract.column;d.shape[0]=rows;d.shape[1]=op.dense_width;d.stride[0]=op.dense_width;d.stride[1]=1;
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
float effect_input_scale(const operation_descriptor& op) noexcept {
    return op.update == output_update::affine_accumulate ? static_cast<float>(op.input_scale) : 1.0f;
}
float effect_destination_scale(const operation_descriptor& op) noexcept {
    return op.update == output_update::affine_accumulate ? static_cast<float>(op.destination_scale)
        : op.update == output_update::accumulate ? 1.0f : 0.0f;
}
double effect_input_scale_f64(const operation_descriptor& op) noexcept {
    return op.update == output_update::affine_accumulate ? op.input_scale : 1.0;
}
double effect_destination_scale_f64(const operation_descriptor& op) noexcept {
    return op.update == output_update::affine_accumulate ? op.destination_scale
        : op.update == output_update::accumulate ? 1.0 : 0.0;
}
__global__ void apply_empty_destination_effect(float* output,std::uint64_t count,float destination_scale) {
    auto index=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count) output[index]=destination_scale==0.0f?0.0f:destination_scale*output[index];
}
__global__ void apply_empty_destination_effect_f64(double* output,std::uint64_t count,double destination_scale) {
    auto index=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count) output[index]=destination_scale==0.0?0.0:destination_scale*output[index];
}
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
bool protected_overlap(const prepared_relation_pair& p,const void* data,std::uint64_t bytes) noexcept {
    auto edges=p.forward.semantic.topology.edge_count;
    auto weight_type=p.forward.semantic.arithmetic.relation_storage;
    return (p.hybrid&&gradient_provider::overlaps_hybrid_storage(*p.hybrid,data,bytes))
        || overlaps(data,bytes,p.values,(weight_type==execution::numeric_type::f16 || p.derive_f16)?edges*2:0)
        || overlaps(data,bytes,p.authoritative_f32,p.authoritative_f32?edges*4:0)
        || overlaps(data,bytes,p.authoritative_f64,p.authoritative_f64?edges*8:0)
        || overlaps(data,bytes,p.structure->csr_forward,p.structure->csr_forward_bytes) || overlaps(data,bytes,p.structure->csr_transpose,p.structure->csr_transpose_bytes) || overlaps(data,bytes,p.structure->logical_map,edges*4)
        || overlaps(data,bytes,p.structure->forward_payload,p.structure->forward_view.header.payload_bytes)
        || overlaps(data,bytes,p.structure->transpose_payload,p.structure->transpose_view.header.payload_bytes)
        || overlaps(data,bytes,p.structure->device_edges,edges*sizeof(gradient_contract::edge_ref_v1))
        || (p.source_scratch && overlaps(data,bytes,p.source_scratch,p.forward.semantic.topology.source.extent*32))
        || (p.cotangent_scratch && overlaps(data,bytes,p.cotangent_scratch,p.forward.semantic.topology.destination.extent*32));
}
__global__ void gather_values(const std::uint16_t* logical,const std::uint32_t* map,
                             std::uint16_t* packed,std::uint32_t count) {
    auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)packed[i]=logical[map[i]];
}
} // namespace
status publish_values(prepared_relation_pair& p,const device_values_binding& binding,cudaStream_t stream) noexcept {
    if(p.forward.semantic.arithmetic.relation_storage!=execution::numeric_type::f16)return {status_code::unsupported_numeric_policy,"use explicit f32 publication"};
    auto s=check_context(p,binding.device_ordinal,stream);if(!s)return s;
    // Capturing a gather does not enqueue a generation. There is deliberately
    // no publication-through-graph protocol in this bounded API.
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    s=cuda_status(cudaStreamIsCapturing(stream,&capture));if(!s)return s;
    if(capture!=cudaStreamCaptureStatusNone)
        return {status_code::unsupported_semantics,"value publication during stream capture is unsupported"};
    const auto& topology=p.forward.semantic.topology;
    if(!execution::same_identity(binding.structure,topology.identity) || binding.epoch.value!=topology.epoch.value)
        return {status_code::stale_structure,"value structure or epoch differs from prepared topology"};
    if(!execution::same_identity(binding.logical_edge_order,topology.logical_edge_order))
        return {status_code::incompatible_order,"value logical edge order differs from prepared topology"};
    if(!binding.generation.value || binding.generation.value<=p.report.latest_enqueued_generation.value)
        return {status_code::stale_generation,"publication requires a strictly increasing nonzero generation"};
    if(binding.count<topology.edge_count)return {status_code::insufficient_capacity,"value buffer count is too small"};
    s=check_pointer(binding.f16_data,topology.edge_count*2,p.device,2);if(!s)return s;
    if(protected_overlap(p,binding.f16_data,topology.edge_count*2))
        return {status_code::invalid_argument,"logical input cannot alias packed value storage"};
    s=readiness_status(p.readiness.validate_write(p.report.latest_enqueued_generation,binding.generation,stream));if(!s)return s;
    cudaError_t submitted=cudaSuccess;
    if(topology.edge_count) {
        gather_values<<<(topology.edge_count+255)/256,256,0,stream>>>(
            static_cast<const std::uint16_t*>(binding.f16_data),p.structure->logical_map,
            static_cast<std::uint16_t*>(p.values),topology.edge_count);
        submitted=cudaPeekAtLastError();
    }
    s=readiness_status(p.readiness.publish(binding.generation,stream,submitted));
    if(submitted!=cudaSuccess||!s){p.poisoned=true;return submitted!=cudaSuccess?cuda_status(submitted):s;}
    p.last_gradient={};p.last_gradient_output={};
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
    if(input.dtype!=op.arithmetic.input_storage || output.dtype!=op.arithmetic.output_storage)
        return {status_code::unsupported_numeric_policy,"state/result dtype differs from operation contract"};
    if(!expected.value || expected.value!=p.report.latest_enqueued_generation.value)
        return {status_code::stale_generation,"launch must consume latest enqueued value generation"};
    if(!matching_axis(input.axis,input_axis(op)) || !matching_axis(output.axis,result_axis(op)))
        return {status_code::invalid_axis,"launch axis identity, order or extent is incompatible"};
    std::uint64_t input_count=0,output_count=0,input_bytes=0,output_bytes=0;
    if(!checked_bytes(input_axis(op).extent,op.dense_width,input_count)
        || !checked_bytes(result_axis(op).extent,op.dense_width,output_count)
        || !checked_bytes(input_count,numeric_bytes(input.dtype),input_bytes)
        || !checked_bytes(output_count,numeric_bytes(output.dtype),output_bytes))
        return {status_code::invalid_shape,"state or result byte range overflows 64 bits"};
    if(input.count<input_count || output.count<output_count)
        return {status_code::insufficient_capacity,"state or result buffer count is too small"};
    s=check_pointer(input.data,input_bytes,p.device,numeric_bytes(input.dtype));if(!s)return s;
    s=check_pointer(output.data,output_bytes,p.device,numeric_bytes(output.dtype));if(!s)return s;
    if(protected_overlap(p,input.data,input_bytes) || protected_overlap(p,output.data,output_bytes)
        || overlaps(input.data,input_bytes,output.data,output_bytes))
        return {status_code::invalid_argument,"input and output byte ranges overlap"};
    return {};
}
} // namespace
status enqueue(prepared_relation_pair& p,const operation_descriptor& op,
    const device_state_view& input,const device_result_view& output,
    execution::value_generation expected,cudaStream_t stream) noexcept {
    auto s=check_launch(p,op,input,output,expected,stream);if(!s)return s;
    if(op.arithmetic.relation_storage==execution::numeric_type::f32
        && op.arithmetic.multiply==execution::numeric_type::f32){
        const auto& t=op.topology;const auto rows=result_axis(op).extent;
        if(!t.edge_count) {
            auto count=rows*op.dense_width;
            if(!count)s={};
            else {
                apply_empty_destination_effect<<<(count+255)/256,256,0,stream>>>(
                    static_cast<float*>(output.data),count,effect_destination_scale(op));
                s=cuda_status(cudaPeekAtLastError());
            }
        }
        else try {
            auto* csr=op.direction==orientation::forward?p.structure->csr_forward:p.structure->csr_transpose;
            runtime::execution_context context{};context.device=p.device;context.stream=stream;
            compute::sparse::project::csr_spmm_fwd_f32(context,csr,csr+rows+1,p.authoritative_f32,rows,input_axis(op).extent,
                static_cast<const float*>(input.data),op.dense_width,op.dense_width,static_cast<float*>(output.data),op.dense_width,csr+rows+1+t.edge_count,
                effect_input_scale(op),effect_destination_scale(op));
        }catch(...){s={status_code::invalid_state,"f32 candidate launch failed"};}
    }else if(op.arithmetic.multiply==execution::numeric_type::f64){
        const auto& t=op.topology;const auto rows=result_axis(op).extent;
        if(!t.edge_count){
            auto count=rows*op.dense_width;
            if(!count)s={};
            else {
                apply_empty_destination_effect_f64<<<(count+255)/256,256,0,stream>>>(
                    static_cast<double*>(output.data),count,effect_destination_scale_f64(op));
                s=cuda_status(cudaPeekAtLastError());
            }
        }else try {
            auto* csr=op.direction==orientation::forward?p.structure->csr_forward:p.structure->csr_transpose;
            runtime::execution_context context{};context.device=p.device;context.stream=stream;
            const auto* indices=csr+rows+1+t.edge_count;
            auto* output_data=static_cast<double*>(output.data);
            if(op.arithmetic.relation_storage==execution::numeric_type::f32
                && op.arithmetic.input_storage==execution::numeric_type::f64)
                compute::sparse::project::csr_spmm_fwd<float,double,double,double,double,double>(
                    context,csr,csr+rows+1,p.authoritative_f32,rows,input_axis(op).extent,
                    static_cast<const double*>(input.data),op.dense_width,op.dense_width,output_data,op.dense_width,
                    indices,effect_input_scale_f64(op),effect_destination_scale_f64(op));
            else if(op.arithmetic.relation_storage==execution::numeric_type::f64
                && op.arithmetic.input_storage==execution::numeric_type::f32)
                compute::sparse::project::csr_spmm_fwd<double,float,double,double,double,double>(
                    context,csr,csr+rows+1,p.authoritative_f64,rows,input_axis(op).extent,
                    static_cast<const float*>(input.data),op.dense_width,op.dense_width,output_data,op.dense_width,
                    indices,effect_input_scale_f64(op),effect_destination_scale_f64(op));
            else
                compute::sparse::project::csr_spmm_fwd<double,double,double,double,double,double>(
                    context,csr,csr+rows+1,p.authoritative_f64,rows,input_axis(op).extent,
                    static_cast<const double*>(input.data),op.dense_width,op.dense_width,output_data,op.dense_width,
                    indices,effect_input_scale_f64(op),effect_destination_scale_f64(op));
        }catch(...){s={status_code::invalid_state,"f64 candidate launch failed"};}
    }else s=submit_half(p,op.direction,input.data,output.data);
    if(!s){p.poisoned=true;return s;}
    if(op.direction==orientation::forward)++p.report.accepted_forward_launches;
    else ++p.report.accepted_transpose_launches;
    return {};
}
} // namespace cellerator::compute::relation

namespace cellerator::compute::relation {
namespace {
status reject_capture(cudaStream_t stream) noexcept {
    cudaStreamCaptureStatus capture{};auto s=cuda_status(cudaStreamIsCapturing(stream,&capture));
    if(!s)return s;
    return capture==cudaStreamCaptureStatusNone?status{}:
        status{status_code::unsupported_semantics,"mutable relation operation cannot be captured"};
}
status check_edge_plane(const prepared_relation_pair& p,const edge_plane_view& plane) noexcept {
    if(!float_gradient_contract(p))
        return {status_code::unsupported_numeric_policy,"edge updates and gradient planes require the FP32 relation contract"};
    const auto& t=p.forward.semantic.topology;
    if(!execution::same_identity(plane.structure,t.identity)||plane.epoch.value!=t.epoch.value)
        return {status_code::stale_structure,"edge plane structure or epoch mismatch"};
    if(!execution::same_identity(plane.order,p.physical_order))
        return {status_code::incompatible_order,"edge plane must use prepared physical order"};
    if(plane.device_ordinal!=p.device)return {status_code::incompatible_device,"edge plane device mismatch"};
    if(plane.count<t.edge_count)return {status_code::insufficient_capacity,"edge plane capacity too small"};
    auto s=check_pointer(plane.f32_data,t.edge_count*4,p.device,4);if(!s)return s;
    if(protected_overlap(p,plane.f32_data,t.edge_count*4))
        return {status_code::invalid_argument,"edge plane overlaps pair-owned storage"};
    return {};
}
}
status inspect_edge_layout(const prepared_relation_pair& p,edge_layout_view* out) noexcept {
    if(!out)return {status_code::invalid_argument,"edge layout output absent"};
    *out={p.physical_order,p.forward.semantic.topology.edge_count,p.structure->logical_to_physical.data()};return {};
}
status inspect_updates(const prepared_relation_pair& p,relation_update_report* out) noexcept {
    if(!out)return {status_code::invalid_argument,"update report output absent"};
    *out=p.updates;out->relation=p.report;
    out->ready_records=p.readiness.ready_records();out->reader_returns=p.readiness.reader_returns();return {};
}
status prepare_relation_gradient(prepared_relation_pair& p,const relation_calculus_descriptor& calculus,
    const gradient_preparation_options& options,cudaStream_t stream) noexcept {
    if(!float_gradient_contract(p))return {status_code::unsupported_numeric_policy,"gradient provider supports FP32 relation/input/output only"};
    auto s=reject_capture(stream);if(!s)return s;
    s=check_context(p,p.device,stream);if(!s)return s;
    s=validate(calculus);if(!s)return s;
    if(!equivalent(calculus.forward,p.forward.semantic)||!equivalent(calculus.transpose,p.transpose.semantic))
        return {status_code::invalid_argument,"gradient calculus differs from prepared relation"};
    if(calculus.forward.dense_width!=16)return {status_code::unsupported_width,"gradient provider requires N16"};
    if(options.route!=gradient_route::automatic && options.route!=gradient_route::force_sparse && options.route!=gradient_route::force_hybrid)
        return {status_code::invalid_argument,"unknown gradient route"};
    if(options.route==gradient_route::force_hybrid && calculus.gradient!=gradient_arithmetic::round_operands_f16_rne)
        return {status_code::unsupported_semantics,"full-f32 semantics cannot use WMMA"};
    if(p.gradient_prepared)return {status_code::invalid_state,"gradient already prepared"};
    auto nx=calculus.forward.topology.source.extent*16,ny=calculus.forward.topology.destination.extent*16;
    auto bytes=p.structure->physical_edges.size()*sizeof(gradient_contract::edge_ref_v1);
    auto scratch=calculus.gradient==gradient_arithmetic::round_operands_f16_rne?(nx+ny)*2:0;
    if((options.scratch_byte_limit && scratch>options.scratch_byte_limit) ||
        (p.persistent_limit && bytes+scratch>p.persistent_limit-p.updates.persistent_bytes))
        return {status_code::insufficient_capacity,"gradient prepared storage exceeds budget"};
    gradient_contract::edge_ref_v1* edges=p.structure->device_edges;
    const bool owns_new_edges=edges==nullptr;__half *x=nullptr,*y=nullptr;
    auto cleanup=[&](){if(owns_new_edges && edges)cudaFree(edges);if(x)cudaFree(x);if(y)cudaFree(y);};
    if(bytes && owns_new_edges) {s=cuda_status(cudaMalloc(&edges,bytes));if(!s){cleanup();return s;}}
    if(scratch && nx) {s=cuda_status(cudaMalloc(&x,nx*2));if(!s){cleanup();return s;}}
    if(scratch && ny) {s=cuda_status(cudaMalloc(&y,ny*2));if(!s){cleanup();return s;}}
    if(bytes && owns_new_edges) {
        auto submitted=cudaMemcpyAsync(edges,p.structure->physical_edges.data(),bytes,cudaMemcpyHostToDevice,stream);
        auto completed=cudaStreamSynchronize(stream);
        s=cuda_status(submitted==cudaSuccess?completed:submitted);if(!s){cleanup();return s;}
    }
    gradient_provider::hybrid_gradient* hybrid=nullptr;
    gradient_provider::hybrid_report hybrid_report{};
    if(options.route==gradient_route::force_hybrid &&
        calculus.gradient==gradient_arithmetic::round_operands_f16_rne && !p.structure->physical_edges.empty()) {
        auto limit=p.persistent_limit?p.persistent_limit-p.updates.persistent_bytes-bytes-scratch:std::uint64_t(256)*1024*1024;
        auto result=limit?gradient_provider::prepare_hybrid_gradient(p.structure->physical_edges.data(),
            {edges,0,static_cast<std::uint32_t>(p.structure->physical_edges.size()),
             static_cast<std::uint32_t>(calculus.forward.topology.source.extent),
             static_cast<std::uint32_t>(calculus.forward.topology.destination.extent)},nullptr,0,limit,stream,&hybrid,options.scratch_byte_limit?options.scratch_byte_limit-scratch:~std::uint64_t{0})
             :gradient_contract::status_v1::unsupported;
        if(result!=gradient_contract::status_v1::success) {
            cleanup();
            if(result==gradient_contract::status_v1::unsupported)return {status_code::insufficient_capacity,"hybrid cover or scratch exceeds preparation budget"};
            if(result==gradient_contract::status_v1::invalid_argument)return {status_code::invalid_argument,"hybrid cover metadata is invalid"};
            return {status_code::cuda_failure,"hybrid preparation failed"};
        }
        if(hybrid)hybrid_report=gradient_provider::inspect_hybrid_gradient(*hybrid);
    }
    gradient_provider::gradient_selection selection{};
    auto choice=options.route==gradient_route::force_hybrid?gradient_provider::gradient_choice::force_hybrid:
        options.route==gradient_route::force_sparse?gradient_provider::gradient_choice::force_sparse:gradient_provider::gradient_choice::automatic;
    auto selected=gradient_provider::select_gradient_route(calculus.gradient==gradient_arithmetic::round_operands_f16_rne,
        choice,hybrid_report.tile_count,selection);
    if(selected!=gradient_contract::status_v1::success){gradient_provider::destroy_hybrid_gradient(hybrid);cleanup();return {status_code::unsupported_semantics,selection.reason?selection.reason:"requested gradient route is ineligible"};}
    // Conservative automatic selection may reject promotion after cold cover
    // discovery. Retain no unused panel storage in that case.
    if(!selection.use_hybrid){gradient_provider::destroy_hybrid_gradient(hybrid);hybrid=nullptr;hybrid_report={};}
    p.structure->device_edges=edges;p.source_scratch=x;p.cotangent_scratch=y;
    p.hybrid=hybrid;p.hybrid_selected=selection.use_hybrid;
    p.calculus=calculus;p.gradient_options=options;p.gradient_prepared=true;
    p.updates.persistent_bytes+=bytes+hybrid_report.persistent_bytes;
    p.updates.scratch_bytes=scratch+hybrid_report.scratch_bytes;++p.updates.gradient_preparations;
    return {};
}
status enqueue_edge_gradient(prepared_relation_pair& p,const relation_calculus_descriptor& calculus,
    const device_state_view& input,const device_state_view& cotangent,
    operand_version input_version,operand_version cotangent_version,
    execution::value_generation expected,const edge_plane_view& output,
    gradient_stamp* produced,cudaStream_t stream) noexcept {
    if(!float_gradient_contract(p))return {status_code::unsupported_numeric_policy,"gradient provider supports FP32 relation/input/output only"};
    if(input.dtype!=execution::numeric_type::f32||cotangent.dtype!=execution::numeric_type::f32)
        return {status_code::unsupported_numeric_policy,"gradient operands must be FP32"};
    auto s=reject_capture(stream);if(!s)return s;
    s=check_context(p,input.device_ordinal,stream);if(!s)return s;
    if(calculus.update!=value_update_kind::delta_add &&
        calculus.update!=value_update_kind::gradient_step)
        return {status_code::unsupported_semantics,"unknown subsequent value update"};
    // The later update style does not change this prepared VJP. Preserve every
    // gradient/axis/numeric contract while allowing both updates on one pair.
    auto gradient_calculus=calculus;
    gradient_calculus.update=p.calculus.update;
    if(!p.gradient_prepared||!equivalent(gradient_calculus,p.calculus))
        return {status_code::invalid_state,"gradient calculus not prepared"};
    if(!produced || !input_version.identity || !input_version.version ||
        !cotangent_version.identity || !cotangent_version.version)
        return {status_code::invalid_argument,"gradient provenance is incomplete"};
    if(!expected.value || expected.value!=p.report.latest_enqueued_generation.value)
        return {status_code::stale_generation,"gradient must bind current forward generation"};
    if(p.updates.gradient_launches==std::numeric_limits<std::uint64_t>::max())
        return {status_code::invalid_state,"gradient producer serial exhausted"};
    const auto& t=p.forward.semantic.topology;auto nx=t.source.extent*16,ny=t.destination.extent*16;
    if(!matching_axis(input.axis,t.source)||!matching_axis(cotangent.axis,t.destination))
        return {status_code::invalid_axis,"gradient source or cotangent axis mismatch"};
    if(cotangent.device_ordinal!=p.device)return {status_code::incompatible_device,"cotangent device mismatch"};
    if(input.count<nx||cotangent.count<ny)return {status_code::insufficient_capacity,"gradient dense capacity too small"};
    s=check_pointer(input.data,nx*4,p.device,4);if(!s)return s;
    s=check_pointer(cotangent.data,ny*4,p.device,4);if(!s)return s;
    s=check_edge_plane(p,output);if(!s)return s;
    if(protected_overlap(p,input.data,nx*4)||protected_overlap(p,cotangent.data,ny*4)||
        overlaps(output.f32_data,t.edge_count*4,input.data,nx*4)||
        overlaps(output.f32_data,t.edge_count*4,cotangent.data,ny*4))
        return {status_code::invalid_argument,"gradient operands overlap protected or output storage"};
    gradient_provider::relation_gradient_request request{};
    request.support={p.structure->device_edges,0,static_cast<std::uint32_t>(t.edge_count),
        static_cast<std::uint32_t>(t.source.extent),static_cast<std::uint32_t>(t.destination.extent)};
    request.source=static_cast<const float*>(input.data);request.cotangent=static_cast<const float*>(cotangent.data);request.output=static_cast<float*>(output.f32_data);
    request.source_capacity=input.count;request.cotangent_capacity=cotangent.count;request.output_capacity=output.count;
    request.half_rounded=calculus.gradient==gradient_arithmetic::round_operands_f16_rne;
    request.source_scratch=p.source_scratch;request.cotangent_scratch=p.cotangent_scratch;
    request.source_scratch_capacity=nx;request.cotangent_scratch_capacity=ny;request.stream=stream;
    auto before=p.hybrid?gradient_provider::inspect_hybrid_gradient(*p.hybrid):gradient_provider::hybrid_report{};
    auto result=p.hybrid?gradient_provider::enqueue_hybrid_gradient(*p.hybrid,request,p.hybrid_selected):
        gradient_provider::enqueue_relation_gradient(request);
    if(result!=gradient_contract::status_v1::success) {
        p.poisoned=true;return {status_code::cuda_failure,"gradient provider submission failed"};
    }
    ++p.updates.gradient_launches;
    if(p.hybrid) {
        auto after=gradient_provider::inspect_hybrid_gradient(*p.hybrid);
        p.updates.sparse_launches+=after.sparse_launches-before.sparse_launches;
        p.updates.wmma_launches+=after.wmma_launches-before.wmma_launches;
        p.updates.residual_launches+=after.residual_launches-before.residual_launches;
        p.updates.operand_pack_refreshes+=after.pack_refreshes-before.pack_refreshes;
    } else if(t.edge_count) {++p.updates.sparse_launches;if(request.half_rounded)++p.updates.operand_pack_refreshes;}
    p.last_gradient={t.identity,t.epoch,p.physical_order,expected,input_version,cotangent_version,
        p.updates.gradient_launches,p.incarnation,calculus.gradient};
    p.last_gradient_output=output;*produced=p.last_gradient;return {};
}
} // namespace cellerator::compute::relation

namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
// Private provider seam; public update semantics remain relation_update.hh.
cudaError_t enqueue_relation_value_update(void*,const float*,std::uint32_t,bool,float,cudaStream_t) noexcept;
cudaError_t enqueue_relation_value_update_f32(float*,void*,const float*,std::uint32_t,bool,float,cudaStream_t) noexcept;
}
namespace cellerator::compute::relation {
namespace {
bool same_version(operand_version a,operand_version b) noexcept {
    return a.identity==b.identity&&a.version==b.version;
}
bool same_stamp(const gradient_stamp& a,const gradient_stamp& b) noexcept {
    return a.producer_serial && a.producer_serial==b.producer_serial &&
        a.pair_incarnation==b.pair_incarnation && execution::same_identity(a.structure,b.structure) &&
        a.epoch.value==b.epoch.value && execution::same_identity(a.order,b.order) &&
        a.forward_generation.value==b.forward_generation.value &&
        same_version(a.input,b.input)&&same_version(a.cotangent,b.cotangent)&&a.arithmetic==b.arithmetic;
}
}
status enqueue_value_update(prepared_relation_pair& p,const value_update_request& r,cudaStream_t stream) noexcept {
    auto s=reject_capture(stream);if(!s)return s;
    s=check_context(p,r.operand.device_ordinal,stream);if(!s)return s;
    if(r.kind!=value_update_kind::delta_add&&r.kind!=value_update_kind::gradient_step)
        return {status_code::invalid_argument,"unknown value update operation"};
    if(!r.expected.value||r.expected.value!=p.report.latest_enqueued_generation.value||
        !r.next.value||r.next.value<=r.expected.value)
        return {status_code::stale_generation,"update requires exact current and strictly greater next generation"};
    s=check_edge_plane(p,r.operand);if(!s)return s;
    if(r.kind==value_update_kind::gradient_step) {
        if(!std::isfinite(r.alpha)||r.alpha<0)
            return {status_code::invalid_argument,"gradient step alpha must be finite and nonnegative"};
        if(!p.gradient_prepared||!same_stamp(r.gradient,p.last_gradient)||
            r.gradient.forward_generation.value!=r.expected.value||
            r.operand.f32_data!=p.last_gradient_output.f32_data)
            return {status_code::stale_generation,"gradient stamp or produced buffer is stale"};
    }
    s=readiness_status(p.readiness.validate_write(r.expected,r.next,stream));if(!s)return s;
    const bool f32=p.forward.semantic.arithmetic.relation_storage==execution::numeric_type::f32;
    auto result=f32?gradient_provider::enqueue_relation_value_update_f32(p.authoritative_f32,p.values,
        static_cast<const float*>(r.operand.f32_data),static_cast<std::uint32_t>(p.forward.semantic.topology.edge_count),
        r.kind==value_update_kind::gradient_step,r.alpha,stream):
        gradient_provider::enqueue_relation_value_update(p.values,static_cast<const float*>(r.operand.f32_data),
        static_cast<std::uint32_t>(p.forward.semantic.topology.edge_count),r.kind==value_update_kind::gradient_step,r.alpha,stream);
    s=readiness_status(p.readiness.publish(r.next,stream,result));
    if(result!=cudaSuccess||!s){p.poisoned=true;return result!=cudaSuccess?cuda_status(result):s;}
    p.report.latest_enqueued_generation=r.next;++p.updates.physical_updates;
    if(f32)p.report.derived_f16_generation=p.derive_f16?r.next:execution::value_generation{};
    p.last_gradient={};p.last_gradient_output={};return {};
}
} // namespace cellerator::compute::relation

namespace cellerator::compute::relation {
status begin_value_read(prepared_relation_pair& p,execution::value_generation generation,
    cudaStream_t consumer,value_read_lease* out) noexcept {
    if(p.forward.semantic.arithmetic.relation_storage!=execution::numeric_type::f16)return {status_code::unsupported_numeric_policy,"legacy lease is f16-only"};
    if(!out)return {status_code::invalid_argument,"lease output absent"};
    if(p.poisoned)return {status_code::invalid_state,"pair poisoned"};
    const auto& t=p.forward.semantic.topology;
    runtime::relation_read_ticket ticket{};
    auto s=readiness_status(p.readiness.begin_read(t.identity,t.epoch,generation,p.device,consumer,&ticket));
    if(!s){if(p.readiness.poisoned())p.poisoned=true;return s;}
    p.reader_ticket=ticket;
    p.active_lease={p.values,t.edge_count,p.physical_order,generation,ticket.nonce,p.incarnation,t.identity,t.epoch,p.device};
    *out=p.active_lease;return {};
}
status end_value_read(prepared_relation_pair& p,value_read_lease& lease,cudaStream_t consumer) noexcept {
    const auto& a=p.active_lease;
    if(!lease.nonce||lease.nonce!=a.nonce||lease.pair_incarnation!=a.pair_incarnation||
        lease.physical_f16_values!=a.physical_f16_values||lease.count!=a.count||
        !execution::same_identity(lease.order,a.order)||!execution::same_identity(lease.structure,a.structure)||
        lease.epoch.value!=a.epoch.value||lease.generation.value!=a.generation.value||lease.device_ordinal!=a.device_ordinal)
        return {status_code::invalid_argument,"lease does not match outstanding reader"};
    auto s=readiness_status(p.readiness.end_read(p.reader_ticket,consumer));
    if(!s){if(p.readiness.poisoned())p.poisoned=true;return s;}
    lease={};p.active_lease={};return {};
}
status close_relation_pair(prepared_relation_pair** slot) noexcept {
    if(!slot)return {status_code::invalid_argument,"close slot absent"};
    auto* p=*slot;if(!p)return {};
    if(p->readiness.active_reader())return {status_code::invalid_state,"unreturned value lease prevents close"};
    int prior=-1;auto s=cuda_status(cudaGetDevice(&prior));if(!s)return s;
    if(prior!=p->device){s=cuda_status(cudaSetDevice(p->device));if(!s)return s;}
    s=readiness_status(p->readiness.close());
    if(s){delete p;*slot=nullptr;}
    else if(p->readiness.poisoned())p->poisoned=true;
    if(prior>=0){auto restored=cuda_status(cudaSetDevice(prior));if(s&&!restored)s=restored;}
    return s;
}
} // namespace cellerator::compute::relation

namespace cellerator::compute::relation {
namespace {
status check_logical_atom(const prepared_relation_pair& p,
    const execution::atom_plane::relation_value_atom_plane_v1& atom, const native_atom_association& association) noexcept {
    if(!float_gradient_contract(p))return {status_code::unsupported_numeric_policy,"atom route supports only the FP32 relation contract"};
    namespace vp=execution::projection_value_plane;
    if(!execution::atom_plane::validate_relation_value_atom_plane_v1(atom,{},nullptr))
        return {status_code::invalid_argument,"invalid existing atom-plane contract"};
    const auto& plane=*atom.values;const auto& component=plane.components[0];
    const auto& topology=p.forward.semantic.topology;
    if(plane.primary_mode!=vp::value_primary_mode_v1::logical || plane.component_count!=1)
        return {status_code::unsupported_semantics,"native atom route requires logical primary ownership"};
    if(!execution::same_identity(plane.logical_edge_order,topology.logical_edge_order) ||
       plane.logical_edge_count!=topology.edge_count ||
       plane.structure_epoch_value.value!=topology.epoch.value)
        return {status_code::stale_structure,"atom topology differs from prepared structure"};
    if(!equivalent(association.native_relation,p.forward.semantic) ||
       !execution::same_structure_handle(association.atom_structure,plane.structure) ||
       !execution::same_axis_identity(association.atom_source,atom.structural_binding->structure->source_axis) ||
       !execution::same_axis_identity(association.atom_destination,atom.structural_binding->structure->destination_axis))
        return {status_code::stale_structure,"explicit atom-to-native structure association differs"};
    if(component.location.residency!=execution::residency_kind::device || component.location.device_ordinal!=p.device)
        return {status_code::incompatible_device,"atom values must reside on instance device"};
    if(plane.numeric.storage!=execution::numeric_type::f16 || plane.numeric.dequantized!=execution::numeric_type::f32 ||
       plane.numeric.accumulation!=execution::numeric_type::f32 || plane.quantization.kind!=execution::quantization_kind::none)
        return {status_code::unsupported_numeric_policy,"atom route requires unquantized f16/f32"};
    if(component.value_bytes<topology.edge_count*2)return {status_code::insufficient_capacity,"atom value bytes too small"};
    for(std::uint64_t i=0;i<topology.edge_count;++i)
        if(component.slot_to_logical_edge[i]!=i)
            return {status_code::incompatible_order,"logical atom cannot contain holes or permutations"};
    return {};
}
__global__ void scatter_atom_gradient(const float* physical,const std::uint32_t* map,float* logical,std::uint32_t count){
    auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)logical[map[i]]=physical[i];
}
}
status publish_atom_values(prepared_relation_pair& p,const execution::atom_plane::relation_value_atom_plane_v1& atom,const native_atom_association& association,cudaStream_t stream) noexcept {
    auto s=check_logical_atom(p,atom,association);if(!s)return s;
    const auto& t=p.forward.semantic.topology;const auto& c=atom.values->components[0];
    return publish_values(p,{c.values,c.slot_count,t.identity,t.epoch,t.logical_edge_order,atom.expected_generation,p.device},stream);
}
status enqueue_atom_gradient(prepared_relation_pair& p,const relation_calculus_descriptor& calculus,
    const device_state_view& input,const device_state_view& cotangent,operand_version input_version,operand_version cotangent_version,
    const execution::atom_plane::gradient_atom_plane_v1& gradient,const native_atom_association& association,const edge_plane_view& scratch,gradient_stamp* produced,cudaStream_t stream) noexcept {
    if(!float_gradient_contract(p))return {status_code::unsupported_numeric_policy,"atom gradient supports only the FP32 relation contract"};
    if(input.dtype!=execution::numeric_type::f32||cotangent.dtype!=execution::numeric_type::f32)
        return {status_code::unsupported_numeric_policy,"atom gradient operands must be FP32"};
    if(!execution::atom_plane::validate_gradient_atom_plane_v1(gradient,{}))
        return {status_code::invalid_argument,"invalid existing gradient atom contract"};
    auto s=check_logical_atom(p,*gradient.primal,association);if(!s)return s;
    if(gradient.component_count!=1)return {status_code::unsupported_semantics,"one trainable logical gradient required"};
    const auto& c=gradient.components[0];auto count=p.forward.semantic.topology.edge_count;
    if(c.gradient_bytes<count*4)return {status_code::insufficient_capacity,"logical gradient capacity too small"};
    s=check_pointer(c.gradients,count*4,p.device,4);if(!s)return s;
    if(protected_overlap(p,c.gradients,count*4) || overlaps(c.gradients,count*4,scratch.f32_data,count*4) ||
       overlaps(c.gradients,count*4,input.data,input.count*4) || overlaps(c.gradients,count*4,cotangent.data,cotangent.count*4))
        return {status_code::invalid_argument,"logical gradient overlaps protected operands"};
    s=enqueue_edge_gradient(p,calculus,input,cotangent,input_version,cotangent_version,gradient.primal_generation,scratch,produced,stream);if(!s)return s;
    if(count){scatter_atom_gradient<<<(count+255)/256,256,0,stream>>>(static_cast<const float*>(scratch.f32_data),p.structure->logical_map,static_cast<float*>(c.gradients),count);
        auto e=cudaPeekAtLastError();if(e!=cudaSuccess){p.poisoned=true;return cuda_status(e);}}
    return {};
}
}

namespace cellerator::compute::relation {
namespace {
template<class T> __global__ void gather_authoritative(const T* logical,const std::uint32_t* map,T* packed,std::uint32_t count){
    auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)packed[i]=logical[map[i]];
}
__global__ void gather_authoritative_f32(const float* logical,const std::uint32_t* map,float* packed,__half* derived,std::uint32_t count){
    auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){float value=logical[map[i]];packed[i]=value;if(derived)derived[i]=__float2half_rn(value);}
}
template<class T> status publish_typed_values(prepared_relation_pair& p,
    const device_numeric_values_binding<T>& binding,cudaStream_t stream,
    execution::numeric_type relation_type) noexcept {
    if(p.forward.semantic.arithmetic.relation_storage!=relation_type)
        return {status_code::unsupported_numeric_policy,"value binding dtype differs from prepared relation storage"};
    auto s=check_context(p,binding.device_ordinal,stream);if(!s)return s;
    cudaStreamCaptureStatus capture{};s=cuda_status(cudaStreamIsCapturing(stream,&capture));if(!s)return s;
    if(capture!=cudaStreamCaptureStatusNone)return {status_code::unsupported_semantics,"publication capture unsupported"};
    const auto& t=p.forward.semantic.topology;
    if(!execution::same_identity(binding.structure,t.identity)||binding.epoch.value!=t.epoch.value)
        return {status_code::stale_structure,"typed publication topology differs"};
    if(!execution::same_identity(binding.logical_edge_order,t.logical_edge_order))
        return {status_code::incompatible_order,"typed logical order differs"};
    if(!binding.generation.value||binding.generation.value<=p.report.latest_enqueued_generation.value)
        return {status_code::stale_generation,"publication requires a strictly increasing nonzero generation"};
    if(binding.count<t.edge_count)return {status_code::insufficient_capacity,"typed publication capacity too small"};
    std::uint64_t bytes=0;
    if(!checked_bytes(t.edge_count,sizeof(T),bytes))return {status_code::invalid_shape,"typed value range overflows 64 bits"};
    s=check_pointer(binding.data,bytes,p.device,alignof(T));if(!s)return s;
    if(protected_overlap(p,binding.data,bytes))return {status_code::invalid_argument,"typed input aliases instance storage"};
    s=readiness_status(p.readiness.validate_write(p.report.latest_enqueued_generation,binding.generation,stream));if(!s)return s;
    cudaError_t submitted=cudaSuccess;
    if(t.edge_count){
        if constexpr(std::is_same_v<T,float>)
            gather_authoritative_f32<<<(t.edge_count+255)/256,256,0,stream>>>(binding.data,p.structure->logical_map,p.authoritative_f32,static_cast<__half*>(p.values),t.edge_count);
        else
            gather_authoritative<<<(t.edge_count+255)/256,256,0,stream>>>(binding.data,p.structure->logical_map,p.authoritative_f64,static_cast<std::uint32_t>(t.edge_count));
        submitted=cudaPeekAtLastError();
    }
    s=readiness_status(p.readiness.publish(binding.generation,stream,submitted));
    if(submitted!=cudaSuccess||!s){p.poisoned=true;return submitted!=cudaSuccess?cuda_status(submitted):s;}
    p.report.latest_enqueued_generation=binding.generation;++p.report.value_refreshes;
    p.report.derived_f16_generation=(relation_type==execution::numeric_type::f32&&p.derive_f16)?binding.generation:execution::value_generation{};
    p.last_gradient={};p.last_gradient_output={};return {};
}
}
status publish_f32_values(prepared_relation_pair& p,const device_f32_values_binding& binding,cudaStream_t stream) noexcept {
    return publish_typed_values(p,binding,stream,execution::numeric_type::f32);
}
status publish_f64_values(prepared_relation_pair& p,const device_f64_values_binding& binding,cudaStream_t stream) noexcept {
    return publish_typed_values(p,binding,stream,execution::numeric_type::f64);
}
status enqueue_derived_f16(prepared_relation_pair& p,const operation_descriptor& op,const device_state_view& input,
    const device_result_view& output,execution::value_generation expected,cudaStream_t stream) noexcept {
    if(op.arithmetic.relation_storage!=execution::numeric_type::f32||!float_gradient_contract(p)||!p.derive_f16)
        return {status_code::unsupported_numeric_policy,"derived f16 execution was not explicitly prepared"};
    auto s=check_launch(p,op,input,output,expected,stream);if(!s)return s;
    if(p.report.derived_f16_generation.value!=expected.value)return {status_code::stale_generation,"derived projection generation differs"};
    s=submit_half(p,op.direction,input.data,output.data);if(!s)p.poisoned=true;return s;
}
}

namespace cellerator::compute::relation {
status replace_relation_epoch(prepared_relation_pair** slot,const operation_descriptor& forward,
    const operation_descriptor& transpose,const csr_host_view& topology,const preparation_options& options,cudaStream_t stream) noexcept {
    if(!slot||!*slot)return {status_code::invalid_argument,"epoch replacement requires owned instance"};
    auto& old=**slot;auto s=check_context(old,options.device_ordinal,stream);if(!s)return s;
    s=reject_capture(stream);if(!s)return s;
    if(old.readiness.active_reader())return {status_code::invalid_state,"live borrow prevents epoch replacement"};
    if(!execution::same_identity(forward.topology.identity,old.forward.semantic.topology.identity) ||
       forward.topology.epoch.value<=old.forward.semantic.topology.epoch.value)
        return {status_code::stale_structure,"replacement requires same identity and strictly newer epoch"};
    prepared_relation_pair* candidate=nullptr;
    s=prepare_relation_pair(forward,transpose,topology,options,stream,&candidate);if(!s)return s;
    std::unique_ptr<prepared_relation_pair> owned(candidate);
    s=close_relation_pair(slot);if(!s)return s;
    *slot=owned.release();return {};
}
}
