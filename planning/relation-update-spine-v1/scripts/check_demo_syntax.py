#!/usr/bin/env python3
"""Syntax-only check of prospective demo using declaration-only temporary shims.
This does NOT link Cellerator, invoke CUDA or validate installed API compatibility.
Shims have no kernel, allocator or runtime implementations and cannot run the demo.
"""
from pathlib import Path
import argparse,json,shutil,subprocess,tempfile,sys
sys.dont_write_bytecode=True
PACKAGE=Path(__file__).resolve().parents[1]
CUDA=r"""#pragma once
#include <cstddef>
struct CUstream_st; using cudaStream_t=CUstream_st*;
using cudaError_t=int;
constexpr cudaError_t cudaSuccess=0;
constexpr unsigned cudaStreamNonBlocking=1;
enum cudaMemcpyKind {cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost};
struct cudaDeviceProp {char name[256];int major,minor;};
cudaError_t cudaGetDeviceCount(int*);cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*,unsigned);
cudaError_t cudaStreamDestroy(cudaStream_t);cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaMalloc(void**,std::size_t);cudaError_t cudaFree(void*);
cudaError_t cudaMallocHost(void**,std::size_t);cudaError_t cudaFreeHost(void*);
cudaError_t cudaMemcpyAsync(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t);
"""
SEMANTIC=r"""#pragma once
#include <cstdint>
namespace cellerator::execution {
constexpr std::uint16_t biological_abi_version=1;
enum class serialized_record_kind:std::uint16_t {persistent_axis_identity=1};
struct serialized_record_header {std::uint16_t schema_version;serialized_record_kind kind;std::uint32_t byte_count;};
template<class T> struct persistent_identity {std::uint64_t low,high;};
struct structure_tag;struct order_tag;struct domain_tag;struct geometry_tag;struct partition_tag;
using structure_id=persistent_identity<structure_tag>;using order_id=persistent_identity<order_tag>;
struct persistent_axis_identity {serialized_record_header header;persistent_identity<domain_tag> domain;order_id order;persistent_identity<geometry_tag> geometry;persistent_identity<partition_tag> partition;};
struct structure_epoch {std::uint64_t value;};struct value_generation {std::uint64_t value;};
}
namespace cellerator::compute::relation {
enum class status_code {success,stale_generation};
struct status {status_code code{};const char* message=nullptr;explicit operator bool() const noexcept;};
struct axis_descriptor {execution::persistent_axis_identity identity{};std::uint64_t extent=0;};
struct topology_descriptor {execution::structure_id identity{};execution::structure_epoch epoch{};axis_descriptor source,destination;execution::order_id logical_edge_order{};std::uint64_t edge_count=0;};
enum class orientation {forward,transpose};struct arithmetic_policy {};
struct operation_descriptor {topology_descriptor topology;orientation direction=orientation::forward;arithmetic_policy arithmetic{};std::uint32_t dense_width=1;};
}
"""
PREPARED=r"""#pragma once
#include <cuda_runtime_api.h>
#include <Cellerator/compute/operation/relation_semantics.hh>
namespace cellerator::compute::relation {
struct prepared_relation_pair;
struct csr_host_view {const std::uint32_t* offsets;std::uint64_t offsets_capacity;const std::uint32_t* sources;std::uint64_t sources_capacity;};
struct preparation_options {int device;std::uint64_t scratch;};
struct device_state_view {const void* data;std::uint64_t count;axis_descriptor axis;int device;};
struct device_result_view {void* data;std::uint64_t count;axis_descriptor axis;int device;};
struct device_values_binding {const void* data;std::uint64_t count;execution::structure_id structure;execution::structure_epoch epoch;execution::order_id order;execution::value_generation generation;int device;};
struct preparation_report {std::uint64_t topology_preparations=0,value_refreshes=0;execution::value_generation latest_enqueued_generation{};std::uint64_t accepted_forward_launches=0,accepted_transpose_launches=0;};
status prepare_relation_pair(const operation_descriptor&,const operation_descriptor&,const csr_host_view&,const preparation_options&,cudaStream_t,prepared_relation_pair**) noexcept;
void destroy(prepared_relation_pair*) noexcept;
status publish_values(prepared_relation_pair&,const device_values_binding&,cudaStream_t) noexcept;
status enqueue(prepared_relation_pair&,const operation_descriptor&,const device_state_view&,const device_result_view&,execution::value_generation,cudaStream_t) noexcept;
}
"""
def check(compiler='c++'):
    if not shutil.which(compiler):raise ValueError('C++ compiler unavailable')
    with tempfile.TemporaryDirectory(prefix='ce-ru1-declarations-') as d:
        root=Path(d)
        def write(path,text):
            f=root/path;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text)
        write('cuda_runtime_api.h',CUDA)
        write('Cellerator/compute/operation/relation_semantics.hh',SEMANTIC)
        write('Cellerator/compute/operation/prepared_relation.hh',PREPARED)
        for h in ['relation_calculus.hh','relation_update.hh']:
            write('Cellerator/compute/operation/'+h,(PACKAGE/'contracts'/h).read_text())
        write('Cellerator/compiler/sema/relation_update_spine_bridge.hh',(PACKAGE/'contracts/relation_update_spine_bridge.hh').read_text())
        demo=PACKAGE.parents[1]/'examples/relation_update_spine_v1/regulatory_learning.cc'
        cmd=[compiler,'-std=c++20','-Wall','-Wextra','-Werror','-fsyntax-only','-I',str(root),str(demo)]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        result={'status':'syntax_only_pass' if r.returncode==0 else 'failed','compiler':compiler,
                'command':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,
                'temporary_declaration_shims':True,'linked_cellerator':False,'cuda_compiled':False,'gpu_executed':False}
        if r.returncode:raise ValueError(r.stderr)
        return result
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--compiler',default='c++');o=a.parse_args()
    print(json.dumps(check(o.compiler),indent=2))
