#include <Cellerator/compute/operation/native_numeric/device_linear.hh>
#include <Cellerator/compute/operation/prepared_relation.hh>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace ce = cellerator::compute::relation;
namespace nn = cellerator::compute::native_numeric;
namespace ex = cellerator::execution;
namespace fs = std::filesystem;
using u64 = std::uint64_t;
using clock_type = std::chrono::steady_clock;

namespace {
void demand(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void cuda_check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
void relation_check(ce::status s, const char* where) {
    if (!s) throw std::runtime_error(std::string(where) + ": " + (s.message ? s.message : "unknown relation error"));
}
u64 mul(u64 a, u64 b, const char* what) {
    demand(b == 0 || a <= std::numeric_limits<u64>::max() / b, std::string("size overflow: ") + what);
    return a * b;
}
std::size_t host_size(u64 n, const char* what) {
    demand(n <= std::numeric_limits<std::size_t>::max(), std::string("host size overflow: ") + what);
    return static_cast<std::size_t>(n);
}
u64 integer(std::string_view s, const char* what) {
    u64 n = 0; auto r = std::from_chars(s.data(), s.data() + s.size(), n);
    demand(r.ec == std::errc{} && r.ptr == s.data() + s.size(), std::string("invalid integer: ") + what);
    return n;
}
struct args {
    fs::path input, output;
    u64 sources=0, destinations=0, features=0, edges=0, warmups=5, repeats=20;
    bool self_test=false, help=false;
};
args parse(int argc, char** argv) {
    args a;
    for (int i=1; i<argc; ++i) {
        std::string_view k(argv[i]);
        if (k=="--self-test") { a.self_test=true; continue; }
        if (k=="--help") { a.help=true; continue; }
        demand(i+1<argc, "missing value for " + std::string(k));
        std::string_view v(argv[++i]);
        if(k=="--input") a.input=v;
        else if(k=="--output") a.output=v;
        else if(k=="--sources") a.sources=integer(v,"sources");
        else if(k=="--destinations") a.destinations=integer(v,"destinations");
        else if(k=="--features") a.features=integer(v,"features");
        else if(k=="--edges") a.edges=integer(v,"edges");
        else if(k=="--warmups") a.warmups=integer(v,"warmups");
        else if(k=="--repeats") a.repeats=integer(v,"repeats");
        else throw std::runtime_error("unknown option: " + std::string(k));
    }
    return a;
}
void help() {
    std::cout << "Usage: ceNativeNeighborhoodMoments --input DIR --output DIR --sources N --destinations M --features F --edges E [--warmups K] [--repeats R]\n"
                 "       ceNativeNeighborhoodMoments --self-test | --help\n";
}
template<class T> std::vector<T> read_raw(const fs::path& p, u64 n, u64& total) {
    const u64 bytes=mul(n,sizeof(T),"input file");
    demand(bytes<=static_cast<u64>(std::numeric_limits<std::streamsize>::max()),"input file exceeds stream API limit");
    std::vector<T> v(host_size(n,"input"));
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    demand(f.good(),"cannot open input: "+p.string());
    demand(f.tellg()>=0 && static_cast<u64>(f.tellg())==bytes,"input byte count mismatch: "+p.string());
    f.seekg(0);
    if(bytes) f.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(bytes));
    demand(f.good(),"input read failed: "+p.string());
    total+=bytes;
    return v;
}
template<class T> void write_raw(const fs::path& p,const std::vector<T>& v,u64& total) {
    const u64 bytes=mul(static_cast<u64>(v.size()),sizeof(T),"output file");
    demand(bytes<=static_cast<u64>(std::numeric_limits<std::streamsize>::max()),"output file exceeds stream API limit");
    std::ofstream f(p,std::ios::binary|std::ios::trunc);
    demand(f.good(),"cannot create output: "+p.string());
    if(bytes) f.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(bytes));
    demand(f.good(),"output write failed: "+p.string());
    total+=bytes;
}
struct stream_owner {
    cudaStream_t s=nullptr;
    stream_owner(){cuda_check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"create nonblocking stream");}
    ~stream_owner(){if(s)cudaStreamDestroy(s);}
};
struct event_owner {
    cudaEvent_t e=nullptr;
    event_owner(){cuda_check(cudaEventCreate(&e),"create CUDA event");}
    ~event_owner(){if(e)cudaEventDestroy(e);}
};
struct raw_buffer {
    void* p=nullptr; u64 bytes=0;
    explicit raw_buffer(u64 b):bytes(b){cuda_check(cudaMalloc(&p,host_size(std::max<u64>(b,1),"device allocation")),"allocate CSR data");}
    ~raw_buffer(){if(p)cudaFree(p);}
};
struct vec {
    nn::resident_vector v{};
    explicit vec(u64 n){cuda_check(nn::allocate(&v,n,nn::device_representation::f32,0),"allocate FP32 vector");}
    ~vec(){if(v.data)nn::release(&v);}
};
ce::axis_descriptor axis(u64 base,u64 extent) {
    ex::persistent_axis_identity id{};
    id.header={ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(id)};
    id.domain={base,1}; id.order={base+1,1}; id.geometry={base+2,1}; id.partition={base+3,1};
    return {id,extent};
}
struct dims {u64 n,m,f,e;};
struct inputs {std::vector<std::uint32_t> rows,cols; std::vector<float> weights,left,right;};
using outputs=std::array<std::vector<float>,10>;
constexpr std::array<const char*,10> names{"mean_left","mean_right","second_left","cross_second","second_right","variance_left","variance_right","covariance","affine_second_left","affine_cross_second"};
void validate(const dims& d,const inputs& x) {
    demand(d.n&&d.m&&d.f,"dimensions must be positive");
    demand(d.n<=UINT32_MAX-128 && d.m<=UINT32_MAX-128 && d.e<=UINT32_MAX-128 && d.f<=UINT32_MAX,
           "shape exceeds the prepared relation descriptor limits");
    demand(std::endian::native==std::endian::little,"native probe requires little-endian typed arrays");
    demand(x.rows.size()==d.m+1 && x.cols.size()==d.e && x.weights.size()==d.e,"CSR shapes do not match CLI");
    const u64 ns=mul(d.n,d.f,"source elements");
    demand(x.left.size()==ns&&x.right.size()==ns,"dense inputs must be source_count x features");
    demand(x.rows.front()==0&&x.rows.back()==d.e,"CSR offsets must start at zero and end at edge count");
    for(std::size_t i=1;i<x.rows.size();++i)demand(x.rows[i-1]<=x.rows[i],"CSR offsets must be monotonic");
    for(auto c:x.cols)demand(c<d.n,"CSR source index outside source axis");
    demand(d.f<=UINT32_MAX,"feature width does not fit operation descriptor");
    (void)mul(d.m,d.f,"destination elements");
}
struct buffers {
    std::array<std::unique_ptr<vec>,2> in;
    std::array<std::unique_ptr<vec>,3> products;
    std::array<std::unique_ptr<vec>,10> out;
    std::unique_ptr<vec> scratch;
    u64 bytes=0;
    buffers(const dims& d) {
        const u64 ns=mul(d.n,d.f,"source elements"), nm=mul(d.m,d.f,"destination elements");
        auto add=[&](u64 n){bytes+=mul(n,sizeof(float),"tracked vector bytes");};
        for(auto& p:in){p=std::make_unique<vec>(ns);add(ns);}
        for(auto& p:products){p=std::make_unique<vec>(ns);add(ns);}
        for(auto& p:out){p=std::make_unique<vec>(nm);add(nm);}
        scratch=std::make_unique<vec>(nm);add(nm);
    }
};
struct relation {
    ce::topology_descriptor topology{};
    ce::operation_descriptor op{};
    ce::prepared_relation_pair* handle=nullptr;
    std::unique_ptr<raw_buffer> weights;
    u64 bytes=0;
    relation()=default;
    relation(const relation&)=delete;
    relation& operator=(const relation&)=delete;
    relation(relation&& other) noexcept
        : topology(other.topology),op(other.op),handle(std::exchange(other.handle,nullptr)),
          weights(std::move(other.weights)),bytes(other.bytes) {}
    relation& operator=(relation&&)=delete;
    ~relation(){if(handle)ce::destroy(handle);}
};
relation prepare(const dims& d,const inputs& x,std::unique_ptr<raw_buffer> weights,cudaStream_t s,double* prep_ms=nullptr) {
    relation r;
    r.topology.identity={0x4d4f4d454e545331ULL,0x43454c4c45524154ULL};
    r.topology.epoch={1}; r.topology.source=axis(0x10000,d.n);r.topology.destination=axis(0x20000,d.m);
    r.topology.logical_edge_order={0x454447454f524431ULL,0x43454c4c45524154ULL};r.topology.edge_count=d.e;
    r.op.topology=r.topology;r.op.direction=ce::orientation::forward;r.op.dense_width=static_cast<std::uint32_t>(d.f);
    r.op.arithmetic.relation_storage=ex::numeric_type::f32;r.op.arithmetic.input_storage=ex::numeric_type::f32;
    r.op.arithmetic.multiply=ex::numeric_type::f32;r.op.arithmetic.accumulation=ex::numeric_type::f32;
    r.op.arithmetic.output_storage=ex::numeric_type::f32;r.op.arithmetic.permit_fma=true;r.op.arithmetic.permit_reassociation=true;
    relation_check(ce::validate(r.op),"validate forward");
    auto transpose=r.op;transpose.direction=ce::orientation::transpose;relation_check(ce::validate(transpose),"validate transpose");
    const u64 wb=mul(x.weights.size(),sizeof(float),"weight storage");
    demand(weights!=nullptr,"relation values storage is required");
    r.weights=std::move(weights);
    r.bytes=std::max<u64>(wb,1);
    const auto preparation_start=clock_type::now();
    relation_check(ce::prepare_relation_pair(r.op,transpose,{x.rows.data(),x.rows.size(),x.cols.data(),x.cols.size()},{0,0,false},s,&r.handle),"prepare paired CSR");
    ce::device_f32_values_binding binding{static_cast<const float*>(r.weights->p),d.e,r.topology.identity,r.topology.epoch,r.topology.logical_edge_order,{1},0};
    relation_check(ce::publish_f32_values(*r.handle,binding,s),"publish relation values generation 1");
    cuda_check(cudaStreamSynchronize(s),"finish relation publication");
    if(prep_ms)*prep_ms=std::chrono::duration<double,std::milli>(clock_type::now()-preparation_start).count();
    return r;
}
void apply(relation& r,const vec& x,vec& y,cudaStream_t s) {
    relation_check(ce::enqueue(*r.handle,r.op,{x.v.data,x.v.elements,r.topology.source,0},
        {y.v.data,y.v.elements,r.topology.destination,0},{1},s),"enqueue prepared relation");
}
void multiply(const vec& a,const vec& b,vec& out,cudaStream_t s,const char* where) {
    cuda_check(nn::enqueue_elementwise_multiply(a.v,b.v,out.v,s),where);
}
void affine(float a,const vec& x,float b,const vec& y,vec& out,cudaStream_t s,const char* where) {
    cuda_check(nn::enqueue_axpby(a,x.v,b,y.v,out.v,s),where);
}
void compose(relation& r,buffers& b,cudaStream_t s) {
    multiply(*b.in[0],*b.in[0],*b.products[0],s,"square left input");
    multiply(*b.in[0],*b.in[1],*b.products[1],s,"multiply input fields");
    multiply(*b.in[1],*b.in[1],*b.products[2],s,"square right input");
    apply(r,*b.in[0],*b.out[0],s);apply(r,*b.in[1],*b.out[1],s);
    apply(r,*b.products[0],*b.out[2],s);apply(r,*b.products[1],*b.out[3],s);apply(r,*b.products[2],*b.out[4],s);
    multiply(*b.out[0],*b.out[0],*b.scratch,s,"square left mean");
    affine(1,*b.out[2],-1,*b.scratch,*b.out[5],s,"left centered second moment");
    multiply(*b.out[1],*b.out[1],*b.scratch,s,"square right mean");
    affine(1,*b.out[4],-1,*b.scratch,*b.out[6],s,"right centered second moment");
    multiply(*b.out[0],*b.out[1],*b.scratch,s,"multiply means");
    affine(1,*b.out[3],-1,*b.scratch,*b.out[7],s,"centered cross moment");
    affine(2,*b.out[2],-1,*b.out[0],*b.out[8],s,"adjust left second moment");
    affine(2,*b.out[3],-1,*b.out[1],*b.out[9],s,"adjust cross second moment");
}
void upload_fields(buffers& b,const inputs& x,cudaStream_t s) {
    if(x.left.empty())return;
    cuda_check(cudaMemcpyAsync(b.in[0]->v.data,x.left.data(),x.left.size()*sizeof(float),cudaMemcpyHostToDevice,s),"upload left");
    cuda_check(cudaMemcpyAsync(b.in[1]->v.data,x.right.data(),x.right.size()*sizeof(float),cudaMemcpyHostToDevice,s),"upload right");
}
outputs download(buffers& b,const dims& d,cudaStream_t s) {
    outputs y; const u64 n=mul(d.m,d.f,"output elements"), bytes=mul(n,sizeof(float),"output bytes");
    for(std::size_t i=0;i<y.size();++i){y[i].resize(host_size(n,"output"));if(n)cuda_check(cudaMemcpyAsync(y[i].data(),b.out[i]->v.data,host_size(bytes,"output copy"),cudaMemcpyDeviceToHost,s),"download result");}
    cuda_check(cudaStreamSynchronize(s),"finish downloads");return y;
}
bool close(float a,double b){return std::isfinite(a)&&std::abs(static_cast<double>(a)-b)<=1e-6;}
u64 peak_rss_bytes() {
#if defined(__linux__)
    rusage usage{};
    if(getrusage(RUSAGE_SELF,&usage)==0 && usage.ru_maxrss>0)
        return static_cast<u64>(usage.ru_maxrss)*1024u;
#endif
    return 0;
}
void self_test() {
    dims d{2,3,131,3};
    inputs x;
    x.rows={0,1,1,3};x.cols={0,0,1};x.weights={1,.25f,.75f};
    x.left.resize(host_size(mul(d.n,d.f,"self-test features"),"self-test features"));
    x.right.resize(x.left.size());
    for(u64 f=0;f<d.f;++f) {
        if(f==0) { x.left[f]=x.left[d.f+f]=4.0f; }
        else if(f==1) { x.left[f]=x.left[d.f+f]=0.0f; }
        else { x.left[f]=static_cast<float>(static_cast<int>(f%9)-4);x.left[d.f+f]=static_cast<float>(static_cast<int>(f%5)-2); }
        if(f==1) { x.right[f]=x.right[d.f+f]=0.0f; }
        else { x.right[f]=static_cast<float>(static_cast<int>(f%7)-3);x.right[d.f+f]=static_cast<float>(static_cast<int>(f%11)-5); }
    }
    validate(d,x);cuda_check(cudaSetDevice(0),"select device");stream_owner s;buffers b(d);
    auto weights=std::make_unique<raw_buffer>(mul(d.e,sizeof(float),"self-test weights"));upload_fields(b,x,s.s);
    cuda_check(cudaMemcpyAsync(weights->p,x.weights.data(),host_size(mul(d.e,sizeof(float),"test weight copy"),"test weight copy"),cudaMemcpyHostToDevice,s.s),"upload self-test weights");
    cuda_check(cudaStreamSynchronize(s.s),"finish self-test uploads");
    auto r=prepare(d,x,std::move(weights),s.s);compose(r,b,s.s);auto y=download(b,d,s.s);
    for(u64 row=0;row<d.m;++row) for(u64 f=0;f<d.f;++f) {
        double ml=0,mr=0,sl=0,sr=0,cross=0;
        for(u64 e=x.rows[row];e<x.rows[row+1];++e) {
            const u64 source=x.cols[e],ix=source*d.f+f;const double w=x.weights[e],left=x.left[ix],right=x.right[ix];
            ml+=w*left;mr+=w*right;sl+=w*left*left;sr+=w*right*right;cross+=w*left*right;
        }
        const u64 ix=row*d.f+f;
        demand(close(y[0][ix],ml)&&close(y[1][ix],mr)&&close(y[2][ix],sl)&&close(y[3][ix],cross)&&close(y[4][ix],sr),"analytic raw moment mismatch");
        demand(close(y[5][ix],sl-ml*ml)&&close(y[6][ix],sr-mr*mr)&&close(y[7][ix],cross-ml*mr),"analytic centered quantity mismatch");
        demand(close(y[8][ix],2*sl-ml)&&close(y[9][ix],2*cross-mr),"analytic adjusted quantity mismatch");
    }
    std::cout<<"NATIVE_NEIGHBORHOOD_MOMENTS_SELF_TEST_PASS\n";
}
void write_metrics(const fs::path& p,const dims& d,const args& a,const cudaDeviceProp& prop,const std::vector<double>& events,
 const std::vector<double>& walls,const std::array<double,11>& times,u64 dev,u64 peak_rss,u64 host_in,u64 host_out,const ce::preparation_report& r) {
    std::ofstream f(p);demand(f.good(),"cannot write metrics.json");
    auto arr=[&](const std::vector<double>& v){f<<'[';for(std::size_t i=0;i<v.size();++i){if(i)f<<',';f<<v[i];}f<<']';};
    f<<std::setprecision(9)<<"{\n\"schema_version\":1,\n\"dimensions\":{\"sources\":"<<d.n<<",\"destinations\":"<<d.m<<",\"features\":"<<d.f<<",\"edges\":"<<d.e<<"},\n"
     <<"\"runs\":{\"warmups\":"<<a.warmups<<",\"repeats\":"<<a.repeats<<"},\n"
     <<"\"native_policy\":{\"relation\":\"FP32\",\"input\":\"FP32\",\"multiply\":\"FP32\",\"accumulation\":\"FP32\",\"output\":\"FP32\",\"tensor_cores\":false},\n"
     <<"\"gpu\":{\"name\":\""<<prop.name<<"\",\"major\":"<<prop.major<<",\"minor\":"<<prop.minor<<"},\n"
     <<"\"timing_scope\":{\"resident_ms\":\"CUDA event, complete composition; measured iterations exclude stage diagnostic\",\"resident_wall_ms\":\"host enqueue through stream sync for same composition\",\"h2d_ms\":\"CUDA event interval for fields and relation values\",\"h2d_wall_ms\":\"host enqueue through completion of all input transfers\",\"d2h_ms\":\"output vector allocation, copy enqueue and synchronization\",\"process_total_ms\":\"inside run from input reading through raw output writes; excludes process startup and metrics write\",\"diagnostic_stage_ms\":\"one extra iteration; start-to-products, products-to-relations, relations-to-centered outputs\"},\n"
     <<"\"resident_ms\":";arr(events);f<<",\n\"resident_wall_ms\":";arr(walls);
    f<<",\n\"preparation_ms\":"<<times[0]<<",\"allocation_ms\":"<<times[1]<<",\"input_read_ms\":"<<times[2]<<",\"h2d_ms\":"<<times[3]<<",\"h2d_wall_ms\":"<<times[4]<<",\"d2h_ms\":"<<times[5]
     <<",\"output_write_ms\":"<<times[6]<<",\"process_total_ms\":"<<times[7]<<",\"diagnostic_stage_ms\":["<<times[8]<<','<<times[9]<<','<<times[10]<<"],\n"
     <<"\"memory_bytes\":{\"tracked_device_allocations\":"<<dev<<",\"native_process_peak_rss\":"<<peak_rss<<",\"host_input_files\":"<<host_in<<",\"host_output_files\":"<<host_out<<"},\n"
     <<"\"preparation_report\":{\"topology_preparations\":"<<r.topology_preparations<<",\"value_refreshes\":"<<r.value_refreshes<<",\"accepted_forward_launches\":"<<r.accepted_forward_launches
     <<",\"shared_structural_bytes\":"<<r.shared_structural_bytes<<",\"instance_value_bytes\":"<<r.instance_value_bytes<<",\"forward_candidate\":\""<<(r.forward_candidate?r.forward_candidate:"")<<"\"}\n}\n";
    demand(f.good(),"metrics write failed");
}
int run(const args& a) {
    const auto total_start=clock_type::now();
    dims d{a.sources,a.destinations,a.features,a.edges};
    demand(a.warmups<=100000&&a.repeats>0&&a.repeats<=100000,"invalid warmups/repeats");
    const u64 ns=mul(d.n,d.f,"source shape");(void)mul(d.m,d.f,"destination shape");
    demand(!a.input.empty()&&!a.output.empty(),"input and output directories are required");
    u64 host_input=0;
    auto t=clock_type::now();
    inputs x{read_raw<std::uint32_t>(a.input/"row_offsets.u32",d.m+1,host_input),
             read_raw<std::uint32_t>(a.input/"column_indices.u32",d.e,host_input),
             read_raw<float>(a.input/"weights.f32",d.e,host_input),
             read_raw<float>(a.input/"left.f32",ns,host_input),
             read_raw<float>(a.input/"right.f32",ns,host_input)};
    validate(d,x);
    const double read_ms=std::chrono::duration<double,std::milli>(clock_type::now()-t).count();
    fs::create_directories(a.output);cuda_check(cudaSetDevice(0),"select GPU");cudaDeviceProp prop{};cuda_check(cudaGetDeviceProperties(&prop,0),"GPU properties");
    stream_owner stream;
    t=clock_type::now();buffers b(d);auto weights=std::make_unique<raw_buffer>(mul(d.e,sizeof(float),"relation values"));const double alloc_ms=std::chrono::duration<double,std::milli>(clock_type::now()-t).count();
    const u64 weight_bytes=mul(d.e,sizeof(float),"weight upload");
    const auto h2d_wall_start=clock_type::now();event_owner h2d_start,h2d_end;cuda_check(cudaEventRecord(h2d_start.e,stream.s),"record H2D start");
    upload_fields(b,x,stream.s);
    if(weight_bytes)cuda_check(cudaMemcpyAsync(weights->p,x.weights.data(),host_size(weight_bytes,"weight upload"),cudaMemcpyHostToDevice,stream.s),"upload relation values");
    cuda_check(cudaEventRecord(h2d_end.e,stream.s),"record H2D end");cuda_check(cudaStreamSynchronize(stream.s),"finish input uploads");
    const double h2d_wall_ms=std::chrono::duration<double,std::milli>(clock_type::now()-h2d_wall_start).count();
    float h2d_elapsed=0;cuda_check(cudaEventElapsedTime(&h2d_elapsed,h2d_start.e,h2d_end.e),"measure H2D event interval");
    double prep_ms=0;auto r=prepare(d,x,std::move(weights),stream.s,&prep_ms);
    event_owner ev0,ev1,ev2,ev3;
    // One diagnostic iteration: four event boundaries, no intermediate syncs.
    cuda_check(cudaEventRecord(ev0.e,stream.s),"diagnostic start");
    multiply(*b.in[0],*b.in[0],*b.products[0],stream.s,"diagnostic left square");
    multiply(*b.in[0],*b.in[1],*b.products[1],stream.s,"diagnostic cross product");
    multiply(*b.in[1],*b.in[1],*b.products[2],stream.s,"diagnostic right square");
    cuda_check(cudaEventRecord(ev1.e,stream.s),"diagnostic products");
    apply(r,*b.in[0],*b.out[0],stream.s);apply(r,*b.in[1],*b.out[1],stream.s);
    apply(r,*b.products[0],*b.out[2],stream.s);apply(r,*b.products[1],*b.out[3],stream.s);apply(r,*b.products[2],*b.out[4],stream.s);
    cuda_check(cudaEventRecord(ev2.e,stream.s),"diagnostic relation end");
    multiply(*b.out[0],*b.out[0],*b.scratch,stream.s,"diagnostic mean square");
    affine(1,*b.out[2],-1,*b.scratch,*b.out[5],stream.s,"diagnostic center");
    multiply(*b.out[1],*b.out[1],*b.scratch,stream.s,"diagnostic mean square");
    affine(1,*b.out[4],-1,*b.scratch,*b.out[6],stream.s,"diagnostic center");
    multiply(*b.out[0],*b.out[1],*b.scratch,stream.s,"diagnostic mean cross");
    affine(1,*b.out[3],-1,*b.scratch,*b.out[7],stream.s,"diagnostic center");
    affine(2,*b.out[2],-1,*b.out[0],*b.out[8],stream.s,"diagnostic adjust");
    affine(2,*b.out[3],-1,*b.out[1],*b.out[9],stream.s,"diagnostic adjust");
    cuda_check(cudaEventRecord(ev3.e,stream.s),"diagnostic end");
    cuda_check(cudaStreamSynchronize(stream.s),"finish diagnostic iteration");
    float p0=0,p1=0,p2=0;
    cuda_check(cudaEventElapsedTime(&p0,ev0.e,ev1.e),"diagnostic product time");
    cuda_check(cudaEventElapsedTime(&p1,ev1.e,ev2.e),"diagnostic relation time");
    cuda_check(cudaEventElapsedTime(&p2,ev2.e,ev3.e),"diagnostic composition time");
    std::vector<double> events,walls;events.reserve(host_size(a.repeats,"repeat count"));walls.reserve(events.capacity());
    event_owner begin,end;
    for(u64 i=0;i<a.warmups+a.repeats;++i){
        const bool measure=i>=a.warmups;const auto wall=clock_type::now();
        if(measure)cuda_check(cudaEventRecord(begin.e,stream.s),"resident start");
        compose(r,b,stream.s);
        if(measure)cuda_check(cudaEventRecord(end.e,stream.s),"resident end");
        cuda_check(cudaStreamSynchronize(stream.s),"composition sync");
        if(measure){float ms=0;cuda_check(cudaEventElapsedTime(&ms,begin.e,end.e),"resident event time");events.push_back(ms);walls.push_back(std::chrono::duration<double,std::milli>(clock_type::now()-wall).count());}
    }
    ce::preparation_report report{};relation_check(ce::inspect(*r.handle,&report),"inspect prepared relation");
    t=clock_type::now();auto y=download(b,d,stream.s);const double d2h=std::chrono::duration<double,std::milli>(clock_type::now()-t).count();
    t=clock_type::now();u64 host_output=0;for(std::size_t i=0;i<y.size();++i)write_raw(a.output/(std::string(names[i])+".f32"),y[i],host_output);
    const double write=std::chrono::duration<double,std::milli>(clock_type::now()-t).count();
    const double total=std::chrono::duration<double,std::milli>(clock_type::now()-total_start).count();
    std::array<double,11> times{prep_ms,alloc_ms,read_ms,h2d_elapsed,h2d_wall_ms,d2h,write,total,p0,p1,p2};
    write_metrics(a.output/"metrics.json",d,a,prop,events,walls,times,b.bytes+r.bytes,peak_rss_bytes(),host_input,host_output,report);
    std::cout<<"NATIVE_NEIGHBORHOOD_MOMENTS_PASS resident_samples="<<events.size()<<" tracked_device_bytes="<<b.bytes+r.bytes<<"\n";
    return 0;
}
} // namespace
int main(int argc,char** argv){
    try{
        auto a=parse(argc,argv);if(a.help){help();return 0;}
        if(!a.self_test){
            demand(!a.input.empty()&&!a.output.empty(),"input and output directories are required");
            demand(a.sources&&a.destinations&&a.features,"dimensions must be positive");
            demand(a.features<=UINT32_MAX&&a.sources<=UINT32_MAX-128&&a.destinations<=UINT32_MAX-128&&a.edges<=UINT32_MAX-128,
                   "shape exceeds the prepared relation descriptor limits");
            (void)mul(a.sources,a.features,"source shape");(void)mul(a.destinations,a.features,"destination shape");
            demand(a.warmups<=100000&&a.repeats>0&&a.repeats<=100000,"invalid warmups/repeats");
        }
        int count=0;cuda_check(cudaGetDeviceCount(&count),"enumerate CUDA devices");demand(count>0,"no CUDA device available");
        if(a.self_test){self_test();return 0;}return run(a);
    }catch(const std::exception& e){std::cerr<<"ceNativeNeighborhoodMoments failed: "<<e.what()<<'\n';return 1;}
}
