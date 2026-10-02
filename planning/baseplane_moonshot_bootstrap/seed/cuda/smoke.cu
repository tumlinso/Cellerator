// Future bounded smoke harness. Compile does not run it. Execution requires
// explicit '--run DEVICE' and a resource assignment in the owning environment.
#include "volta.cuh"
#include <bp_moon/reference.hpp>
#include <iostream>
#include <string>
#include <vector>
static void ck(cudaError_t r){if(r!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(r));}
template<class T>struct Buffer{
    T* p=nullptr;std::size_t n;
    explicit Buffer(std::size_t size):n(size){if(n)ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    ~Buffer(){if(p)cudaFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void put(const std::vector<T>& x,cudaStream_t s){if(x.size()!=n)throw std::runtime_error("size mismatch");if(n)ck(cudaMemcpyAsync(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice,s));}
    std::vector<T> get(cudaStream_t s){std::vector<T>x(n);if(n)ck(cudaMemcpyAsync(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return x;}
};
struct Stream{cudaStream_t s{};Stream(){ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){cudaStreamDestroy(s);}};
static void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
int main(int argc,char** argv){
    if(argc!=3||std::string(argv[1])!="--run"){
        std::cout<<"Compiled seed harness. Execute only with an assigned device: --run DEVICE\n";return 0;
    }
    try{
        std::size_t used=0;int device=std::stoi(argv[2],&used);require(used==std::string(argv[2]).size()&&device>=0,"device argument");
        ck(cudaSetDevice(device));cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,device));
        require(prop.major==7&&prop.minor==0,"This receipt is specifically for a Volta sm_70 device");
        Stream s;
        auto l=bp_moon::summarize_sequence("ACGTAC"),r=bp_moon::summarize_sequence("GGAT"),expected=bp_moon::compose(l,r);
        Buffer<unsigned>a(32),b(32),c(32);a.put(std::vector<unsigned>(l.to.begin(),l.to.end()),s.s);b.put(std::vector<unsigned>(r.to.begin(),r.to.end()),s.s);
        bp_moon_cuda::dfa_compose<<<1,32,0,s.s>>>(a.p,b.p,c.p,1);ck(cudaGetLastError());auto states=c.get(s.s);
        require(std::equal(states.begin(),states.end(),expected.to.begin()),"DFA mismatch");
        std::vector<unsigned char> flags(65);for(unsigned i=0;i<65;++i)flags[i]=(i%3)==0;
        Buffer<unsigned char> f(65);Buffer<unsigned>ids(5);Buffer<bp_moon_cuda::EmitCounts>counts(1);
        f.put(flags,s.s);counts.put({{0,0,0}},s.s);
        bp_moon_cuda::emit_selected<<<1,96,0,s.s>>>(f.p,65,ids.p,5,counts.p);ck(cudaGetLastError());auto n=counts.get(s.s)[0];
        require(n.produced==22&&n.stored==5&&n.dropped==17,"emit accounting");
        auto gotids=ids.get(s.s);for(auto i:gotids)require(i<65&&flags[i],"emit member");
        std::sort(gotids.begin(),gotids.end());require(std::adjacent_find(gotids.begin(),gotids.end())==gotids.end(),"duplicate output");
        std::vector<__half>ha(256),hb(256);for(unsigned i=0;i<16;++i)for(unsigned j=0;j<16;++j){ha[i*16+j]=__float2half(float(i+j));hb[i*16+j]=__float2half(i==j?1.f:0.f);}
        Buffer<__half>ta(256),tb(256);Buffer<float>tc(256);ta.put(ha,s.s);tb.put(hb,s.s);
        bp_moon_cuda::wmma16<<<1,32,0,s.s>>>(ta.p,tb.p,tc.p,1);ck(cudaGetLastError());auto result=tc.get(s.s);
        for(unsigned i=0;i<16;++i)for(unsigned j=0;j<16;++j)require(result[i*16+j]==float(i+j),"WMMA identity mismatch");
        std::cout<<"{\"status\":\"cuda_smoke_pass\",\"device\":"<<device<<",\"compute_capability\":\"7.0\",\"checks\":[\"dfa\",\"emit_overflow\",\"wmma_identity\"],\"benchmark\":false}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
