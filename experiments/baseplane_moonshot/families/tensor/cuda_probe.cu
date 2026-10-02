#include <ce_moon/tensor.cuh>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
using namespace ce_moon::tensor;
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void equal(const Matrix16& actual,const Matrix16& expected){for(unsigned i=0;i<256;++i)if(!std::isfinite(actual[i])||!std::isfinite(expected[i])||std::abs(actual[i]-expected[i])>1e-5f)throw std::runtime_error("GPU oracle mismatch or nonfinite value");}
void comparator_self_check(){
 Matrix16 zero{},nonfinite{};equal(zero,zero);
 auto rejects=[&](const Matrix16& actual,const Matrix16& expected){bool rejected=false;try{equal(actual,expected);}catch(const std::runtime_error&){rejected=true;}if(!rejected)throw std::runtime_error("oracle comparator accepted invalid value");};
 nonfinite[0]=std::numeric_limits<float>::quiet_NaN();rejects(nonfinite,zero);rejects(zero,nonfinite);
 nonfinite[0]=std::numeric_limits<float>::infinity();rejects(nonfinite,zero);rejects(zero,nonfinite);
 nonfinite[0]=1.f;rejects(nonfinite,zero);
}
int main(int argc,char** argv){
 comparator_self_check();
 if(argc>1){if(argc==2&&std::string(argv[1])=="--self-check"){std::cout<<"Tensor comparator finite/mismatch self-check passed; no GPU accessed\n";return 0;}throw std::runtime_error("usage: ce_moon_tensor_cuda [--self-check]");}
 int device=0;checked(cudaGetDevice(&device));cudaDeviceProp prop{};checked(cudaGetDeviceProperties(&prop,device));if(prop.major<7)throw std::runtime_error("sm70 required");
 float *a,*b,*result;__half *pa,*pb;unsigned char *exists,*mask;unsigned long long* ids;cuda::DevicePair* pairs;unsigned *required,*overflow;
 checked(cudaMalloc(&a,1024));checked(cudaMalloc(&b,1024));checked(cudaMalloc(&result,1024));checked(cudaMalloc(&pa,512));checked(cudaMalloc(&pb,512));checked(cudaMalloc(&exists,256));checked(cudaMalloc(&mask,256));checked(cudaMalloc(&ids,128));checked(cudaMalloc(&pairs,sizeof(cuda::DevicePair)));checked(cudaMalloc(&required,4));checked(cudaMalloc(&overflow,4));cudaStream_t stream;checked(cudaStreamCreate(&stream));
 auto load=[&](const Matrix16& x,const Matrix16& w){checked(cudaMemcpyAsync(a,x.data(),1024,cudaMemcpyHostToDevice,stream));checked(cudaMemcpyAsync(b,w.data(),1024,cudaMemcpyHostToDevice,stream));};
 auto read=[&](){Matrix16 out;checked(cudaMemcpyAsync(out.data(),result,1024,cudaMemcpyDeviceToHost,stream));checked(cudaStreamSynchronize(stream));return out;};
 Matrix16 x{},w{};for(unsigned i=0;i<16;++i){x[i*16]=float(i+1);x[i*16+1]=2;}w[0]=2;w[16]=3;w[1]=-1;
 load(x,w);checked(cuda::object_features(a,b,pa,pb,result,{3,2,2},stream));equal(read(),multiply(x,w,{3,2,2}));
 x={};w={};x[0]=1;x[1]=2;x[16]=3;x[17]=-1;w[0]=2;w[16]=-1;w[17]=4;
 load(x,w);checked(cuda::relation_scores(a,b,pa,pb,result,2,2,stream));equal(read(),relation_scores(x,w,2,2));
 std::array<unsigned long long,16> source_ids{};source_ids[0]=7;source_ids[1]=9001;PairMask m{};m[1]=m[16]=1;checked(cudaMemcpyAsync(ids,source_ids.data(),128,cudaMemcpyHostToDevice,stream));checked(cudaMemcpyAsync(mask,m.data(),256,cudaMemcpyHostToDevice,stream));checked(cuda::compact_relations(result,ids,mask,2,0,pairs,1,required,overflow,stream));unsigned count=0,over=0;cuda::DevicePair pair{};checked(cudaMemcpyAsync(&count,required,4,cudaMemcpyDeviceToHost,stream));checked(cudaMemcpyAsync(&over,overflow,4,cudaMemcpyDeviceToHost,stream));checked(cudaMemcpyAsync(&pair,pairs,sizeof(pair),cudaMemcpyDeviceToHost,stream));checked(cudaStreamSynchronize(stream));if(count!=2||over!=1||pair.from!=7||pair.to!=9001)throw std::runtime_error("pair provenance/capacity mismatch");
 x={};w={};x[1]=x[2]=1;w[19]=w[35]=w[36]=1;load(x,w);checked(cuda::finite_relation_counts(a,b,pa,pb,result,exists,stream));equal(read(),multiply(x,w,{}));std::array<unsigned char,256> binary{};checked(cudaMemcpyAsync(binary.data(),exists,256,cudaMemcpyDeviceToHost,stream));checked(cudaStreamSynchronize(stream));if(binary[3]!=1||binary[4]!=1||binary[5]!=0)throw std::runtime_error("threshold mismatch");
 x={};w={};x[0]=-1;x[16]=1;w[0]=1;load(x,w);checked(cuda::possible_states(a,b,pa,pb,result,{2,1,1},stream));equal(read(),possible_states(x,w,{2,1,1}));
 checked(cudaStreamDestroy(stream));for(void* p:{static_cast<void*>(a),static_cast<void*>(b),static_cast<void*>(result),static_cast<void*>(pa),static_cast<void*>(pb),static_cast<void*>(exists),static_cast<void*>(mask),static_cast<void*>(ids),static_cast<void*>(pairs),static_cast<void*>(required),static_cast<void*>(overflow)})checked(cudaFree(p));
 std::cout<<"E21-E24 GPU compared against host oracles; directed provenance and overflow passed\n";
}
