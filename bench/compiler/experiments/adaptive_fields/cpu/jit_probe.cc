#include <Cellerator/compiler/backend/compile_generated_c_into_ordinary_objects_v1.hh>
#include <dlfcn.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <spawn.h>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <vector>

extern char** environ;
using namespace cellerator::compiler::backend::v1;
namespace fs=std::filesystem;
using kernel_t = int (*)(const std::uint64_t*,const std::uint64_t*,const float*,
 const float*,float*,std::uint64_t,const std::uint64_t*,std::uint64_t) noexcept;

static std::uint64_t hash(std::string_view s) {
    std::uint64_t h=1469598103934665603ULL;for(unsigned char c:s){h^=c;h*=1099511628211ULL;}return h;
}
static std::string read(const fs::path&p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
static std::string compiler_version(const std::string&compiler){
    const std::string command=compiler+" --version";
    FILE*pipe=popen(command.c_str(),"r");if(!pipe)return "unavailable";
    char buffer[256];std::string out;if(fgets(buffer,sizeof(buffer),pipe))out=buffer;
    pclose(pipe);return out;
}
static bool spawn_wait(std::vector<std::string>&args){
    std::vector<char*>argv;for(auto&arg:args)argv.push_back(arg.data());argv.push_back(nullptr);
    pid_t pid{};if(posix_spawnp(&pid,args[0].c_str(),nullptr,nullptr,argv.data(),environ)!=0)return false;
    int st{};return waitpid(pid,&st,0)>=0&&WIFEXITED(st)&&WEXITSTATUS(st)==0;
}

int main(int argc,char**argv){
    if(argc<2||argc>5)return 2;
    const fs::path root=argv[1];fs::create_directories(root);
    const std::string compiler=argc>2?argv[2]:"g++";
    const std::string contract=argc>3?argv[3]:"fp32-rn-v1";
    const std::string topology=argc>4?argv[4]:"csr4x3-seg2-v1";
    if(contract!="fp32-rn-v1"&&contract!="fp32-fast-v1"){std::cerr<<"unsupported numerical contract\n";return 2;}
    const std::string version=compiler_version(compiler);
    const std::vector<std::string> flags=contract=="fp32-fast-v1"
        ?std::vector<std::string>{"-fPIC","-ffast-math"}:std::vector<std::string>{"-fPIC"};
    const std::string flags_id=contract=="fp32-fast-v1"?"-fPIC -ffast-math":"-fPIC";
    const std::string source=std::string(R"cpp(
#include <cmath>
#include <cstdint>
namespace { constexpr const char* topology_contract = ")cpp")+topology+R"cpp("; }
extern "C" int cellerator_region(const std::uint64_t* off,const std::uint64_t* col,
 const float* values,const float* x,float* out,std::uint64_t rows,
 const std::uint64_t* seg,std::uint64_t segments) noexcept {
 if(rows!=4||segments!=2||topology_contract[0]=='\0')return 1;
 float temp[4]{};
 for(std::uint64_t r=0;r<rows;++r){float sum=0;
  for(auto e=off[r];e<off[r+1];++e)sum+=values[e]*x[col[e]];temp[r]=sum;}
 for(std::uint64_t s=0;s<segments;++s){auto b=seg[s],e=seg[s+1];
  if(b==e)continue;float m=temp[b];for(auto r=b+1;r<e;++r)if(temp[r]>m)m=temp[r];
  float d=0;for(auto r=b;r<e;++r)d+=std::exp(temp[r]-m);
  for(auto r=b;r<e;++r)out[r]=std::exp(temp[r]-m)/d;}
 return 0;
}
)cpp";
    const std::string contract_record="storage=f32;compute=f32;accumulate=f32;output=f32;"+contract;
    const auto key=hash(source+"|"+compiler+"|"+version+"|"+flags_id+"|"+contract_record+"|"+topology);
    const auto stem=std::to_string(key);const auto src=root/(stem+".cc");
    const auto obj=root/(stem+".o");const auto shared=root/(stem+".so");
    const auto dep=root/(stem+".d");const auto manifest=root/(stem+".manifest");
    {std::ofstream f(src);f<<source;}
    const auto source_hash=hash(source);
    auto valid_cached=[&](){const std::string m=read(manifest);
      return fs::exists(shared)&&m.find("input_key="+stem+"\n")!=std::string::npos&&
        m.find("source_hash="+std::to_string(source_hash)+"\n")!=std::string::npos&&
        m.find("depfile_hash="+std::to_string(hash(read(dep)))+"\n")!=std::string::npos&&
        m.find("artifact_hash="+std::to_string(hash(read(shared)))+"\n")!=std::string::npos;};
    unsigned compile_count=0,link_count=0;double compile_ms=0,link_ms=0;bool warm=valid_cached();
    auto prepare=[&](){if(valid_cached())return true;
      auto compile_start=std::chrono::steady_clock::now();
      compile_generated_cpp_request_v1 req{compiler,src.string(),obj.string(),dep.string(),root.string(),flags,{}, {}};
      compile_generated_cpp_receipt_v1 receipt{};
      if(compile_generated_cpp_object_v1(req,&receipt)!=compile_object_status_v1::success)return false;
      compile_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-compile_start).count();
      ++compile_count;
      std::vector<std::string> args{compiler,"-shared",obj.string(),"-o",shared.string()};
      auto link_start=std::chrono::steady_clock::now();if(!spawn_wait(args))return false;
      link_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-link_start).count();++link_count;
      std::ofstream m(manifest);m<<"input_key="<<stem<<"\nsource_hash="<<source_hash
        <<"\ndepfile_hash="<<hash(read(dep))<<"\nartifact_hash="<<hash(read(shared))
        <<"\ntoolchain="<<version<<"compiler="<<compiler<<"\nflags="<<flags_id<<"\nnumerical_contract="
        <<contract_record<<"\ntopology="<<topology<<"\n";
      m.close();
      return valid_cached();};
    const auto t0=std::chrono::steady_clock::now();
    if(!prepare()){std::cerr<<"prepare failed\n";return 3;}
    // Same-process lookup must hit the validated artifact manifest.
    if(!prepare()||compile_count!=(warm?0u:1u)||link_count!=(warm?0u:1u))return 4;
    const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    auto load_start=std::chrono::steady_clock::now();
    void*lib=dlopen(shared.c_str(),RTLD_NOW|RTLD_LOCAL);if(!lib){std::cerr<<dlerror()<<"\n";return 5;}
    const auto load_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-load_start).count();
    auto run=reinterpret_cast<kernel_t>(dlsym(lib,"cellerator_region"));if(!run)return 6;
    const std::uint64_t off[]{0,2,3,5,6},off_v2[]{0,1,3,5,6};
    const std::uint64_t col[]{0,2,1,0,2,1},col_v2[]{1,2,0,1,2,1},seg[]{0,2,4};
    const bool topology_v2=topology.ends_with("-v2");
    const auto*active_off=topology_v2?off_v2:off;
    const auto*active_col=topology_v2?col_v2:col;
    const float x[]{2,3,4};float vals[]{.5F,.25F,1,.75F,.5F,2},out[4]{};
    if(run(active_off,active_col,vals,x,out,4,seg,2)!=0)return 7;
    const float first=out[0];vals[0]=2.0F;out[0]=out[1]=out[2]=out[3]=0;
    if(run(active_off,active_col,vals,x,out,4,seg,2)!=0||out[0]==first)return 8;
    dlclose(lib);
    std::cout<<"key="<<stem<<" warm_hit="<<(warm?1:0)<<" compile_count="<<compile_count
      <<" link_count="<<link_count<<" prepare_ms="<<elapsed<<" compile_ms="<<compile_ms
      <<" link_ms="<<link_ms<<" load_ms="<<load_ms<<" artifact_bytes="<<fs::file_size(shared)
      <<" source_hash="<<source_hash<<" artifact_hash="<<hash(read(shared))
      <<" value_rebind=pass toolchain=["<<version.substr(0,version.find('\n'))
      <<"] contract="<<contract<<" topology="<<topology<<" artifact="<<shared<<"\n";
}
