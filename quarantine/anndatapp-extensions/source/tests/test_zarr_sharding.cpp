#include "zarr_engine.hpp"
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <array>
#include <string>
#include <sys/stat.h>
#include <vector>
#if defined(__linux__) && defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8
#define ANNDATAPP_TEST_FAULT_IO 1
#include <sys/syscall.h>
#include <unistd.h>
// Test-only syscall interposition covers both static and shared libraries. The
// selected worker thread receives EIO after the requested number of operations.
static thread_local int fail_read_after = -1, fail_write_after = -1;
static thread_local bool partial_write = false;
extern "C" ssize_t pread(int fd, void* data, size_t n, off_t offset) {
    if (!fail_read_after) { fail_read_after = -1; errno = EIO; return -1; }
    if (fail_read_after > 0) --fail_read_after;
    return static_cast<ssize_t>(::syscall(SYS_pread64, fd, data, n, offset));
}
extern "C" ssize_t pwrite(int fd, const void* data, size_t n, off_t offset) {
    if (!fail_write_after) { fail_write_after = -1; errno = EIO; return -1; }
    if (fail_write_after > 0) --fail_write_after;
    if (partial_write && n > 1) { n /= 2; partial_write = false; }
    return static_cast<ssize_t>(::syscall(SYS_pwrite64, fd, data, n, offset));
}
#endif

import anndatapp;

using namespace anndatapp_zarr;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
Bytes bytes(std::initializer_list<unsigned> values) {
    Bytes result; for (auto value : values) result.push_back(std::byte(value)); return result;
}
Bytes file_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open shard");
    auto size = std::filesystem::file_size(path);
    Bytes result(size);
    if (size && !file.read(reinterpret_cast<char*>(result.data()), std::streamsize(size)))
        throw std::runtime_error("cannot read shard");
    return result;
}
template<class F> void error(F&& f, Failure expected) {
    try { f(); } catch (const Error& e) { check(e.failure == expected, "wrong error category"); return; }
    throw std::runtime_error("expected error");
}
void no_temps(const std::filesystem::path& root) {
    for (auto& p : std::filesystem::directory_iterator(root))
        check(p.path().filename().string().find(".tmp-") == std::string::npos, "temporary file survived");
}
void json_file(const std::filesystem::path& path,const Json& value) {
    std::ofstream file(path);
    if(!file || !(file<<value.dump())) throw std::runtime_error("cannot write Zarr metadata fixture");
}
void raw_file(const std::filesystem::path& path,const Bytes& value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path,std::ios::binary);
    if(!file || (value.size() && !file.write(reinterpret_cast<const char*>(value.data()),std::streamsize(value.size()))))
        throw std::runtime_error("cannot write Zarr payload fixture");
}
void v2_half_array(const std::filesystem::path& root,const std::string& dtype,const Json& fill,
                   Shape shape={1},Shape chunks={1}) {
    std::filesystem::create_directories(root/"a");
    json_file(root/".zgroup",{{"zarr_format",2}});
    json_file(root/"a"/".zarray",{{"zarr_format",2},{"shape",shape},{"chunks",chunks},{"dtype",dtype},
        {"compressor",nullptr},{"filters",nullptr},{"fill_value",fill},{"order","C"},{"dimension_separator","."}});
    json_file(root/"a"/".zattrs",Json::object());
}
void v3_half_array(const std::filesystem::path& root,const Json& fill,const std::string& endian,
                   Shape shape={1},Shape chunks={1}) {
    std::filesystem::create_directories(root/"a");
    json_file(root/"zarr.json",{{"zarr_format",3},{"node_type","group"},{"attributes",Json::object()}});
    json_file(root/"a"/"zarr.json",{{"zarr_format",3},{"node_type","array"},{"shape",shape},{"data_type","float16"},
        {"chunk_grid",{{"name","regular"},{"configuration",{{"chunk_shape",chunks}}}}},
        {"chunk_key_encoding",{{"name","default"},{"configuration",{{"separator","/"}}}}},
        {"fill_value",fill},{"codecs",Json::array({Json{{"name","bytes"},{"configuration",{{"endian",endian}}}}})},
        {"attributes",Json::object()}});
}
std::uint16_t read_half(const std::filesystem::path& root) {
    Store store(root,false);
    std::uint16_t result=0;
    store.read_numeric("a",{0},{1},&result);
    return result;
}
void half_storage_tests(const std::filesystem::path& root) {
    check(type_size(Type::float16)==2,"float16 storage width");

    // v2 parses both byte orders and swaps a two-byte payload into native-endian bits.
    auto v2_big=root/"fp16-v2-big";
    v2_half_array(v2_big,">f2",0,{4},{4});
    raw_file(v2_big/"a"/"0",bytes({0x3c,0x00,0x80,0x00,0x7e,0x55,0xfc,0x00}));
    {
        Store store(v2_big,false);
        check(store.inspect("a").array.type==Type::float16,"v2 >f2 dtype");
        std::uint16_t actual[4]{};
        store.read_numeric("a",{0},{4},actual);
        check(actual[0]==0x3c00 && actual[1]==0x8000 && actual[2]==0x7e55 && actual[3]==0xfc00,
              "v2 big-endian float16 payload");
    }
    auto v2_little=root/"fp16-v2-little";
    v2_half_array(v2_little,"<f2",0,{4},{4});
    raw_file(v2_little/"a"/"0",bytes({0x00,0x3c,0x00,0x80,0x55,0x7e,0x00,0xfc}));
    {
        Store store(v2_little,false);
        check(store.inspect("a").array.type==Type::float16,"v2 <f2 dtype");
        std::uint16_t actual[4]{};
        store.read_numeric("a",{0},{4},actual);
        check(actual[0]==0x3c00 && actual[1]==0x8000 && actual[2]==0x7e55 && actual[3]==0xfc00,
              "v2 little-endian float16 payload");
    }

    // This double lies just above the half-way point, but rounds to that point
    // if first narrowed to float. The storage fill must round the original double.
    const double just_above_tie=1.0+std::ldexp(1.0,-11)+std::ldexp(1.0,-25);
    auto tie=root/"fp16-v2-tie";
    v2_half_array(tie,"<f2",1.0+std::ldexp(1.0,-11));
    check(read_half(tie)==0x3c00,"binary16 halfway tie rounds to even");
    auto above_tie=root/"fp16-v2-above-tie";
    v2_half_array(above_tie,"<f2",just_above_tie);
    check(read_half(above_tie)==0x3c01,"binary16 fill rounds directly from double");
    for(const auto& [fill,expected] : std::array<std::pair<const char*,std::uint16_t>,3>{{
            {"NaN",0x7e00},{"Infinity",0x7c00},{"-Infinity",0xfc00}}}) {
        auto path=root/(std::string("fp16-v2-")+fill);
        v2_half_array(path,"<f2",fill);
        check(read_half(path)==expected,"v2 float16 special fill value");
    }
    auto invalid_v2_hex=root/"fp16-v2-invalid-hex";
    v2_half_array(invalid_v2_hex,"<f2","0x7e55");
    error([&]{ (void)read_half(invalid_v2_hex); },Failure::encoding);

    // v3 accepts canonical special values and exact 16-bit pattern strings.
    for(const auto& [fill,expected] : std::array<std::pair<const char*,std::uint16_t>,5>{{
            {"NaN",0x7e00},{"Infinity",0x7c00},{"-Infinity",0xfc00},
            {"0x7E55",0x7e55},{"0x8000",0x8000}}}) {
        auto path=root/(std::string("fp16-v3-")+fill);
        v3_half_array(path,fill,"little");
        check(read_half(path)==expected,"v3 float16 special or exact-bit fill value");
    }
    auto v3_big=root/"fp16-v3-big";
    v3_half_array(v3_big,"0x7E55","big",{4},{4});
    raw_file(v3_big/"a"/"c"/"0",bytes({0x3c,0x00,0x80,0x00,0x7e,0x55,0xfc,0x00}));
    {
        Store store(v3_big,false);
        check(store.inspect("a").array.type==Type::float16,"v3 float16 dtype");
        std::uint16_t actual[4]{};
        store.read_numeric("a",{0},{4},actual);
        check(actual[0]==0x3c00 && actual[1]==0x8000 && actual[2]==0x7e55 && actual[3]==0xfc00,
              "v3 big-endian float16 payload");
    }

    // Exercise both adapter mappings: DType::float16 <-> Zarr Type::float16.
    for(const auto& [label,format] : std::array<std::pair<const char*,anndatapp::ZarrFormat>,2>{{
            {"api-v2",anndatapp::ZarrFormat::v2},{"api-v3",anndatapp::ZarrFormat::v3}}}) {
        auto path=root/label;
        auto data=anndatapp::AnnData::empty(1,2);
        data.set_x(anndatapp::Array::zeros<anndatapp::Float16>({1,2}));
        auto* values=data.edit_x().dense<anndatapp::Float16>().data();
        values[0]=anndatapp::Float16::from_bits(0x3c00);
        values[1]=anndatapp::Float16::from_bits(0xfe55);
        anndatapp::ZarrWriteOptions options;
        options.format=format;
        options.compression=anndatapp::ZarrCompression::none;
        anndatapp::write_zarr(path.string(),data,options);
        auto disk=anndatapp::open_zarr(path.string());
        auto loaded=disk.x().read();
        const auto* actual=loaded.view().dense<anndatapp::Float16>().data();
        check(actual[0].bits==0x3c00 && actual[1].bits==0xfe55,"public v2/v3 float16 round trip");
    }
}

void run(const std::filesystem::path& path, const std::string& index_location, bool transpose, int crc_count, const char* endian) {
    ShardSpec spec;
    spec.shard_shape = {2,3}; spec.chunk_shape = {1,1}; spec.index_location = index_location;
    spec.scratch_bytes = 128*1024;
    spec.index_codecs = Json::array();
    if (transpose) spec.index_codecs.push_back({{"name","transpose"},{"configuration",{{"order",{1,0,2}}}}});
    spec.index_codecs.push_back({{"name","bytes"},{"configuration",{{"endian",endian}}}});
    for (int i = 0; i < crc_count; ++i) spec.index_codecs.push_back({{"name","crc32c"}});
    const auto index_bytes = shard_index_workspace(spec);
    const auto encoded_size = 6*16 + 4*crc_count;
    const bool simple = !transpose && crc_count <= 1;
    check(index_bytes == encoded_size*(simple ? 1 : 4) && index_bytes < spec.scratch_bytes, "index reservation");
    Bytes copy(7), encoded;
    ShardWorkspace workspace;
    const auto a = bytes({1,2,3}), b = bytes({7,8,9,10}), c = bytes({0,255}), d = bytes({19});
    {
        ShardWriter writer(path,spec,workspace);
        check(!writer.read({0,0},encoded,4096), "missing shard must have missing chunks");
        writer.replace({0,0},a); writer.replace({0,2},b); writer.replace({1,1},c);
        error([&]{ writer.replace({0,0},a); }, Failure::argument);
        writer.commit(copy);
        error([&]{ writer.commit(copy); }, Failure::argument);
    }
    const auto* saved_index = workspace.index.data();
    const auto allocations = workspace.allocations;
    check(read_shard_chunk(path,spec,{0,0}) == a, "first chunk");
    check(read_shard_chunk(path,spec,{0,2}) == b, "second chunk");
    check(read_shard_chunk(path,spec,{1,1}) == c, "third chunk");
    check(read_shard_chunk(path,spec,{1,2}).empty(), "missing chunk");
    check(::chmod(path.c_str(),0640) == 0, "chmod fixture");
    auto original = file_bytes(path);
    {
        ShardWriter writer(path,spec,workspace);
        check(writer.read({0,0},encoded,4096) && encoded == a, "old chunk read");
        auto* p = encoded.data();
        check(writer.read({1,1},encoded,4096) && encoded == c && encoded.data() == p, "old payload buffer reused");
        writer.replace({0,0},d);
        writer.replace({1,1},{});
        writer.replace({1,2},a);
        check(writer.read({0,2},encoded,4096) && encoded == b, "unchanged old payload readable after replacement");
        check(file_bytes(path) == original, "original changed before commit");
        writer.commit(copy);
    }
    if (simple) check(workspace.index.data() == saved_index && workspace.allocations == allocations, "index workspace reused");
    struct stat st{}; check(::stat(path.c_str(),&st)==0 && (st.st_mode&0777)==0640, "permissions retained");
    check(read_shard_chunk(path,spec,{0,0}) == d, "replacement chunk");
    check(read_shard_chunk(path,spec,{0,2}) == b, "unchanged encoded chunk");
    check(read_shard_chunk(path,spec,{1,1}).empty(), "deleted chunk");
    check(read_shard_chunk(path,spec,{1,2}) == a, "new chunk");
    original = file_bytes(path);
    {
        ShardWriter writer(path,spec,workspace);
        error([&]{ writer.read({0,2},encoded,1); }, Failure::argument);
        writer.replace({0,0},b);
    }
    check(file_bytes(path) == original, "uncommitted or failed writer changed shard");
    no_temps(path.parent_path());
#if defined(ANNDATAPP_TEST_FAULT_IO)
    // Index read fails before creating a temporary file.
    fail_read_after = 0;
    error([&]{ ShardWriter writer(path,spec,workspace); }, Failure::io);
    check(file_bytes(path) == original, "failed index read changed shard");
    no_temps(path.parent_path());
    // Old payload read fails after the index was read and temporary file made.
    {
        ShardWriter writer(path,spec,workspace);
        fail_read_after = 0;
        error([&]{ writer.read({0,2},encoded,4096); }, Failure::io);
    }
    check(file_bytes(path) == original, "failed payload read changed shard");
    no_temps(path.parent_path());
    // Failure partway through a replacement leaves only private temp bytes.
    {
        ShardWriter writer(path,spec,workspace);
        writer.replace({0,0},b);
        partial_write = true; fail_write_after = 1;
        error([&]{ writer.replace({1,2},a); }, Failure::io);
    }
    check(file_bytes(path) == original, "partial payload write changed shard");
    no_temps(path.parent_path());
    // Fail the final index write after copying both untouched old payloads.
    {
        ShardWriter writer(path,spec,workspace);
        writer.replace({0,0},b);
        fail_write_after = 2;
        error([&]{ writer.commit(copy); }, Failure::io);
    }
    check(file_bytes(path) == original, "failed index write changed shard");
    no_temps(path.parent_path());
#endif
    auto invalid = spec;
    const std::size_t bytes_codec = transpose ? 1 : 0;
    invalid.index_codecs[bytes_codec]["configuration"]["endian"] = "invalid";
    error([&]{ shard_index_workspace(invalid); }, Failure::encoding);
    auto bad = spec; bad.scratch_bytes = index_bytes-1;
    error([&]{ shard_index_workspace(bad); }, Failure::argument);
    error([&]{ ShardWriter writer(path,bad,workspace); }, Failure::argument);
    check(file_bytes(path) == original, "insufficient budget changed shard");
    no_temps(path.parent_path());
    // Corrupt the independently stored index checksum, preserving the original.
    auto damaged = path.parent_path()/(path.filename().string()+"-bad");
    auto corrupt = original;
    const auto index_start = index_location == "start" ? 0 : corrupt.size()-encoded_size;
    if (crc_count) corrupt[index_start+encoded_size-1] ^= std::byte{1};
    else for (std::size_t i = 0; i < 8; ++i) corrupt[index_start+i] = std::byte{0xff};
    { std::ofstream out(damaged,std::ios::binary); out.write(reinterpret_cast<const char*>(corrupt.data()),std::streamsize(corrupt.size())); }
    error([&]{ ShardWriter writer(damaged,spec,workspace); }, Failure::encoding);
    check(file_bytes(damaged) == corrupt, "corrupt shard changed on rejected index");
    no_temps(path.parent_path());
}
int main(int argc,char** argv) {
    try {
        auto root = std::filesystem::path(argc > 1 ? argv[1] : "build/zarr-sharding-test");
        std::filesystem::create_directories(root);
        auto pattern = (root/"trial-XXXXXX").string();
        if (!::mkdtemp(pattern.data())) throw std::runtime_error("cannot create fixture directory");
        root = pattern;
        for (const auto& location : {"start","end"}) for (bool transpose : {false,true})
            for (int crc : {0,1,3}) for (const auto& endian : {"little","big"})
                run(root/(std::string(location)+(transpose?"-transpose":"-plain")+"-crc"+std::to_string(crc)+endian),location,transpose,crc,endian);
        half_storage_tests(root);
        std::cout << "shard writer and float16 tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
