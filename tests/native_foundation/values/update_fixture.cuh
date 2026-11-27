#pragma once
#include "fixture.cuh"
#include "trace_cusparse_fixture.cuh"
#include <chrono>
#include <cstring>
#include <fstream>

extern "C" void start_memory() __attribute__((weak));
extern "C" void begin_steps() __attribute__((weak));
extern "C" void report_memory(const char*, int) __attribute__((weak));

namespace cellerator::execution::planning::architecture::providers::nvidia::sm70::edge_value_gradient {
cudaError_t enqueue_relation_value_update_f32(float*, void*, const float*, std::uint32_t,
    bool, float, cudaStream_t) noexcept;
}
namespace trace_update = cellerator::execution::planning::architecture::providers::nvidia::sm70::edge_value_gradient;
void update_test(bool f32,bool derive=false){
    storage owner;cuda_ok(cudaStreamCreate(&owner.a));cuda_ok(cudaStreamCreate(&owner.b));
    rel::operation_descriptor f;f.topology={{40,1},{1},axis(1),axis(2),{41,1},3};f.dense_width=16;if(f32)f.arithmetic.relation_storage=ex::numeric_type::f32;
    auto t=f;t.direction=rel::orientation::transpose;
    std::array<std::uint32_t,3> offsets{0,2,3},sources{0,1,0};ok(rel::prepare_relation_pair(f,t,{offsets.data(),3,sources.data(),3},{0,0,derive},owner.a,&owner.first));ok(rel::create_relation_instance(*owner.first,owner.b,&owner.second));
    float *values{},*cotangent{},*gradient1{},*gradient2{},*delta{};
    cuda_ok(cudaMalloc(&values,12));cuda_ok(cudaMalloc(&owner.weights,6));cuda_ok(cudaMalloc(&owner.input,128));cuda_ok(cudaMalloc(&owner.output,128));
    cuda_ok(cudaMalloc(&cotangent,128));cuda_ok(cudaMalloc(&gradient1,12));cuda_ok(cudaMalloc(&gradient2,12));cuda_ok(cudaMalloc(&delta,12));
    try{
        std::array<float,3> first{1,2,3},second{4,5,6};
        auto publish=[&](rel::prepared_relation_pair& pair,cudaStream_t stream,const std::array<float,3>& w){
            if(f32){cuda_ok(cudaMemcpy(values,w.data(),12,cudaMemcpyHostToDevice));ok(rel::publish_f32_values(pair,{values,3,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0},stream));}
            else{std::array<__half,3> half{__float2half(w[0]),__float2half(w[1]),__float2half(w[2])};cuda_ok(cudaMemcpy(owner.weights,half.data(),6,cudaMemcpyHostToDevice));ok(rel::publish_values(pair,{owner.weights,3,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0},stream));}
            cuda_ok(cudaStreamSynchronize(stream));
        };publish(*owner.first,owner.a,first);publish(*owner.second,owner.b,second);
        std::vector<float> x(32),dy(32);for(int c=0;c<16;++c){x[c]=1;x[16+c]=2;dy[c]=3;dy[16+c]=4;}
        cuda_ok(cudaMemcpy(owner.input,x.data(),128,cudaMemcpyHostToDevice));cuda_ok(cudaMemcpy(cotangent,dy.data(),128,cudaMemcpyHostToDevice));
        auto check=[&](rel::prepared_relation_pair& pair,cudaStream_t stream,const std::array<float,3>& w,std::uint64_t gen){
            ok(rel::enqueue(pair,f,{owner.input,32,f.topology.source,0},{owner.output,32,f.topology.destination,0},{gen},stream));cuda_ok(cudaStreamSynchronize(stream));
            std::vector<float> got(32);cuda_ok(cudaMemcpy(got.data(),owner.output,128,cudaMemcpyDeviceToHost));
            for(int c=0;c<16;++c)require(got[c]==w[0]+2*w[1] && got[16+c]==w[2],"independent instance update result");
        };
        rel::relation_calculus_descriptor calculus;calculus.forward=f;calculus.transpose=t;
        ok(rel::prepare_relation_gradient(*owner.first,calculus,{},owner.a));ok(rel::prepare_relation_gradient(*owner.second,calculus,{},owner.b));
        rel::edge_layout_view layout{};ok(rel::inspect_edge_layout(*owner.first,&layout));
        auto plane=[&](float* p){return rel::edge_plane_view{p,3,f.topology.identity,f.topology.epoch,layout.order,0};};
        auto response=[&](rel::prepared_relation_pair& pair,cudaStream_t stream,float* output,std::uint64_t gen){
            rel::gradient_stamp stamp;ok(rel::enqueue_edge_gradient(pair,calculus,{owner.input,32,f.topology.source,0},{cotangent,32,f.topology.destination,0},{1,1},{2,1},{gen},plane(output),&stamp,stream));cuda_ok(cudaStreamSynchronize(stream));
            std::array<float,3> got{};cuda_ok(cudaMemcpy(got.data(),output,12,cudaMemcpyDeviceToHost));const std::array<float,3> expected{48,96,64};
            for(int e=0;e<3;++e)require(got[layout.logical_to_physical[e]]==expected[e],"actual response equals independent derivative");return stamp;
        };
        auto old_first=response(*owner.first,owner.a,gradient1,1);auto stable_second=response(*owner.second,owner.b,gradient2,1);
        rel::value_update_request request;request.kind=rel::value_update_kind::gradient_step;request.operand=plane(gradient2);request.expected={1};request.next={2};request.alpha=1.f/16;request.gradient=stable_second;
        require(!rel::enqueue_value_update(*owner.first,request,owner.a),"foreign instance response rejected");
        std::array<float,3> packed{};const float d=f32?std::ldexp(1.f,-18):.25f;packed[layout.logical_to_physical[0]]=d;
        cuda_ok(cudaMemcpy(delta,packed.data(),12,cudaMemcpyHostToDevice));request.kind=rel::value_update_kind::delta_add;request.operand=plane(delta);request.gradient={};
        ok(rel::enqueue_value_update(*owner.first,request,owner.a));first[0]+=d;check(*owner.first,owner.a,first,2);check(*owner.second,owner.b,second,1);
        request.kind=rel::value_update_kind::gradient_step;request.operand=plane(gradient1);request.expected={2};request.next={3};request.gradient=old_first;
        require(!rel::enqueue_value_update(*owner.first,request,owner.a),"parameter update invalidates old response");
        require(!rel::enqueue(*owner.first,f,{owner.input,32,f.topology.source,0},{owner.output,32,f.topology.destination,0},{1},owner.a),"old value generation is not a historical snapshot");
        auto fresh=response(*owner.first,owner.a,gradient1,2);request.gradient=fresh;ok(rel::enqueue_value_update(*owner.first,request,owner.a));
        const std::array<float,3> grad{48,96,64};for(int e=0;e<3;++e)first[e]=std::fma(-1.f/16,grad[e],first[e]);check(*owner.first,owner.a,first,3);
        request.operand=plane(gradient2);request.expected={1};request.next={2};request.gradient=stable_second;
        ok(rel::enqueue_value_update(*owner.second,request,owner.b));for(int e=0;e<3;++e)second[e]=std::fma(-1.f/16,grad[e],second[e]);check(*owner.second,owner.b,second,2);
        rel::preparation_report a{},b{};ok(rel::inspect(*owner.first,&a));ok(rel::inspect(*owner.second,&b));require(a.structural_preparation_id==b.structural_preparation_id,"updates share structure only");
        if(f32)require(a.derived_f16_generation.value==(derive?3:0) && b.derived_f16_generation.value==(derive?2:0),"derived projections follow their own authoritative updates");
        ok(rel::close_relation_pair(&owner.first));ok(rel::close_relation_pair(&owner.second));
    }catch(...){cudaFree(delta);cudaFree(gradient2);cudaFree(gradient1);cudaFree(cotangent);cudaFree(values);throw;}
    cudaFree(delta);cudaFree(gradient2);cudaFree(gradient1);cudaFree(cotangent);cudaFree(values);
}

// Text fixture: row count, column count, edge count, row offsets, column indices.
void fp32_trace_update_test(const char* path, bool benchmark = false, bool vendor = false,
    bool mapped = false, bool prepared_vendor = false) {
    constexpr std::uint32_t width = 16;
    constexpr float learning_rate = 1.f / 64;
    std::ifstream trace(path);
    std::uint32_t rows = 0, cols = 0, edges = 0;
    require(bool(trace >> rows >> cols >> edges) && rows && cols && edges, "read CSR trace dimensions");
    std::vector<std::uint32_t> offsets(std::size_t(rows) + 1), indices(edges);
    for (auto& offset : offsets) require(bool(trace >> offset), "read CSR trace offsets");
    for (auto& index : indices) require(bool(trace >> index), "read CSR trace column indices");

    rel::operation_descriptor forward{};
    forward.topology = {{40, 2}, {1}, axis(1), axis(2), {41, 2}, edges};
    forward.topology.source.extent = cols;
    forward.topology.destination.extent = rows;
    forward.dense_width = width;
    forward.arithmetic.relation_storage = ex::numeric_type::f32;
    auto transpose = forward;
    transpose.direction = rel::orientation::transpose;

    const std::size_t nx = std::size_t(cols) * width, ny = std::size_t(rows) * width;
    std::vector<float> weights(edges), x(nx), dy(ny), y(ny), dx(nx), observed(edges);
    std::vector<double> expected_y(ny), expected_dx(nx);
    for (std::size_t e = 0; e < edges; ++e) weights[e] = float(int((e * 13 + 1) % 23) - 11) / 32;
    for (std::size_t i = 0; i < nx; ++i) x[i] = float(int((i * 17 + 3) % 31) - 15) / 16;
    for (std::size_t i = 0; i < ny; ++i) dy[i] = float(int((i * 11 + 7) % 29) - 14) / 32;

    storage owner;
    trace_cusparse library;
    cuda_ok(cudaStreamCreateWithFlags(&owner.a, cudaStreamNonBlocking));
    std::vector<std::uint32_t> logical_to_physical;
    if (mapped) {
        // Obtain the actual native map outside the comparison. The mapped
        // candidate assumes the owning relation already supplies its ordering.
        ok(rel::prepare_relation_pair(forward, transpose,
            {offsets.data(), offsets.size(), indices.data(), indices.size()}, {}, owner.a, &owner.first));
        rel::edge_layout_view layout{};
        ok(rel::inspect_edge_layout(*owner.first, &layout));
        logical_to_physical.assign(layout.logical_to_physical, layout.logical_to_physical + edges);
        ok(rel::close_relation_pair(&owner.first));
    }
    const bool account = benchmark && start_memory && begin_steps && report_memory;
    if (account) start_memory();
    using clock = std::chrono::steady_clock;
    const auto prepare_start = clock::now();
    cuda_ok(cudaMalloc(&owner.input, ((mapped ? 3 : 2) * std::size_t(edges) + 2 * nx + 2 * ny) * sizeof(float)));
    float* dw = owner.input;
    float* d_x = dw + edges;
    float* d_dy = d_x + nx;
    float* d_y = d_dy + ny;
    float* d_dx = d_y + ny;
    float* d_gradient = d_dx + nx;
    float* d_physical = mapped ? d_gradient + edges : nullptr;
    std::uint32_t* d_map = nullptr;
    if (mapped) {
        // Reuse the fixture's second allocation slot to own the device map.
        cuda_ok(cudaMalloc(&owner.output, edges * sizeof(*d_map)));
        d_map = reinterpret_cast<std::uint32_t*>(owner.output);
        std::vector<float> physical(edges);
        for (std::size_t e = 0; e < edges; ++e) physical[logical_to_physical[e]] = weights[e];
        cuda_ok(cudaMemcpy(d_map, logical_to_physical.data(), edges * sizeof(*d_map), cudaMemcpyHostToDevice));
        cuda_ok(cudaMemcpy(d_physical, physical.data(), edges * sizeof(float), cudaMemcpyHostToDevice));
    }
    cuda_ok(cudaMemcpyAsync(dw, weights.data(), edges * sizeof(float), cudaMemcpyHostToDevice, owner.a));
    cuda_ok(cudaMemcpyAsync(d_x, x.data(), nx * sizeof(float), cudaMemcpyHostToDevice, owner.a));
    cuda_ok(cudaMemcpyAsync(d_dy, dy.data(), ny * sizeof(float), cudaMemcpyHostToDevice, owner.a));
    const auto& t = forward.topology;
    const rel::device_state_view input{d_x, nx, t.source, 0}, cotangent{d_dy, ny, t.destination, 0};
    const rel::device_result_view output{d_y, ny, t.destination, 0}, input_gradient{d_dx, nx, t.source, 0};
    rel::relation_calculus_descriptor calculus{forward, transpose};
    rel::edge_layout_view layout{};
    std::size_t reordered = 0;
    if (vendor) {
        // The trace's logical order is CSR order. Both SpMM directions borrow
        // dw, and SGD updates it in place. SDDMM writes its own CSR value plane.
        library.prepare(offsets, indices, cols, width, dw, d_dx, d_y, d_gradient, d_x, d_dy, owner.a);
        for (std::size_t e = 0; mapped && e < edges; ++e) reordered += logical_to_physical[e] != e;
    } else {
        ok(rel::prepare_relation_pair(forward, transpose,
            {offsets.data(), offsets.size(), indices.data(), indices.size()}, {}, owner.a, &owner.first));
        const rel::fp32_fixed_buffers buffers{input, cotangent, output, input_gradient};
        // The native timing arm also tests automatic fallback when the CSR
        // samples cannot fit the budget. Its native full-FP32 gradient uses no scratch.
        const rel::gradient_preparation_options options{rel::gradient_route::automatic,
            benchmark && !prepared_vendor ? 1u : 0u, benchmark ? &buffers : nullptr};
        ok(rel::prepare_relation_gradient(*owner.first, calculus, options, owner.a));
        ok(rel::inspect_edge_layout(*owner.first, &layout));
        for (std::size_t e = 0; e < edges; ++e) reordered += layout.logical_to_physical[e] != e;
        ok(rel::publish_f32_values(*owner.first, {dw, edges, t.identity, t.epoch, t.logical_edge_order, {1}, 0}, owner.a));
    }
    cuda_ok(cudaStreamSynchronize(owner.a));
    const auto prepare_ms = std::chrono::duration<double, std::milli>(clock::now() - prepare_start).count();
    const rel::edge_plane_view gradient{d_gradient, edges, t.identity, t.epoch, layout.order, 0};
    rel::gradient_stamp stamp{};
    auto apply = [&](ex::value_generation generation) {
        if (vendor) {
            sparse_ok(library.forward.enqueue());
            sparse_ok(library.transpose.enqueue());
        } else {
            ok(rel::enqueue(*owner.first, forward, input, output, generation, owner.a));
            ok(rel::enqueue(*owner.first, transpose, cotangent, input_gradient, generation, owner.a));
        }
    };
    auto differentiate = [&](ex::value_generation generation) {
        if (vendor) library.edge_gradient();
        else ok(rel::enqueue_edge_gradient(*owner.first, calculus, input, cotangent,
            {1, 1}, {2, 1}, generation, gradient, &stamp, owner.a));
    };
    auto update = [&](ex::value_generation generation) {
        if (mapped) {
            cuda_ok(trace_sparse::update_and_refresh_csr(d_physical, dw, d_gradient,
                d_map, edges, true, learning_rate, true, owner.a));
        } else if (vendor) cuda_ok(trace_update::enqueue_relation_value_update_f32(
            dw, nullptr, d_gradient, edges, true, learning_rate, owner.a));
        else ok(rel::enqueue_value_update(*owner.first, {rel::value_update_kind::gradient_step,
            gradient, generation, {generation.value + 1}, learning_rate, stamp}, owner.a));
    };

    auto agrees = [](float actual, double expected) {
        return std::abs(actual - expected) <= 2e-5 + 1e-5 * std::abs(expected);
    };
    auto check_apply = [&](ex::value_generation generation) {
        apply(generation);
        cuda_ok(cudaMemcpyAsync(y.data(), d_y, ny * sizeof(float), cudaMemcpyDeviceToHost, owner.a));
        cuda_ok(cudaMemcpyAsync(dx.data(), d_dx, nx * sizeof(float), cudaMemcpyDeviceToHost, owner.a));
        std::fill(expected_y.begin(), expected_y.end(), 0);
        std::fill(expected_dx.begin(), expected_dx.end(), 0);
        for (std::size_t row = 0; row < rows; ++row) {
            for (auto e = offsets[row]; e < offsets[row + 1]; ++e) {
                const std::size_t col = indices[e];
                for (std::size_t k = 0; k < width; ++k) {
                    expected_y[row * width + k] += double(weights[e]) * x[col * width + k];
                    expected_dx[col * width + k] += double(weights[e]) * dy[row * width + k];
                }
            }
        }
        cuda_ok(cudaStreamSynchronize(owner.a));
        for (std::size_t i = 0; i < ny; ++i) require(agrees(y[i], expected_y[i]), "CSR forward oracle mismatch");
        for (std::size_t i = 0; i < nx; ++i) require(agrees(dx[i], expected_dx[i]), "CSR transpose oracle mismatch");
    };

    check_apply({1});
    if (prepared_vendor) {
        auto alternate = output;
        alternate.data = d_dx;
        require(rel::enqueue(*owner.first, forward, input, alternate, {1}, owner.a).code
            == rel::status_code::invalid_argument, "cuSPARSE must reject changed dense addresses");
    }
    differentiate({1});
    cuda_ok(cudaMemcpyAsync(observed.data(), d_gradient, edges * sizeof(float), cudaMemcpyDeviceToHost, owner.a));
    cuda_ok(cudaStreamSynchronize(owner.a));
    for (std::size_t row = 0; row < rows; ++row) {
        for (auto e = offsets[row]; e < offsets[row + 1]; ++e) {
            const std::size_t col = indices[e];
            double derivative = 0;
            for (std::size_t k = 0; k < width; ++k) derivative += double(dy[row * width + k]) * x[col * width + k];
            require(agrees(observed[vendor ? e : layout.logical_to_physical[e]], derivative), "CSR edge gradient oracle mismatch");
            weights[e] = float(double(weights[e]) - learning_rate * derivative);
        }
    }

    update({1});
    check_apply({2});
    if (mapped) {
        cuda_ok(cudaMemcpy(observed.data(), d_physical, edges * sizeof(float), cudaMemcpyDeviceToHost));
        for (std::size_t e = 0; e < edges; ++e)
            require(agrees(observed[logical_to_physical[e]], weights[e]), "mapped SGD oracle mismatch");
    }
    if (!vendor) {
        rel::relation_update_report report{};
        ok(rel::inspect_updates(*owner.first, &report));
        const char* expected_backend = prepared_vendor ? "cusparse-csr-f32" : "retained-csr-f32";
        require(report.relation.forward_candidate && report.relation.transpose_candidate &&
            !std::strcmp(report.relation.forward_candidate, expected_backend) &&
            !std::strcmp(report.relation.transpose_candidate, expected_backend), "expected FP32 CSR route in both directions");
        require(report.relation.topology_preparations == 1 && report.gradient_preparations == 1 &&
            report.gradient_launches == 1 && report.physical_updates == 1 &&
            report.relation.latest_enqueued_generation.value == 2, "expected one prepared topology and one published update");
        if (prepared_vendor) {
            require(report.cusparse_gradient_launches == 1 && report.csr_value_refreshes == 2,
                "expected SDDMM and two CSR refreshes");
        }
        if (benchmark) {
            for (auto& weight : weights) weight *= 0.5f;
            cuda_ok(cudaMemcpyAsync(dw, weights.data(), edges * sizeof(float), cudaMemcpyHostToDevice, owner.a));
            ok(rel::publish_f32_values(*owner.first, {dw, edges, t.identity, t.epoch, t.logical_edge_order, {3}, 0}, owner.a));
            check_apply({3});
        }
    }
    std::cout << "FP32 CSR width " << width << ", " << rows << " rows, " << cols << " columns, " << edges
        << " edges: forward, transpose, edge gradient and updated forward/transpose passed CPU oracles\n";
    if (benchmark) {
        const char* backend = prepared_vendor ? "prepared-cusparse" : mapped ? "cusparse-mapped" : vendor ? "cusparse" : "native";
        std::cout << "prepare " << backend << ' ' << prepare_ms << " ms, reordered_edges=" << reordered;
        if (vendor || prepared_vendor) {
            int major = 0, minor = 0, patch = 0;
            sparse_ok(cusparseGetProperty(MAJOR_VERSION, &major));
            sparse_ok(cusparseGetProperty(MINOR_VERSION, &minor));
            sparse_ok(cusparseGetProperty(PATCH_LEVEL, &patch));
            std::cout << ", cusparse_version=" << major << '.' << minor << '.' << patch;
        }
        std::cout << '\n';
        ex::value_generation generation{vendor ? 2u : 3u};
        auto step = [&] { apply(generation); differentiate(generation); update(generation); ++generation.value; };
        for (int i = 0; i < 12; ++i) step();
        cuda_ok(cudaStreamSynchronize(owner.a));
        constexpr int steps = 32;
        if (account) {
            begin_steps();
            for (int i = 0; i < steps; ++i) step();
            cuda_ok(cudaStreamSynchronize(owner.a));
            report_memory(backend, steps);
        } else {
            for (int trial = 0; trial < 5; ++trial) {
                const auto start = clock::now();
                for (int i = 0; i < steps; ++i) step();
                cuda_ok(cudaStreamSynchronize(owner.a));
                std::cout << "step " << backend << ' '
                    << std::chrono::duration<double, std::milli>(clock::now() - start).count() / steps << " ms\n";
            }
        }
        if (prepared_vendor) {
            rel::relation_update_report report{};
            ok(rel::inspect_updates(*owner.first, &report));
            require(report.cusparse_gradient_launches == generation.value - 2
                && report.csr_value_refreshes == generation.value, "expected SDDMM and refresh on every timed update");
        }
    }
    if (owner.first) ok(rel::close_relation_pair(&owner.first));
}
