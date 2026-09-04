#include <Cellerator/compiler/backend/nvptx/implement_fatbinary_object_embedding_for_direct_ptx_v1.hh>

#include <algorithm>
#include <cctype>
#include <dlfcn.h>

namespace Cellerator::compiler::backend::nvptx {
namespace {

std::string binary_symbol(std::string value) {
    std::replace_if(value.begin(), value.end(), [](const char character) {
        return !std::isalnum(static_cast<unsigned char>(character));
    }, '_');
    return "_binary_" + value;
}

template <class Function>
Function symbol(void* library, const char* name) {
    return reinterpret_cast<Function>(dlsym(library, name));
}

struct driver_api {
    using initialize_fn = int (*)(unsigned);
    using device_get_fn = int (*)(int*, int);
    using context_create_fn = int (*)(void**, unsigned, int);
    using context_destroy_fn = int (*)(void*);
    using module_load_fn = int (*)(void**, const void*);
    using module_unload_fn = int (*)(void*);
    using module_function_fn = int (*)(void**, void*, const char*);
    using launch_fn = int (*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                              unsigned, void*, void**, void**);
    using synchronize_fn = int (*)();

    void* library = nullptr;
    initialize_fn initialize = nullptr;
    device_get_fn device_get = nullptr;
    context_create_fn context_create = nullptr;
    context_destroy_fn context_destroy = nullptr;
    module_load_fn module_load = nullptr;
    module_unload_fn module_unload = nullptr;
    module_function_fn module_function = nullptr;
    launch_fn launch = nullptr;
    synchronize_fn synchronize = nullptr;

    driver_api() {
        library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) return;
        initialize = symbol<initialize_fn>(library, "cuInit");
        device_get = symbol<device_get_fn>(library, "cuDeviceGet");
        context_create = symbol<context_create_fn>(library, "cuCtxCreate_v2");
        context_destroy = symbol<context_destroy_fn>(library, "cuCtxDestroy_v2");
        module_load = symbol<module_load_fn>(library, "cuModuleLoadData");
        module_unload = symbol<module_unload_fn>(library, "cuModuleUnload");
        module_function = symbol<module_function_fn>(library, "cuModuleGetFunction");
        launch = symbol<launch_fn>(library, "cuLaunchKernel");
        synchronize = symbol<synchronize_fn>(library, "cuCtxSynchronize");
    }
    ~driver_api() {
        if (library != nullptr) dlclose(library);
    }
    explicit operator bool() const noexcept {
        return library != nullptr && initialize != nullptr && device_get != nullptr &&
            context_create != nullptr && context_destroy != nullptr && module_load != nullptr &&
            module_unload != nullptr && module_function != nullptr && launch != nullptr &&
            synchronize != nullptr;
    }
};

}  // namespace

cuda_object_embedding_plan_v1 make_cuda_object_embedding_plan_v1(
    const std::string& objcopy_executable,
    const std::string& image_basename,
    const std::string& object_path,
    const embedded_cuda_image_kind_v1 image_kind) {
    cuda_object_embedding_plan_v1 plan;
    plan.image_kind = image_kind;
    if (objcopy_executable.empty() || image_basename.empty() || object_path.empty()) return plan;
    plan.executable = objcopy_executable;
    plan.object_path = object_path;
    plan.section_name = image_kind == embedded_cuda_image_kind_v1::ptx
        ? ".nv_cellerator_ptx" : image_kind == embedded_cuda_image_kind_v1::cubin
            ? ".nv_cellerator_cubin" : ".nv_cellerator_fatbin";
    plan.arguments = {"--input-target=binary", "--output-target=elf64-x86-64",
                      "--binary-architecture=i386:x86-64",
                      "--rename-section", ".data=" + plan.section_name + ",alloc,load,readonly,data,contents",
                      image_basename, object_path};
    const auto base_symbol = binary_symbol(image_basename);
    plan.start_symbol = base_symbol + "_start";
    plan.end_symbol = base_symbol + "_end";
    return plan;
}

embedded_cuda_launch_result_v1 launch_embedded_cuda_image_v1(
    const embedded_cuda_image_v1& image,
    const embedded_cuda_launch_v1& launch_configuration) {
    embedded_cuda_launch_result_v1 result;
    if (image.begin == nullptr || image.end == nullptr || image.end <= image.begin ||
        image.kernel_symbol.empty()) {
        result.diagnostic = "embedded image range or kernel symbol is invalid";
        return result;
    }
    result.image_bytes = static_cast<std::size_t>(image.end - image.begin);
    if (launch_configuration.grid_x == 0u || launch_configuration.grid_y == 0u ||
        launch_configuration.grid_z == 0u || launch_configuration.block_x == 0u ||
        launch_configuration.block_y == 0u || launch_configuration.block_z == 0u) {
        result.status = embedded_cuda_launch_status_v1::invalid_launch;
        result.diagnostic = "launch dimensions must be nonzero";
        return result;
    }
    driver_api api;
    if (!api) {
        result.status = embedded_cuda_launch_status_v1::driver_unavailable;
        result.diagnostic = "CUDA driver ABI is unavailable";
        return result;
    }
    int device = 0;
    void* context = nullptr;
    void* module = nullptr;
    void* function = nullptr;
    auto fail = [&](const int code, const char* message) {
        if (module != nullptr) api.module_unload(module);
        if (context != nullptr) api.context_destroy(context);
        result.status = embedded_cuda_launch_status_v1::driver_error;
        result.driver_code = code;
        result.diagnostic = message;
        return result;
    };
    int code = api.initialize(0u);
    if (code != 0) return fail(code, "cuInit failed");
    code = api.device_get(&device, 0);
    if (code != 0) return fail(code, "cuDeviceGet failed");
    code = api.context_create(&context, 0u, device);
    if (code != 0) return fail(code, "cuCtxCreate failed");
    code = api.module_load(&module, image.begin);
    if (code != 0) return fail(code, "cuModuleLoadData failed");
    code = api.module_function(&function, module, image.kernel_symbol.c_str());
    if (code != 0) return fail(code, "cuModuleGetFunction failed");
    auto arguments = launch_configuration.arguments;
    code = api.launch(function, launch_configuration.grid_x, launch_configuration.grid_y,
                      launch_configuration.grid_z, launch_configuration.block_x,
                      launch_configuration.block_y, launch_configuration.block_z,
                      launch_configuration.dynamic_shared_bytes, launch_configuration.stream,
                      arguments.empty() ? nullptr : arguments.data(), nullptr);
    if (code != 0) return fail(code, "cuLaunchKernel failed");
    code = api.synchronize();
    if (code != 0) return fail(code, "cuCtxSynchronize failed");
    api.module_unload(module);
    api.context_destroy(context);
    result.status = embedded_cuda_launch_status_v1::success;
    result.driver_code = 0;
    result.diagnostic = "embedded image loaded and kernel completed";
    return result;
}

}  // namespace Cellerator::compiler::backend::nvptx
