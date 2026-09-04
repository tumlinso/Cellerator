#include <Cellerator/compiler/driver/track_downstream_c_language_and_abi_mode_v1.hh>
namespace cellerator::compiler::driver {
downstream_mode_v1 track_downstream_language_and_abi_v1(const std::vector<std::string>& args) {
    downstream_mode_v1 out;
    for (std::size_t i = 0; i != args.size(); ++i) {
        const auto& arg = args[i];
        if (arg.rfind("-std=", 0) == 0) { out.language_standard = arg.substr(5); out.compiler_flags.push_back(arg); }
        else if ((arg == "--target" || arg == "-target") && i + 1 < args.size()) { out.target = args[++i]; out.compiler_flags.insert(out.compiler_flags.end(), {arg, out.target}); }
        else if (arg.rfind("--target=", 0) == 0) { out.target = arg.substr(9); out.compiler_flags.push_back(arg); }
        else if (arg.rfind("-D_GLIBCXX_USE_CXX11_ABI=", 0) == 0) out.compiler_flags.push_back(arg);
        else if (arg.rfind("-I", 0) == 0 || arg.rfind("-D", 0) == 0 || arg.rfind("-U", 0) == 0) out.preprocessor_flags.push_back(arg);
        else if (arg.rfind("-Wl,", 0) == 0 || arg == "-shared" || arg.rfind("-l", 0) == 0 || arg.rfind("-L", 0) == 0) out.linker_flags.push_back(arg);
        else if (arg == "-fexceptions" || arg == "-fno-exceptions" || arg == "-frtti" || arg == "-fno-rtti" || arg.rfind("-fsanitize=", 0) == 0 || arg.rfind("-fvisibility=", 0) == 0 || arg.rfind("-fabi-version=", 0) == 0) out.compiler_flags.push_back(arg);
        else out.unclassified.push_back(arg);
    }
    return out;
}
}  // namespace cellerator::compiler::driver
