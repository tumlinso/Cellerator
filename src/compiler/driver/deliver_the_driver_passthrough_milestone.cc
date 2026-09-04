#include <Cellerator/compiler/driver/deliver_the_driver_passthrough_milestone_v1.hh>
#include <cstdlib>
#include <stdexcept>
#include <sys/wait.h>
namespace cellerator::compiler::driver {
namespace { std::string shell_quote(const std::string& value) { std::string out{"'"}; for (char c : value) out += c == '\'' ? "'\\''" : std::string(1, c); return out + "'"; } }
driver_passthrough_result_v1 run_driver_passthrough_v1(const std::string& compiler, const std::vector<std::string>& arguments) { if (compiler.empty()) throw std::invalid_argument("downstream compiler is required"); std::string command = shell_quote(compiler); for (const auto& argument : arguments) command += " " + shell_quote(argument); const int status = std::system(command.c_str()); if (status == -1) return {-1, false}; return {WIFEXITED(status) ? WEXITSTATUS(status) : 128, false}; }
}  // namespace cellerator::compiler::driver
