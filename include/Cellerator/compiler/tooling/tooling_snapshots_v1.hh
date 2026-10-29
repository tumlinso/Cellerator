#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>
namespace Cellerator::compiler::tooling {
struct tooling_symbol_v1{std::string name;std::uint64_t begin=0,end=0;};
struct tooling_snapshot_data_v1{std::uint64_t revision=0;std::string source;std::vector<tooling_symbol_v1> symbols;std::vector<std::string> diagnostics;};
class tooling_snapshot_v1{public: explicit tooling_snapshot_v1(tooling_snapshot_data_v1 data);std::uint64_t revision()const;std::optional<tooling_symbol_v1> symbol_at(std::uint64_t offset)const;const std::vector<std::string>&diagnostics()const;private:std::shared_ptr<const tooling_snapshot_data_v1> data_;};
class tooling_cancellation_v1{public:tooling_cancellation_v1();void cancel()const;bool cancelled()const;private:std::shared_ptr<std::atomic_bool> state_;};
using background_compile_hook_v1=std::function<void(std::string,tooling_cancellation_v1)>;
void request_background_compile_v1(const background_compile_hook_v1&,std::string,tooling_cancellation_v1);
} // namespace Cellerator::compiler::tooling
