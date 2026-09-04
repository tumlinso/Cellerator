#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>

namespace cellerator::compiler::api::v1 {

class immutable_registry_v1 {
  public:
    using entries_type = std::map<std::string, std::string>;
    explicit immutable_registry_v1(entries_type entries);
    [[nodiscard]] const std::string* find(const std::string& name) const noexcept;

  private:
    const entries_type entries_;
};

class isolated_context_v1 {
  public:
    explicit isolated_context_v1(std::shared_ptr<const immutable_registry_v1> registry);
    void set(std::string key, std::string value);
    [[nodiscard]] std::string get(const std::string& key) const;
    [[nodiscard]] const immutable_registry_v1& registry() const noexcept;

  private:
    std::shared_ptr<const immutable_registry_v1> registry_;
    mutable std::shared_mutex mutex_;
    std::map<std::string, std::string> state_;
};

class context_builder_v1 {
  public:
    explicit context_builder_v1(std::shared_ptr<const immutable_registry_v1> registry);
    context_builder_v1(const context_builder_v1&) = delete;
    context_builder_v1& operator=(const context_builder_v1&) = delete;
    void set(std::string key, std::string value);
    [[nodiscard]] std::unique_ptr<isolated_context_v1> finish();

  private:
    void require_owner() const;
    std::thread::id owner_;
    std::shared_ptr<const immutable_registry_v1> registry_;
    std::map<std::string, std::string> initial_state_;
    bool finished_ = false;
};

struct backend_process_isolation_v1 {
    std::uint64_t process_id = 0;
    bool shares_mutable_compiler_state = false;
};

}  // namespace cellerator::compiler::api::v1
