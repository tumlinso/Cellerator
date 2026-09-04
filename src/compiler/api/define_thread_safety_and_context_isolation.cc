#include <Cellerator/compiler/api/define_thread_safety_and_context_isolation_v1.hh>

#include <mutex>
#include <stdexcept>
#include <utility>

namespace cellerator::compiler::api::v1 {

immutable_registry_v1::immutable_registry_v1(entries_type entries) : entries_(std::move(entries)) {}
const std::string* immutable_registry_v1::find(const std::string& name) const noexcept {
    const auto found = entries_.find(name);
    return found == entries_.end() ? nullptr : &found->second;
}
isolated_context_v1::isolated_context_v1(std::shared_ptr<const immutable_registry_v1> registry)
    : registry_(std::move(registry)) {
    if (!registry_) throw std::invalid_argument("registry is required");
}
void isolated_context_v1::set(std::string key, std::string value) {
    std::unique_lock lock(mutex_);
    state_.insert_or_assign(std::move(key), std::move(value));
}
std::string isolated_context_v1::get(const std::string& key) const {
    std::shared_lock lock(mutex_);
    const auto found = state_.find(key);
    return found == state_.end() ? std::string{} : found->second;
}
const immutable_registry_v1& isolated_context_v1::registry() const noexcept { return *registry_; }
context_builder_v1::context_builder_v1(std::shared_ptr<const immutable_registry_v1> registry)
    : owner_(std::this_thread::get_id()), registry_(std::move(registry)) {
    if (!registry_) throw std::invalid_argument("registry is required");
}
void context_builder_v1::require_owner() const {
    if (owner_ != std::this_thread::get_id()) throw std::logic_error("context builder is thread-confined");
    if (finished_) throw std::logic_error("context builder is already finished");
}
void context_builder_v1::set(std::string key, std::string value) {
    require_owner();
    initial_state_.insert_or_assign(std::move(key), std::move(value));
}
std::unique_ptr<isolated_context_v1> context_builder_v1::finish() {
    require_owner();
    auto context = std::make_unique<isolated_context_v1>(registry_);
    for (const auto& [key, value] : initial_state_) context->set(key, value);
    finished_ = true;
    return context;
}

}  // namespace cellerator::compiler::api::v1
