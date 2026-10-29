#include <Cellerator/execution/conditional/parameter_store.hh>

namespace cellerator::execution::conditional {

void ParameterBuffer::validate() const {
    if (bytes && (!owner || !data))
        throw std::invalid_argument("nonempty parameter buffer needs data and shared ownership");
    if (bytes && (dtype.empty() || device.empty()))
        throw std::invalid_argument("nonempty parameter buffer needs explicit dtype and device");
}
void ParameterBuffers::validate() const {
    parameters.validate();
    master_parameters.validate();
    gradients.validate();
    optimizer_state.validate();
}
const ParameterRevision& ParameterLease::revision() const {
    if (!value_) throw std::logic_error("parameter lease was released");
    return *value_;
}
void ParameterStore::register_initial(ExpertKey key, ParameterBuffers buffers) {
    buffers.validate();
    auto value = std::make_shared<const ParameterRevision>(ParameterRevision{key, std::move(buffers)});
    std::lock_guard<std::mutex> guard(mutex_);
    if (current_.count(key.expert_id) || used_.count(key))
        throw std::invalid_argument("expert already registered; publish a new revision");
    used_.insert(key);
    revisions_.emplace(key, std::move(value));
    current_.emplace(key.expert_id, key);
}
ExpertKey ParameterStore::current(std::uint64_t expert_id) const {
    std::lock_guard<std::mutex> guard(mutex_);
    return current_.at(expert_id);
}
ParameterLease ParameterStore::acquire(ExpertKey key) const {
    std::lock_guard<std::mutex> guard(mutex_);
    return ParameterLease(revisions_.at(key));
}
std::vector<ParameterLease> ParameterStore::snapshot(const std::vector<ExpertKey>& keys) const {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<ParameterLease> result;
    result.reserve(keys.size());
    for (const auto key : keys) result.push_back(ParameterLease(revisions_.at(key)));
    return result;
}
void ParameterStore::publish(ExpertKey expected, ExpertKey next, ParameterBuffers buffers) {
    buffers.validate();
    if (next.expert_id != expected.expert_id || next.revision <= expected.revision)
        throw std::invalid_argument("new revision must advance the same logical expert");
    auto value = std::make_shared<const ParameterRevision>(ParameterRevision{next, std::move(buffers)});
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = current_.find(expected.expert_id);
    if (it == current_.end() || it->second != expected)
        throw StaleParameterRevision("parameter update base is no longer current");
    if (used_.count(next)) throw std::invalid_argument("parameter revision identity was already used");
    used_.insert(next);
    revisions_.emplace(next, std::move(value));
    it->second = next;
}
ExpertKey ParameterStore::update(ExpertKey expected, std::uint64_t new_revision,
                                ParameterBuffer gradients, const Update& backend) {
    if (!backend) throw std::invalid_argument("parameter update backend is required");
    gradients.validate();
    if (new_revision <= expected.revision)
        throw std::invalid_argument("parameter revision must increase");
    auto lease = acquire(expected);
    if (current(expected.expert_id) != expected)
        throw StaleParameterRevision("parameter update base is no longer current");
    auto result = backend(lease.revision(), gradients);
    ExpertKey next{expected.expert_id, new_revision};
    publish(expected, next, std::move(result));
    return next;
}
bool ParameterStore::retire(ExpertKey key) {
    std::shared_ptr<const ParameterRevision> retired;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        auto current = current_.find(key.expert_id);
        if (current != current_.end() && current->second == key)
            throw std::logic_error("cannot retire the current parameter revision");
        auto it = revisions_.find(key);
        if (it == revisions_.end()) return false;
        retired = std::move(it->second);
        revisions_.erase(it);
    } // Backend deleters run outside the store lock.
    return true;
}
std::size_t ParameterStore::leases(ExpertKey key) const {
    std::lock_guard<std::mutex> guard(mutex_);
    return revisions_.at(key).use_count() - 1;
}

} // namespace cellerator::execution::conditional
