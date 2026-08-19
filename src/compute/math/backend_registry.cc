#include <Cellerator/compute/math/planner.hh>

namespace cellerator::compute::math {

backend_registration_status SpMMBackendRegistry::add(
    SpMMBackend *backend) noexcept {
    if (backend == nullptr || backend->identity() == 0u
        || backend->name() == nullptr || backend->name()[0] == '\0') {
        return {backend_registration_code::invalid_backend,
            "backend registration requires a named nonzero identity"};
    }
    for (std::size_t index = 0u; index < size_; ++index) {
        if (backends_[index]->identity() == backend->identity()) {
            return {backend_registration_code::duplicate_identity,
                "backend identity is already registered"};
        }
    }
    if (size_ == max_spmm_backend_count) {
        return {backend_registration_code::capacity_exceeded,
            "backend registry capacity exceeded"};
    }
    backends_[size_++] = backend;
    return {};
}

backend_registration_status SpMMBackendRegistry::remove(u64 identity) noexcept {
    for (std::size_t index = 0u; index < size_; ++index) {
        if (backends_[index]->identity() != identity) continue;
        for (std::size_t next = index + 1u; next < size_; ++next)
            backends_[next - 1u] = backends_[next];
        backends_[--size_] = nullptr;
        return {};
    }
    return {backend_registration_code::not_found,
        "backend identity is not registered"};
}

void SpMMBackendRegistry::clear() noexcept {
    for (std::size_t index = 0u; index < size_; ++index) backends_[index] = nullptr;
    size_ = 0u;
}

std::size_t SpMMBackendRegistry::size() const noexcept {
    return size_;
}

SpMMBackend *SpMMBackendRegistry::at(std::size_t index) const noexcept {
    return index < size_ ? backends_[index] : nullptr;
}

SpMMBackendRegistry &global_spmm_backend_registry() noexcept {
    static SpMMBackendRegistry registry;
    return registry;
}

} // namespace cellerator::compute::math
