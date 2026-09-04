#include "mock_backend_provider_v1.hh"

#include <array>
#include <cassert>

namespace cb = cellerator::compiler::backend::v1;

int main() {
    auto provider = make_mock_backend_provider_v1();
    assert(cb::validate_backend_provider_v1(provider) == cb::backend_status_v1::success);

    std::size_t target_count = 0;
    assert(provider.discover_targets(provider.context, nullptr, 0, &target_count, {}) ==
           cb::backend_status_v1::success);
    assert(target_count == 1);
    cb::backend_target_v1 target;
    assert(provider.discover_targets(provider.context, &target, 1, &target_count, {}) ==
           cb::backend_status_v1::success);

    const std::array<std::byte, 4> realization{
        std::byte{0x43}, std::byte{0x45}, std::byte{0x49}, std::byte{0x52}};
    std::array<std::byte, 4> object_bytes{};
    cb::backend_object_buffer_v1 object{object_bytes.data(), object_bytes.size(), 0};
    assert(cb::emit_backend_object_v1(
               provider, target, {realization.data(), realization.size(), 1, 0},
               &object) == cb::backend_status_v1::success);
    assert(object.size == realization.size());
    assert(object_bytes == realization);

    object.capacity = 2;
    assert(cb::emit_backend_object_v1(
               provider, target, {realization.data(), realization.size(), 1, 0},
               &object) == cb::backend_status_v1::insufficient_capacity);
    assert(object.size == realization.size());

    auto missing_emitter = provider;
    missing_emitter.emit_object = nullptr;
    assert(cb::validate_backend_provider_v1(missing_emitter) ==
           cb::backend_status_v1::invalid_argument);

    const auto& receipt = cb::get_backend_provider_abi_receipt_v1();
    assert(receipt.abi_version == 1);
    assert(receipt.host_only && receipt.ordinary_objects_required);
    assert(receipt.native_fragments_optional);
}
