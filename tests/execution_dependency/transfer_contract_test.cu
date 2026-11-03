#include <Cellerator/execution/dependency/transfer.cuh>

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dependency = cellerator::execution::dependency;
namespace mc = cellerator::memory::cuda;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Exception, class Function>
void rejects(Function&& function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void check(cudaError_t status, const char* operation) {
    mc::check(status, operation);
}

bool can_peer(int source_device, int destination_device) {
    int available = 0;
    mc::DeviceGuard guard(destination_device);
    check(cudaDeviceCanAccessPeer(&available, destination_device, source_device),
          "cudaDeviceCanAccessPeer test");
    return available != 0;
}

void copy_and_check(dependency::PreparedTransfer& transfer,
                    const mc::DeviceBuffer& source,
                    mc::DeviceBuffer& destination,
                    const mc::Stream& source_stream,
                    const mc::Stream& destination_stream,
                    const std::vector<unsigned char>& expected,
                    mc::PinnedBuffer& host_output) {
    transfer.enqueue(source.view(), destination.view(), expected.size(),
                     source_stream, destination_stream);
    transfer.wait();
    destination.download_async(host_output.data(), expected.size(),
                               destination_stream);
    destination_stream.synchronize();
    require(std::memcmp(host_output.data(), expected.data(), expected.size()) == 0,
            "transfer did not preserve byte contents");
}

} // namespace

int main() {
    try {
        int device_count = 0;
        check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count < 2) {
            std::cerr << "SKIP: transfer_contract_test requires two CUDA devices\n";
            return 77;
        }

        int source_device = 0;
        int destination_device = 1;
        bool peer_supported = false;
        for (int source = 0; source < device_count && !peer_supported; ++source) {
            for (int destination = 0; destination < device_count; ++destination) {
                if (source == destination) continue;
                if (can_peer(source, destination)) {
                    source_device = source;
                    destination_device = destination;
                    peer_supported = true;
                    break;
                }
            }
        }

        constexpr std::size_t bytes = 64;
        std::vector<unsigned char> expected(bytes);
        for (std::size_t i = 0; i < bytes; ++i)
            expected[i] = static_cast<unsigned char>((i * 37u + 11u) & 0xffu);

        mc::Stream source_stream(source_device);
        mc::Stream destination_stream(destination_device);
        mc::DeviceBuffer source(bytes, source_device);
        mc::DeviceBuffer destination(bytes, destination_device);
        mc::PinnedBuffer host_source(bytes);
        mc::PinnedBuffer host_output(bytes);
        std::memcpy(host_source.data(), expected.data(), bytes);
        source.upload_async(host_source.data(), bytes, source_stream);

        dependency::PreparedTransfer staged(
            source_device, destination_device, bytes, dependency::TransferMode::staged);
        require(!staged.uses_peer(), "staged mode selected peer access");

        auto wrong_source = source.view();
        wrong_source.device = destination_device;
        rejects<std::invalid_argument>([&] {
            staged.enqueue(wrong_source, destination.view(), bytes,
                           source_stream, destination_stream);
        }, "wrong source device metadata was accepted");

        auto wrong_destination = destination.view();
        wrong_destination.device = source_device;
        rejects<std::invalid_argument>([&] {
            staged.enqueue(source.view(), wrong_destination, bytes,
                           source_stream, destination_stream);
        }, "wrong destination device metadata was accepted");

        const auto wrong_source_stream =
            mc::Stream::borrow(source_stream.get(), destination_device);
        rejects<std::invalid_argument>([&] {
            staged.enqueue(source.view(), destination.view(), bytes,
                           wrong_source_stream, destination_stream);
        }, "wrong source stream device was accepted");

        const auto wrong_destination_stream =
            mc::Stream::borrow(destination_stream.get(), source_device);
        rejects<std::invalid_argument>([&] {
            staged.enqueue(source.view(), destination.view(), bytes,
                           source_stream, wrong_destination_stream);
        }, "wrong destination stream device was accepted");

        rejects<std::out_of_range>([&] {
            staged.enqueue(source.view(), destination.view(), bytes + 1,
                           source_stream, destination_stream);
        }, "transfer larger than prepared capacity was accepted");

        copy_and_check(staged, source, destination, source_stream,
                       destination_stream, expected, host_output);

        dependency::PreparedTransfer conditional_peer(
            source_device, destination_device, bytes,
            dependency::TransferMode::peer_if_supported);
        require(conditional_peer.uses_peer() == peer_supported,
                "peer_if_supported selected an unexpected path");
        copy_and_check(conditional_peer, source, destination, source_stream,
                       destination_stream, expected, host_output);

        if (peer_supported) {
            dependency::PreparedTransfer required_peer(
                source_device, destination_device, bytes,
                dependency::TransferMode::peer_required);
            require(required_peer.uses_peer(), "peer_required did not select peer path");
            copy_and_check(required_peer, source, destination, source_stream,
                           destination_stream, expected, host_output);
        } else {
            bool rejected_for_capability = false;
            try {
                dependency::PreparedTransfer required_peer(
                    source_device, destination_device, bytes,
                    dependency::TransferMode::peer_required);
            } catch (const std::runtime_error& error) {
                rejected_for_capability =
                    std::string(error.what()).find("peer transfer required") !=
                    std::string::npos;
            }
            require(rejected_for_capability,
                    "peer_required did not reject a non-peer-capable pair");
        }

        std::cout << "transfer_contract_test: staged and conditional peer paths passed\n";
        if (peer_supported)
            std::cout << "transfer_contract_test: direct peer path passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "transfer_contract_test: " << error.what() << '\n';
        return 1;
    }
}
