#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace cellerator::execution::conditional {

using DeviceId = std::uint32_t;

inline double nonnegative_seconds(double value) {
    if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("cost must be finite and nonnegative");
    return value;
}
inline std::size_t checked_bytes_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("byte count overflow");
    return a + b;
}

struct DeviceSpec {
    DeviceId id = 0;
    std::size_t memory_bytes = 0; // Zero is a real zero budget.
    double compute_scale = 1.0;
    bool executable = true; // False for storage-only host/backing nodes.
};

struct LinkSpec {
    DeviceId source = 0, target = 0;
    double bandwidth_bytes_per_second = 1.0;
    double latency_seconds = 0.0;
    bool bidirectional = false;
    double seconds(std::size_t bytes) const {
        return bytes == 0 ? 0.0 : nonnegative_seconds(
            latency_seconds + static_cast<double>(bytes) / bandwidth_bytes_per_second);
    }
};

struct TransferRoute {
    std::vector<DeviceId> path;
    std::size_t bytes = 0;
    double seconds = 0;
};

class UnreachableDevice : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Caller-supplied directed topology. Routes charge every store-and-forward hop;
// they do not imply discovered hardware, overlap, contention or a transport.
class HardwareTopology final {
public:
    HardwareTopology(std::vector<DeviceSpec> devices, std::vector<LinkSpec> links = {}) {
        for (const auto& device : devices) {
            if (!std::isfinite(device.compute_scale) || device.compute_scale <= 0)
                throw std::invalid_argument("compute scale must be finite and positive");
            if (!devices_.emplace(device.id, device).second)
                throw std::invalid_argument("duplicate device identity");
            edges_[device.id] = {};
        }
        for (auto link : links) {
            if (!devices_.count(link.source) || !devices_.count(link.target))
                throw std::invalid_argument("link endpoint is not registered");
            if (!std::isfinite(link.bandwidth_bytes_per_second) || link.bandwidth_bytes_per_second <= 0)
                throw std::invalid_argument("bandwidth must be finite and positive");
            nonnegative_seconds(link.latency_seconds);
            edges_[link.source].push_back(link);
            if (link.bidirectional && link.source != link.target) {
                std::swap(link.source, link.target);
                edges_[link.source].push_back(link);
            }
        }
    }

    const std::map<DeviceId, DeviceSpec>& devices() const noexcept { return devices_; }

    TransferRoute route(DeviceId source, DeviceId target, std::size_t bytes) const {
        devices_.at(source);
        devices_.at(target);
        if (source == target) return {{source}, bytes, 0};
        if (bytes == 0) return {{source, target}, 0, 0};
        // Sequence preserves supplied edge order for equal-cost paths.
        using Node = std::tuple<double, std::size_t, DeviceId>;
        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> queue;
        std::map<DeviceId, double> best{{source, 0}};
        std::map<DeviceId, DeviceId> parent;
        std::size_t sequence = 0;
        queue.emplace(0, sequence++, source);
        while (!queue.empty()) {
            const auto [cost, unused, current] = queue.top();
            (void)unused;
            queue.pop();
            if (cost != best.at(current)) continue;
            if (current == target) {
                std::vector<DeviceId> path{target};
                while (path.back() != source) path.push_back(parent.at(path.back()));
                std::reverse(path.begin(), path.end());
                return {std::move(path), bytes, cost};
            }
            for (const auto& link : edges_.at(current)) {
                const auto candidate = nonnegative_seconds(cost + link.seconds(bytes));
                const auto found = best.find(link.target);
                if (found == best.end() || candidate < found->second) {
                    best[link.target] = candidate;
                    parent[link.target] = current;
                    queue.emplace(candidate, sequence++, link.target);
                }
            }
        }
        throw UnreachableDevice("no directed transfer path between selected devices");
    }

private:
    std::map<DeviceId, DeviceSpec> devices_;
    std::map<DeviceId, std::vector<LinkSpec>> edges_;
};

} // namespace cellerator::execution::conditional
