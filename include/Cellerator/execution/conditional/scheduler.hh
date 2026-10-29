#pragma once

#include <Cellerator/execution/conditional/parameter_store.hh>
#include <Cellerator/execution/conditional/topology.hh>
#include <array>
#include <exception>
#include <functional>
#include <optional>
#include <set>
#include <string>

namespace cellerator::execution::conditional {

enum class MemoryCategory : std::size_t { parameter, activation, cell_state, optimizer };
struct MemoryUsage {
    std::array<std::size_t, 4> bytes{};
    std::size_t total() const {
        std::size_t result = 0;
        for (auto value : bytes) result = checked_bytes_add(result, value);
        return result;
    }
    void add(MemoryCategory category, std::size_t value) {
        auto& target = bytes[static_cast<std::size_t>(category)];
        target = checked_bytes_add(target, value);
    }
};

using CompletionEvent = std::uint64_t;
enum class ResidencyState { ready, pending };
struct ResidentCopy {
    ExpertKey key;
    DeviceId device = 0;
    std::size_t bytes = 0;
    ResidencyState state = ResidencyState::ready;
    bool executable = true;
    std::size_t pins = 0;
    std::optional<CompletionEvent> completion;
    double remaining_seconds = 0;
};

// Includes every resident and reserved in-flight copy, even pinned copies of
// obsolete revisions. other_usage excludes these copies and includes all other
// live allocations/reservations. Planning never evicts or mutates this snapshot.
struct ResidencySnapshot {
    std::uint64_t generation = 0;
    std::vector<ResidentCopy> copies;
    std::map<DeviceId, MemoryUsage> other_usage;
};

struct ExpertSpec {
    ExpertKey key;
    std::size_t parameter_bytes = 0;
    double compute_seconds_per_use = 0;
    std::vector<DeviceId> allowed_devices; // Empty permits every executable device.
};

struct ExpertDemand {
    ExpertKey key;
    DeviceId origin_device = 0;
    std::size_t activation_bytes = 0, output_bytes = 0;
    std::vector<std::size_t> selection_indices; // Original occurrences, including duplicates.
};

enum class ExecutionStrategy { local, remote_activations, move_weights, replicate };

struct ExecutionCosts {
    double compute = 0, preparation = 0, allocation = 0, synchronization = 0;
    double total() const {
        return nonnegative_seconds(nonnegative_seconds(compute) + nonnegative_seconds(preparation)
            + nonnegative_seconds(allocation) + nonnegative_seconds(synchronization));
    }
};
struct PlannedTransfer {
    MemoryCategory category = MemoryCategory::activation;
    TransferRoute route;
};
struct CostCandidate {
    ExecutionStrategy strategy = ExecutionStrategy::local;
    DeviceId execution_device = 0, parameter_source = 0;
    double estimated_seconds = 0;
    ExecutionCosts execution_costs;
    std::vector<PlannedTransfer> transfers;
    std::size_t new_parameter_bytes = 0;
    bool feasible = true;
    std::string reason;
    std::vector<std::size_t> dependencies; // Earlier replica producer decisions.
    std::vector<CompletionEvent> pending_events; // Existing in-flight copy readiness.
    std::map<DeviceId, std::size_t> activation_reservations;
};
struct DispatchDecision {
    ExpertDemand demand;
    CostCandidate chosen;
    std::vector<CostCandidate> alternatives;
};
struct ExecutionPlan {
    std::shared_ptr<const DispatchPlan> work;
    std::vector<DeviceId> selection_devices;
    std::map<ExpertKey, std::size_t> parameter_bytes;
    std::uint64_t work_revision = 0, residency_generation = 0;
    std::vector<DispatchDecision> decisions;
    std::map<DeviceId, MemoryUsage> projected_memory, peak_memory;
    std::map<DeviceId, std::size_t> peak_total_bytes;
    std::vector<ResidentCopy> proposed_replicas; // Proposals only, never ready copies.
    double estimated_seconds() const {
        double result = 0;
        for (const auto& decision : decisions) result = nonnegative_seconds(result + decision.chosen.estimated_seconds);
        return result;
    }
};

class PlanningError : public std::runtime_error {
public:
    ExpertDemand demand;
    std::vector<CostCandidate> candidates;
    PlanningError(ExpertDemand value, std::vector<CostCandidate> alternatives, const char* message)
        : std::runtime_error(message), demand(std::move(value)), candidates(std::move(alternatives)) {}
};

// Deterministic sequential-horizon planner mirroring execution/planning.py.
// It consumes the existing DispatchPlan, rather than defining another routing
// API. Costs are caller estimates, not measured performance. A move is a
// temporary execution copy; it never deletes its immutable source revision.
class PlacementPlanner final {
public:
    using CostModel = std::function<ExecutionCosts(const ExpertDemand&, DeviceId, ExecutionStrategy)>;
    using OriginDevice = std::function<DeviceId(const Origin&)>;
    using RepeatHints = std::map<std::pair<ExpertKey, DeviceId>, double>;

    PlacementPlanner(HardwareTopology topology, std::vector<ExpertSpec> experts,
                     ResidencySnapshot residency, CostModel costs = {}, double retention_penalty = 0)
        : topology_(std::move(topology)), snapshot_(std::move(residency)), costs_(std::move(costs)),
          penalty_(nonnegative_seconds(retention_penalty)) {
        for (auto& expert : experts) {
            nonnegative_seconds(expert.compute_seconds_per_use);
            for (auto device : expert.allowed_devices) topology_.devices().at(device);
            if (!experts_.emplace(expert.key, std::move(expert)).second)
                throw std::invalid_argument("duplicate expert revision specification");
        }
        for (const auto& pair : topology_.devices()) baseline_[pair.first] = {};
        for (const auto& pair : snapshot_.other_usage) {
            topology_.devices().at(pair.first);
            baseline_[pair.first] = pair.second;
        }
        std::set<std::pair<ExpertKey, DeviceId>> seen;
        for (const auto& copy : snapshot_.copies) {
            topology_.devices().at(copy.device);
            if (!seen.emplace(copy.key, copy.device).second)
                throw std::invalid_argument("duplicate expert/device residency");
            if (copy.state == ResidencyState::pending && !copy.completion)
                throw std::invalid_argument("pending copy needs an explicit completion event");
            nonnegative_seconds(copy.remaining_seconds);
            const auto spec = experts_.find(copy.key);
            if (spec != experts_.end() && copy.bytes != spec->second.parameter_bytes)
                throw std::invalid_argument("resident size differs from exact expert revision");
            baseline_.at(copy.device).add(MemoryCategory::parameter, copy.bytes);
        }
        for (const auto& pair : baseline_)
            if (pair.second.total() > topology_.devices().at(pair.first).memory_bytes)
                throw std::invalid_argument("resident and in-flight bytes exceed capacity");
    }

    ExecutionPlan plan(const DispatchPlan& work, OriginDevice origin_device,
                       std::size_t activation_bytes_per_row, std::size_t output_bytes_per_row,
                       std::uint64_t work_revision = 0, const RepeatHints& repeats = {},
                       bool allow_replication = true) const {
        if (!origin_device) throw std::invalid_argument("origin-device mapping is required");
        ExecutionPlan result;
        result.work = std::make_shared<const DispatchPlan>(work);
        result.work_revision = work_revision;
        result.residency_generation = snapshot_.generation;
        for (const auto& pair : experts_) result.parameter_bytes.emplace(pair.first, pair.second.parameter_bytes);
        result.projected_memory = result.peak_memory = baseline_;
        for (const auto& pair : baseline_) result.peak_total_bytes[pair.first] = pair.second.total();
        std::vector<ExpertDemand> demands;
        std::map<std::pair<ExpertKey, DeviceId>, std::size_t> grouped;
        for (std::size_t i = 0; i < work.size(); ++i) {
            const auto& selection = work.selections()[i];
            const auto device = origin_device(selection.origin);
            topology_.devices().at(device);
            result.selection_devices.push_back(device);
            const auto identity = std::make_pair(selection.expert, device);
            auto entry = grouped.emplace(identity, demands.size());
            if (entry.second) demands.push_back({selection.expert, device, 0, 0, {}});
            auto& demand = demands[entry.first->second];
            demand.activation_bytes = checked_bytes_add(demand.activation_bytes, activation_bytes_per_row);
            demand.output_bytes = checked_bytes_add(demand.output_bytes, output_bytes_per_row);
            demand.selection_indices.push_back(i);
        }
        struct Available { ResidentCopy copy; std::optional<std::size_t> producer; };
        std::map<std::pair<ExpertKey, DeviceId>, Available> copies;
        for (const auto& copy : snapshot_.copies) copies.emplace(std::make_pair(copy.key, copy.device), Available{copy, {}});
        for (const auto& demand : demands) {
            const auto spec = experts_.find(demand.key);
            if (spec == experts_.end()) throw PlanningError(demand, {}, "selected expert revision is unregistered");
            const auto& expert = spec->second;
            auto hint = repeats.find({demand.key, demand.origin_device});
            const auto repeat = hint == repeats.end() ? 1.0 : nonnegative_seconds(hint->second);
            if (repeat < 1) throw std::invalid_argument("repeat hint must be at least one");
            std::vector<CostCandidate> candidates;
            auto candidate = [&](ExecutionStrategy strategy, DeviceId device, const Available& source,
                                 bool new_copy) {
                const auto& hardware = topology_.devices().at(device);
                if (!hardware.executable || (!expert.allowed_devices.empty() &&
                    std::find(expert.allowed_devices.begin(), expert.allowed_devices.end(), device) == expert.allowed_devices.end())) return;
                CostCandidate value;
                value.strategy = strategy;
                value.execution_device = device;
                value.parameter_source = source.copy.device;
                value.new_parameter_bytes = new_copy ? expert.parameter_bytes : 0;
                if (source.producer) value.dependencies.push_back(*source.producer);
                if (source.copy.state == ResidencyState::pending && source.copy.completion)
                    value.pending_events.push_back(*source.copy.completion);
                value.activation_reservations[device] = checked_bytes_add(demand.activation_bytes, demand.output_bytes);
                if (strategy == ExecutionStrategy::remote_activations)
                    value.activation_reservations[demand.origin_device] = demand.output_bytes;
                for (const auto& reservation : value.activation_reservations) {
                    auto required = checked_bytes_add(result.projected_memory.at(reservation.first).total(), reservation.second);
                    if (reservation.first == device) required = checked_bytes_add(required, value.new_parameter_bytes);
                    if (required > topology_.devices().at(reservation.first).memory_bytes) {
                        value.feasible = false; value.reason = "insufficient memory";
                    }
                }
                if (value.feasible) {
                    try {
                        if (new_copy)
                            value.transfers.push_back({MemoryCategory::parameter,
                                topology_.route(source.copy.device, device, expert.parameter_bytes)});
                        if (strategy == ExecutionStrategy::remote_activations) {
                            value.transfers.push_back({MemoryCategory::activation,
                                topology_.route(demand.origin_device, device, demand.activation_bytes)});
                            value.transfers.push_back({MemoryCategory::activation,
                                topology_.route(device, demand.origin_device, demand.output_bytes)});
                        }
                    } catch (const UnreachableDevice&) {
                        value.feasible = false; value.reason = "unreachable transfer path";
                    }
                }
                if (value.feasible) {
                    value.execution_costs = costs_ ? costs_(demand, device, strategy) : ExecutionCosts{
                        expert.compute_seconds_per_use * demand.selection_indices.size() / hardware.compute_scale, 0, 0, 0};
                    double communication = 0;
                    for (const auto& transfer : value.transfers) communication += transfer.route.seconds;
                    const auto execution = value.execution_costs.total();
                    value.estimated_seconds = nonnegative_seconds(source.copy.remaining_seconds +
                        (strategy == ExecutionStrategy::replicate
                          ? communication + execution * repeat + penalty_ * value.new_parameter_bytes
                          : (communication + execution) * repeat));
                } else value.estimated_seconds = std::numeric_limits<double>::infinity();
                candidates.push_back(std::move(value));
            };
            for (const auto& pair : copies) {
                const auto& source = pair.second;
                if (pair.first.first != demand.key || !source.copy.executable) continue;
                candidate(source.copy.device == demand.origin_device ? ExecutionStrategy::local : ExecutionStrategy::remote_activations,
                          source.copy.device, source, false);
            }
            if (!copies.count({demand.key, demand.origin_device})) {
                for (const auto& pair : copies) {
                    if (pair.first.first != demand.key) continue;
                    candidate(ExecutionStrategy::move_weights, demand.origin_device, pair.second, true);
                    if (allow_replication) candidate(ExecutionStrategy::replicate, demand.origin_device, pair.second, true);
                }
            }
            auto chosen = candidates.end();
            for (auto item = candidates.begin(); item != candidates.end(); ++item)
                if (item->feasible && (chosen == candidates.end() || item->estimated_seconds < chosen->estimated_seconds)) chosen = item;
            if (chosen == candidates.end()) throw PlanningError(demand, std::move(candidates), "no feasible placement for the exact selected revision");
            const auto selected = *chosen;
            for (const auto& reservation : selected.activation_reservations) {
                auto current = result.projected_memory.at(reservation.first);
                if (reservation.first == selected.execution_device) current.add(MemoryCategory::parameter, selected.new_parameter_bytes);
                current.add(MemoryCategory::activation, reservation.second);
                auto& peak = result.peak_memory.at(reservation.first);
                for (std::size_t i = 0; i < peak.bytes.size(); ++i) peak.bytes[i] = std::max(peak.bytes[i], current.bytes[i]);
                auto& total = result.peak_total_bytes.at(reservation.first);
                total = std::max(total, current.total());
            }
            if (selected.strategy == ExecutionStrategy::replicate) {
                result.projected_memory.at(selected.execution_device).add(MemoryCategory::parameter, expert.parameter_bytes);
                ResidentCopy replica{demand.key, selected.execution_device, expert.parameter_bytes,
                                     ResidencyState::pending, true, 0, {}, 0};
                result.proposed_replicas.push_back(replica);
                copies[{demand.key, selected.execution_device}] = {replica, result.decisions.size()};
            }
            result.decisions.push_back({demand, selected, std::move(candidates)});
        }
        return result;
    }

private:
    HardwareTopology topology_;
    ResidencySnapshot snapshot_;
    std::map<ExpertKey, ExpertSpec> experts_;
    std::map<DeviceId, MemoryUsage> baseline_;
    CostModel costs_;
    double penalty_;
};

enum class TaskState { ready, running, yielded, completed, failed };
struct TaskPoll {
    bool completed = true;
    std::optional<CompletionEvent> wait_for;
    static TaskPoll done() { return {}; }
    static TaskPoll yield(CompletionEvent event) { return {false, event}; }
};

// One bounded sequential queue per plan, matching its memory/cost horizon.
// Each callback owns a resumable transfer/execute/combine state machine. A yield
// is not a replay request: the next call resumes the SAME decision. Completion
// means all its device reads/writes finished and any proposed replica is usable.
// Caller serializes queue access and keeps it alive while async work is active.
class PlannedExecution final {
public:
    using Callback = std::function<TaskPoll(std::size_t, const DispatchDecision&,
                                           const DispatchPlan&, const ParameterLease&)>;
    PlannedExecution(ExecutionPlan plan, ParameterStore& store, std::uint64_t work_revision,
                     std::uint64_t residency_generation, std::size_t max_decisions = 4096)
        : plan_(std::move(plan)) {
        if (!plan_.work || work_revision != plan_.work_revision || residency_generation != plan_.residency_generation)
            throw std::invalid_argument("plan structure/residency snapshot is stale or absent");
        if (plan_.decisions.size() > max_decisions) throw std::length_error("execution queue capacity exceeded");
        if (plan_.selection_devices.size() != plan_.work->size()) throw std::invalid_argument("missing occurrence devices");
        std::vector<bool> seen(plan_.work->size(), false);
        std::set<std::pair<ExpertKey, DeviceId>> demand_groups;
        std::size_t previous_first = 0;
        for (std::size_t i = 0; i < plan_.decisions.size(); ++i) {
            const auto& decision = plan_.decisions[i];
            if (!decision.chosen.feasible || decision.demand.selection_indices.empty())
                throw std::invalid_argument("infeasible or empty planned decision");
            if (!demand_groups.emplace(decision.demand.key, decision.demand.origin_device).second ||
                (i && decision.demand.selection_indices.front() <= previous_first))
                throw std::invalid_argument("plan splits or reorders selected demand groups");
            previous_first = decision.demand.selection_indices.front();
            for (auto dependency : decision.chosen.dependencies) {
                if (dependency >= i) throw std::invalid_argument("dependency must precede its consumer");
                const auto& producer = plan_.decisions[dependency];
                if (producer.chosen.strategy != ExecutionStrategy::replicate || producer.demand.key != decision.demand.key ||
                    producer.chosen.execution_device != decision.chosen.parameter_source)
                    throw std::invalid_argument("dependency is not the exact replica producer");
            }
            for (auto event : decision.chosen.pending_events) known_events_.insert(event);
            std::size_t previous = 0;
            bool first = true;
            for (auto index : decision.demand.selection_indices) {
                if (index >= seen.size() || seen[index] || (!first && index <= previous) ||
                    plan_.work->selections()[index].expert != decision.demand.key ||
                    plan_.selection_devices[index] != decision.demand.origin_device)
                    throw std::invalid_argument("plan changes selected occurrence identity/order");
                seen[index] = true; previous = index; first = false;
            }
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end()) throw std::invalid_argument("plan omits selected work");
        for (auto& lease : store.snapshot(plan_.work->experts())) {
            if (lease.buffers().parameters.bytes != plan_.parameter_bytes.at(lease.key()))
                throw std::invalid_argument("logical parameter size differs from planned residency");
            leases_.emplace(lease.key(), std::move(lease));
        }
        states_.assign(plan_.decisions.size(), TaskState::ready);
        waits_.resize(states_.size());
    }

    PlannedExecution(const PlannedExecution&) = delete;
    PlannedExecution& operator=(const PlannedExecution&) = delete;

    // Mark a backend event successful. Failed events use fail(), preventing all
    // remaining work; the caller must drain already submitted device activity.
    void signal(CompletionEvent event) {
        if (!known_events_.count(event) && (next_ >= waits_.size() || waits_[next_] != event))
            throw std::invalid_argument("event does not belong to pending execution work");
        signaled_.insert(event);
    }

    std::size_t advance(const Callback& callback, std::size_t max_callbacks = 1) {
        if (!callback) throw std::invalid_argument("execution callback is required");
        if (closed_) throw std::logic_error("execution is closed");
        if (failure_) std::rethrow_exception(failure_);
        std::size_t called = 0;
        while (next_ < states_.size() && called < max_callbacks) {
            const auto& decision = plan_.decisions[next_];
            bool waiting = waits_[next_] && !signaled_.count(*waits_[next_]);
            for (auto event : decision.chosen.pending_events) waiting |= !signaled_.count(event);
            if (waiting) { states_[next_] = TaskState::yielded; break; }
            for (auto dependency : decision.chosen.dependencies)
                if (states_[dependency] != TaskState::completed) throw std::logic_error("unsatisfied producer dependency");
            if (waits_[next_] && !known_events_.count(*waits_[next_])) signaled_.erase(*waits_[next_]);
            waits_[next_].reset();
            states_[next_] = TaskState::running;
            ++called;
            try {
                const auto polled = callback(next_, decision, *plan_.work, leases_.at(decision.demand.key));
                if (polled.completed && polled.wait_for) throw std::invalid_argument("completed task cannot wait");
                if (!polled.completed && !polled.wait_for) throw std::invalid_argument("yield requires a completion event");
                if (polled.completed) states_[next_++] = TaskState::completed;
                else { states_[next_] = TaskState::yielded; waits_[next_] = polled.wait_for; break; }
            } catch (...) {
                failure_ = std::current_exception(); states_[next_] = TaskState::failed;
                throw;
            }
        }
        return called;
    }

    void fail(std::exception_ptr error) {
        if (!error) throw std::invalid_argument("failure needs an exception");
        if (done() || closed_) throw std::logic_error("execution is already complete");
        failure_ = std::move(error); states_[next_] = TaskState::failed;
    }
    bool done() const noexcept { return next_ == states_.size(); }
    std::size_t completed_decisions() const noexcept { return next_; }
    const std::vector<TaskState>& states() const noexcept { return states_; }
    const ExecutionPlan& plan() const noexcept { return plan_; }

    // Explicit backend-drained boundary. No release while normal work remains.
    // On failure the caller guarantees outstanding transport/CUDA work drained.
    void close(bool failed_backend_drained = false) {
        if (!done() && !(failure_ && failed_backend_drained))
            throw std::logic_error("execution has unfinished backend work");
        leases_.clear(); closed_ = true;
    }

private:
    ExecutionPlan plan_;
    std::map<ExpertKey, ParameterLease> leases_;
    std::vector<TaskState> states_;
    std::vector<std::optional<CompletionEvent>> waits_;
    std::set<CompletionEvent> signaled_;
    std::set<CompletionEvent> known_events_;
    std::size_t next_ = 0;
    std::exception_ptr failure_;
    bool closed_ = false;
};

} // namespace cellerator::execution::conditional
