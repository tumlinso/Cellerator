#pragma once

#include <Cellerator/execution/planning/conditional_placement.hh>
#include <exception>
#include <functional>
#include <optional>
#include <set>

namespace cellerator::execution::conditional {

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
