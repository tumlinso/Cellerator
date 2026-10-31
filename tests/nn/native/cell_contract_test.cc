#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/cell.hh>

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace m = cellerator::modeling;

namespace adl_programs {
struct Head {};
struct QueryHead {};
struct Request { int multiplier; };
struct Prediction { int value; };

Prediction predict(const Head&, const int& information, const Request& request) {
    return {information * request.multiplier};
}

std::string query(const Head&, const int& information, const Request& request) {
    return std::to_string(information) + ":" + std::to_string(request.multiplier);
}

Prediction predict_query(const QueryHead&, const int& information, const Request& request) {
    return {information * request.multiplier + 1};
}
} // namespace adl_programs

struct LiveProvider {
    const int* value;
    const int& operator()() const noexcept { return *value; }
};

struct StatefulProgram {
    std::unique_ptr<int> calls = std::make_unique<int>(0);
    StatefulProgram() = default;
    StatefulProgram(StatefulProgram&&) = default;
    StatefulProgram& operator=(StatefulProgram&&) = default;
    StatefulProgram(const StatefulProgram&) = delete;
    StatefulProgram& operator=(const StatefulProgram&) = delete;

    int operator()(const int& information) {
        return information + ++*calls;
    }
};

int main() {
    m::Cell<int, std::string> value(3, "owned-id");
    assert(value.data() == 3 && value.information() == 3);
    assert(value.id() == "owned-id" && value.identity() == "owned-id");

    // A plain lvalue callable is borrowed; an rvalue callable is retained.
    auto borrowed_program = [](const int& input, int increment) { return input + increment; };
    auto bound = value.bind(borrowed_program);
    assert(bound.predict(4) == 7);

    auto owned = value.bind(StatefulProgram{});
    assert(owned.predict() == 4);
    assert(owned.predict() == 5);

    // reference_wrapper and shared_ptr information stay live rather than
    // snapshotting the value when a cell or bound program is created.
    int external = 8;
    auto borrowed_data = m::make_cell(std::ref(external), std::string("borrowed"));
    auto read = borrowed_data.bind([](const int& input) { return input; });
    assert(read.predict() == 8);
    external = 11;
    assert(read.predict() == 11 && &borrowed_data.data() == &external);

    auto shared_data = std::make_shared<int>(13);
    auto shared_cell = m::make_cell(shared_data, std::string("shared"));
    shared_data.reset();
    assert(shared_cell.data() == 13); // Cell retains the source owner.

    int provided = 17;
    auto provider_cell = m::make_cell(LiveProvider{&provided}, 19);
    auto provider_read = provider_cell.bind([](const int& input) { return input; });
    assert(provider_read.predict() == 17);
    provided = 23;
    assert(provider_read.predict() == 23);

    // Callable ownership can be shared, and explicitly borrowed function
    // objects keep their identity and mutable captures.
    struct Counter {
        int calls = 0;
        int operator()(const int& input) { return input + ++calls; }
    } counter;
    auto borrowed_callable = value.bind(std::ref(counter));
    assert(borrowed_callable.predict() == 4 && borrowed_callable.predict() == 5);
    counter.calls = 0;
    auto retained_callable = value.bind_owned(counter);
    counter.calls = 20;
    assert(retained_callable.predict() == 4); // Retained copy has separate state.
    auto shared_callable = std::make_shared<Counter>();
    auto shared_bound = value.bind(shared_callable);
    shared_callable.reset();
    assert(shared_bound.predict() == 4 && shared_bound.predict() == 5);

    // Associated namespaces may provide typed model hooks without a base class
    // or a tensor dependency. Results retain their caller-defined types.
    adl_programs::Head head;
    auto adl_bound = value.bind(head);
    static_assert(std::is_same_v<decltype(adl_bound.predict(adl_programs::Request{2})),
                                 adl_programs::Prediction>);
    assert(adl_bound.predict(adl_programs::Request{2}).value == 6);
    assert(adl_bound.query(adl_programs::Request{5}) == "3:5");

    adl_programs::QueryHead query_head;
    auto query_bound = value.bind(query_head);
    assert(query_bound.predict(adl_programs::Request{2}).value == 7);
}
